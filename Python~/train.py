import os, sys
import datetime, time
import argparse
import json
import traceback
import copy
import logging

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import Adam, lr_scheduler
    from tensorboardX import SummaryWriter
    from torch_kmeans import KMeans
    import matplotlib.pyplot as plt

    project_path = os.path.abspath(os.path.dirname(__file__))
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    
    from neural_deformer_trainer.data import DeformerDataset, DeformerDataSource
    from neural_deformer_trainer.model import DeformerNMM

except Exception as e:
    error_info = {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc()
    }
    print("[Exception] %s\n" % json.dumps(error_info), file=sys.stderr)
    sys.exit(1)

def parse_opt() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--import-dir", "-i",   type=str)
    arg_parser.add_argument("--report-dir", "-r",   type=str)
    arg_parser.add_argument("--export-dir", "-o",   type=str)
    arg_parser.add_argument("--no-report",          action="store_true")
    arg_parser.add_argument("--run-name",   "-n",   type=str,   default=None)
    arg_parser.add_argument("--gpu",                type=int,   default=0)
    arg_parser.add_argument("--seed",       "-s",   type=int,   default=0)
    arg_parser.add_argument("--centers",            type=int,   default=32)
    arg_parser.add_argument("--centers-init",       type=str,   default="kmeans", choices=["kmeans", "random"])
    arg_parser.add_argument("--betas-init",         type=float, default=0.5)
    arg_parser.add_argument("--batch-size", "-b",   type=int,   default=8)
    arg_parser.add_argument("--no-shuffle",         action="store_true")
    arg_parser.add_argument("--epochs",     "-e",   type=int,   default=300)
    arg_parser.add_argument("--lr",                 type=float, default=5e-3)

    opt, _ = arg_parser.parse_known_args()

    if torch.cuda.is_available() and opt.gpu < torch.cuda.device_count():
        opt.device = torch.device(f'cuda:{opt.gpu}')
    else:
        opt.device = torch.device('cpu')
    opt.report = not opt.no_report
    opt.shuffle = not opt.no_shuffle
    
    return opt


def validate_opt(opt: argparse.Namespace):
    # import-dir
    assert isinstance(opt.import_dir, str) and opt.import_dir, "'--import-dir' must be a non-empty string"
    assert os.path.exists(opt.import_dir) and os.path.isdir(opt.import_dir), f"'--import-dir' path '{opt.import_dir}' does not exist or is not a directory"
    # report-dir
    assert isinstance(opt.report_dir, str) and opt.report_dir, "'--report-dir' must be a non-empty string"
    assert os.path.exists(opt.report_dir) and os.path.isdir(opt.report_dir), f"'--report-dir' path '{opt.report_dir}' does not exist or is not a directory"
    # export-dir
    assert isinstance(opt.export_dir, str) and opt.export_dir, "'--export-dir' must be a non-empty string"
    assert os.path.exists(opt.export_dir) and os.path.isdir(opt.export_dir), f"'--export-dir' path '{opt.export_dir}' does not exist or is not a directory"
    # run-name
    if opt.run_name is not None:
        assert isinstance(opt.run_name, str), "'--run-name' must be a string"
        assert len(opt.run_name.strip()) > 0, "'--run-name' must not be an empty string"
    # gpu
    assert isinstance(opt.gpu, int) and opt.gpu >= 0, "'--gpu' must be a non-negative integer"
    # seed
    assert isinstance(opt.seed, int), "'--seed' must be an integer"
    # centers
    assert isinstance(opt.centers, int) and opt.centers > 0, "'--centers' must be a positive integer"
    # centers-init
    assert opt.centers_init in ["kmeans", "random"], "'--centers-init' must be 'kmeans' or 'random'"
    # betas-init
    assert isinstance(opt.betas_init, float), "'--betas-init' must be a float"
    # batch-size
    assert isinstance(opt.batch_size, int) and opt.batch_size > 0, "'--batch-size' must be a positive integer"
    # epochs
    assert isinstance(opt.epochs, int) and opt.epochs > 0 and opt.epochs < 1000, "'--epochs' must be a positive integer in range (0, 1000)"
    # lr
    assert isinstance(opt.lr, float) and opt.lr > 0, "'--lr' must be a positive float"


def train(opt: argparse.Namespace):
    print("[Summary] init training, PyTorch version: %s" % torch.__version__)
    
    dt_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if opt.seed >= 0:
        torch.manual_seed(opt.seed)

    source = DeformerDataSource(data_dir=opt.import_dir)
    train_dataset = DeformerDataset(source, 'train')
    valid_dataset = DeformerDataset(source, 'valid')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=0, pin_memory=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    if opt.centers_init == 'kmeans':
        samples = train_dataset.joint_rotations.permute(1, 0, 2).to(device=opt.device)
        km_model = KMeans(n_clusters=opt.centers, verbose=False)
        init_centers = km_model(samples).centers
        init_betas = opt.betas_init
    else:
        init_centers = None
        init_betas = None
    
    model = DeformerNMM(opt.centers, source.joint_count, source.vertex_count, init_centers, init_betas).to(device=opt.device)
    model_name = opt.run_name or (f"C{opt.centers}({opt.centers_init[0].upper()})_" + 
                              f"B{init_betas or str.format('(%s)' % opt.centers_init[0].upper())}_" + 
                              f"BS{opt.batch_size}_" + 
                              f"LR{opt.lr}")
    model_name = f"{dt_now}_{model_name}"

    optimizer = Adam(model.parameters(), lr=opt.lr)
    lr_decay_fn = lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)

    loss_history = {'train': [], 'valid': []}
    best_valid_loss = 1e10
    best_valid_epoch = 0
    best_state_dict = None
    
    if opt.report:
        reporter = SummaryWriter(logdir=os.path.join(opt.report_dir, model_name))
    else:
        os.makedirs(os.path.join(opt.report_dir, model_name), exist_ok=False)

    t1 = time.time()
    for epoch in range(opt.epochs):
        train_loss = 0
        model.train()
        for _, (x, y) in enumerate(train_dataloader):
            x = x.to(device=opt.device)
            y = y.to(device=opt.device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = model.loss_fn(y, y_pred)

            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item() * x.shape[0]
        train_loss /= len(train_dataset)

        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(valid_dataloader):
                x = x.to(device=opt.device)
                y = y.to(device=opt.device)

                y_pred = model(x)
                loss = model.loss_fn(y, y_pred)

                valid_loss += loss.cpu().item() * x.shape[0]
        valid_loss /= len(valid_dataset)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if opt.report:
            reporter.add_scalars("loss", {"train_loss": train_loss, "valid_loss": valid_loss}, epoch)
            reporter.add_scalar("train_loss", train_loss, epoch)
            reporter.add_scalar("valid_loss", valid_loss, epoch)
            reporter.add_scalar("lr", lr_decay_fn.get_last_lr()[0], epoch)
            reporter.flush()

        loss_history['train'].append(train_loss)
        loss_history['valid'].append(valid_loss)
        print("[Epoch %03d/%03d]  train_loss: %.6f  valid_loss: %.6f  lr: %.8f  %s" % 
              (epoch + 1, opt.epochs, train_loss, valid_loss, lr_decay_fn.get_last_lr()[0], "(*)" if best_valid_epoch == epoch else ""))

        lr_decay_fn.step()
        
    t2 = time.time()
    if opt.report:
        reporter.close()
    print("[Summary] duration: %.1fs; best valid loss: %.6f @ epoch %d/%d" % (t2 - t1, best_valid_loss, best_valid_epoch + 1, opt.epochs))
    
    plt.plot(loss_history['train'], label="Train Loss")
    plt.plot(loss_history['valid'], label="Valid Loss")
    plt.legend(loc='best')
    plt_path = os.path.join(opt.report_dir, model_name, "loss.png")
    plt.savefig(plt_path)
    print("[Summary] loss figure saved to path: %s" % plt_path)

    
    logging.getLogger("torch.onnx._internal.exporter._registration").setLevel(logging.ERROR)
    model.load_state_dict(best_state_dict)
    onnx_file = f"{model_name}_{best_valid_loss:.6f}.onnx"
    onnx_path = os.path.join(opt.export_dir, onnx_file)
    torch.onnx.export(
        model=model.cpu(),
        args=(source.joint_rotations[0:1]),
        f=onnx_path,
        input_names=["joint_rotations"],
        output_names=["vertex_offsets"],
        verbose=False,
        export_params=True,
        **({"dynamo": True, "optimize": True, "external_data": False} if ([int(x) for x in torch.__version__.split('.')[:2]] >= [2, 5]) else {})
    )
    print("[Summary] model saved to path: %s" % onnx_path)


if __name__ == '__main__':
    try:
        opt = parse_opt()
        validate_opt(opt)
        train(opt)
    except Exception as e:
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        print("[Exception] %s\n" % json.dumps(error_info), file=sys.stderr)
        sys.exit(1)