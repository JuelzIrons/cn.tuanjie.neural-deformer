import torch
import torch.nn as nn

DIM_QUATERNION = 4
DIM_VERTEX_OFF = 3

class DeformerNMM(nn.Module):
    def __init__(self,
                 centers: int,
                 joints: int,
                 vertices: int,
                 init_centers: torch.Tensor | None = None,
                 init_betas: torch.Tensor | float | None = None,
                 **kwargs):
        
        super().__init__()

        self.rbf_layer = JointsBatchRBFLayer(dim_in=DIM_QUATERNION,
                                             dim_out=centers,
                                             joints=joints,
                                             init_centers=init_centers,
                                             init_betas=init_betas,
                                             **kwargs)
        
        self.trunk = nn.Sequential(
            nn.LayerNorm(centers * joints),
            nn.Linear(centers * joints, centers * joints),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(centers * joints, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, DIM_VERTEX_OFF * vertices, bias=False),
        )

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.vertex_count = vertices


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.rbf_layer(x)
        x = x.view(batch_size, -1)
        y = self.trunk(x)
        y = y.view(batch_size, -1, DIM_VERTEX_OFF)
        return y
    

    def loss_fn(self, y_gt: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        batch_size = y_gt.size(0)
        return self.mse_loss(y_gt, y_pred) / batch_size / self.vertex_count


class JointsBatchRBFLayer(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 joints: int,
                 init_centers: torch.Tensor | None = None,
                 init_betas: torch.Tensor | float | None = None,
                 optimize_centers: bool = True,
                 optimize_betas: bool = True,
                 **kwargs):
        
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        
        if init_centers is not None:
            assert init_centers.shape[-3:] == (joints, dim_out, dim_in), f"The shape of init_centers is expected to be (..., {joints}, {dim_out}, {dim_in}), but got {init_centers.shape}"
            self.init_centers = init_centers
        else:
            self.init_centers = torch.empty(joints, dim_out, dim_in)
            nn.init.uniform_(self.init_centers, -1, 1)

        self.centers = nn.Parameter(self.init_centers, requires_grad=optimize_centers)

        if init_betas is not None:
            if isinstance(init_betas, torch.Tensor):
                assert init_betas.shape[-2:] == (joints, dim_out), f"The shape of init_betas is expected to be (..., {joints}, {dim_out}), but got {init_betas.shape}"
            else:
                init_betas = torch.ones((joints, dim_out)) * init_betas
            self.init_betas = init_betas
        else:
            self.init_betas = torch.empty(joints, dim_out)
            nn.init.uniform_(self.init_betas, 0, 1)

        self.betas = nn.Parameter(self.init_betas, requires_grad=optimize_betas)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = x.unsqueeze(-2) - self.centers
        return torch.exp(-self.betas * torch.sum(dx ** 2, dim=-1))
    

if __name__ == '__main__':
    rbf = JointsBatchRBFLayer(4, 32, 17).cuda()
    x = torch.randn((10, 17, 4)).cuda()
    y = rbf(x).cpu()
    print(y.shape)