import os
import struct
import glob
import torch

from functools import reduce
from operator import mul
from torch.utils.data import Dataset, DataLoader

class DeformerDataSource:
    def __init__(self, data_dir: str, valid_split: float = 0.1):
        self.data_dir = data_dir
        self.valid_split = valid_split
        self.loaded = False
        return


    def load(self):
        if not self.loaded:
            print('[Loading] ...')
            self.load_meta()
            self.load_x()
            self.load_y()
            assert len(self.joint_rotations) == len(self.vertex_offsets), f"The length of feature ({len(self.joint_rotations)}) don't match that of targets ({len(self.vertex_offsets)})."

            self.split_train_valid()
            self.loaded = True
        return


    def load_meta(self):
        meta_path = os.path.join(self.data_dir, "meta.txt")
        assert os.path.exists(meta_path) and os.path.isfile(meta_path), f"File 'meta.txt' not found in dataset folder '{self.data_dir}'."

        with open(meta_path) as f:
            print('[Loading] ' + meta_path)
            for line in f.readlines():
                key, value = line.split(':')[0].strip(), line.split(':')[1].strip()
                if key == 'JointCount':
                    self.joint_count = int(value)
                elif key == 'UniqueVertexCount':
                    self.vertex_count = int(value)



    def load_x(self):
        joint_dir = os.path.join(self.data_dir, 'x')
        assert os.path.exists(joint_dir) and os.path.isdir(joint_dir), f"Directory 'x' not found in dataset folder '{self.data_dir}'."

        self.joint_rotations = self._load_float_bin(joint_dir, (-1, self.joint_count, 4))


    def load_y(self):
        vertex_dir = os.path.join(self.data_dir, 'y')
        assert os.path.exists(vertex_dir) and os.path.isdir(vertex_dir), f"Directory 'y' not found in dataset folder '{self.data_dir}'."

        vertex_offsets = self._load_float_bin(vertex_dir, (-1, self.vertex_count, 3))
        
        min_offset = torch.min(vertex_offsets)
        max_offset = torch.max(vertex_offsets)
        offset_range = max_offset - min_offset if min_offset != max_offset else 1e-7

        self.vertex_offsets = 2 * (vertex_offsets - min_offset) / offset_range - 1
        print('[Loading] scale range of y from [%.3f, %.3f] ' % (min_offset.item(), max_offset.item()) + 
              'to [%.3f, %.3f]' % (torch.min(self.vertex_offsets).item(), torch.max(self.vertex_offsets).item()))


    def split_train_valid(self):
        n = len(self.joint_rotations)
        temp = torch.linspace(0, 1, n)[torch.randperm(n)]
        self.train_indices = torch.where(temp >= self.valid_split)[0]
        self.valid_indices = torch.where(temp < self.valid_split)[0]


    def get_split_data(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert split == 'train' or split == 'valid', f"The split argument should be either 'train' or 'valid', but got '{split}'."
        indices = self.train_indices if split == 'train' else self.valid_indices
        return self.joint_rotations[indices], self.vertex_offsets[indices]


    @staticmethod
    def _load_float_bin(dir: str, shape: tuple[int]) -> torch.Tensor:
        files = glob.glob(os.path.join(dir, '*.bin'))
        data = []

        for file in files:
            with open(file, 'rb') as f:
                print('[Loading] ' + os.path.split(file)[1])
                floats = os.path.getsize(file) // 4
                chunk_size = reduce(mul, shape[1:])
                array = [torch.Tensor(a).to(torch.float32) for a in struct.iter_unpack('f' * chunk_size, f.read(floats * 4))]
                data.append(torch.stack(array).reshape(shape).to(torch.float32))

        return torch.cat(data, dim=0)


class DeformerDataset(Dataset):
    def __init__(self, 
                 source: DeformerDataSource,
                 split: str):
        
        super().__init__()

        self.source = source
        self.source.load()
        
        self.joint_count = self.source.joint_count
        self.vertex_count = self.source.vertex_count
        
        self.joint_rotations, self.vertex_offsets = self.source.get_split_data(split)

    def __len__(self):
        return len(self.joint_rotations)
    

    def __getitem__(self, idx):
        x = self.joint_rotations[idx]
        y = self.vertex_offsets[idx]
        return x, y


if __name__ == '__main__':
    source = DeformerDataSource(data_dir="dataset/TG_Outfit_01")

    train_dataset = DeformerDataset(source, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    print(len(train_dataset))
    for i, batch in enumerate(train_dataloader):
        x, y = batch
        print(x.shape, y.shape)
        break
    
    valid_dataset = DeformerDataset(source, 'valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    print(len(valid_dataset))
    for i, batch in enumerate(valid_dataloader):
        x, y = batch
        print(x.shape, y.shape)
        break