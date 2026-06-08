import torch
import os
import os.path as osp
from torch_geometric.data import Dataset, DataLoader


class PairedSnapshotDataset(Dataset):
    def __init__(self, root, name,split='train'):
        super().__init__(root, transform=None, pre_transform=None)
        self.name = name
        self.data_dir = osp.join(root, 'processed', name) # e.g. data/processed/reddit
        print(self.data_dir)
        
        # Count available snapshots
        if not osp.exists(self.data_dir):
            raise FileNotFoundError(f"Directory {self.data_dir} not found. Run preprocessing first!")
            
        self.files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('snap_') and f.endswith('.pt')])
        total_pairs = len(self.files) - 1

        split_idx = int(total_pairs * 0.85) # 85% Train, 15% Val
        
        if split == 'train':
            self.valid_indices = range(0, split_idx)
        elif split == 'val':
            self.valid_indices = range(split_idx, total_pairs)

    def len(self):
        return len(self.valid_indices)

    def get(self, idx):

        real_idx = self.valid_indices[idx]
        # Load Pair: (Previous, Current)
        # Prev is Condition, Curr is Target
        prev_data = torch.load(osp.join(self.data_dir, f'snap_{real_idx}.pt'))
        curr_data = torch.load(osp.join(self.data_dir, f'snap_{real_idx+1}.pt'))
        
        # Add batching fix
        prev_data.batch_nodes = torch.arange(prev_data.num_nodes)
        curr_data.batch_nodes = torch.arange(curr_data.num_nodes)
        
        return prev_data, curr_data

def get_paired_loader(root_dir, name, batch_size=1, split='train',num_workers=0):
    dataset = PairedSnapshotDataset(root_dir, name,split=split)
    # Important: follow_batch helps PyG merge graph attributes correctly
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), follow_batch=['batch_nodes'],num_workers=num_workers)
    return loader