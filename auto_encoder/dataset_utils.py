"""
Dataset module for Counterfactual MRI Reconstruction
Handles paired 0.5T and 1.5T MRI data from k-space
"""

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict


def kspace_to_image_rss(kspace: np.ndarray) -> np.ndarray:
    """
    Convert k-space to image using Root Sum of Squares (RSS) coil combination.
    Optimized with vectorization.
    
    Args:
        kspace: Complex k-space data with shape (H, W, num_slices, num_coils) 
                OR (H, W, num_coils) for single slice
    
    Returns:
        RSS combined magnitude image with shape (num_slices, H, W) 
        OR (H, W) for single slice
    """
    # Vectorized implementation
    # Calculate magnitude squared: |z|^2 = Re(z)^2 + Im(z)^2
    # But np.abs() handles complex numbers correctly
    
    if kspace.ndim == 3:
        # Shape: (H, W, num_coils) -> Output: (H, W)
        return np.sqrt(np.sum(np.abs(kspace)**2, axis=-1)).astype(np.float32)
        
    elif kspace.ndim == 4:
        # Shape: (H, W, num_slices, num_coils) -> Output: (num_slices, H, W)
        # Sum over coils (axis 3)
        rss = np.sqrt(np.sum(np.abs(kspace)**2, axis=3))
        # Transpose from (H, W, num_slices) to (num_slices, H, W)
        return rss.transpose(2, 0, 1).astype(np.float32)
    
    else:
        raise ValueError(f"Unsupported kspace shape: {kspace.shape}")


def normalize_image(image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """
    Normalize image to [0, 1] range using percentile-based normalization.
    
    Args:
        image: Input image
        percentile: Upper percentile for clipping
    
    Returns:
        Normalized image in [0, 1] range
    """
    # Clip extreme values
    upper = np.percentile(image, percentile)
    lower = np.percentile(image, 100 - percentile)
    
    image = np.clip(image, lower, upper)
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    return image.astype(np.float32)


def center_crop(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop image to target size.
    If image is smaller than target size, pad it with zeros first.
    
    Args:
        image: Input image with shape (H, W) or (N, H, W)
    
    Returns:
        Cropped/Padded image of size target_size
    """
    if len(image.shape) == 2:
        H, W = image.shape
        th, tw = target_size
        
        # Pad if needed
        pad_h = max(th - H, 0)
        pad_w = max(tw - W, 0)
        
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
            
        # Update shape
        H, W = image.shape
        
        start_h = (H - th) // 2
        start_w = (W - tw) // 2
        return image[start_h:start_h+th, start_w:start_w+tw]
    else:
        N, H, W = image.shape
        th, tw = target_size
        
        # Pad if needed
        pad_h = max(th - H, 0)
        pad_w = max(tw - W, 0)
        
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            # Pad H and W dimensions (axis 1 and 2)
            image = np.pad(image, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
            
        # Update shape
        H, W = image.shape[1], image.shape[2]
        
        start_h = (H - th) // 2
        start_w = (W - tw) // 2
        return image[:, start_h:start_h+th, start_w:start_w+tw]


class MRIPairedDataset(Dataset):
    """
    Dataset for paired 0.5T and 1.5T MRI data.
    Loads k-space data from .mat files and reconstructs images.
    """
    
    def __init__(
        self,
        data_root: str,
        sequence_type: str = 'fse',  # 'fse', 'irfse', or 'se'
        target_size: Tuple[int, int] = (256, 256),
        split: str = 'train',
        preload: bool = False,
        return_kspace: bool = False
    ):
        """
        Args:
            data_root: Root path containing train/test folders
            sequence_type: MRI sequence type ('fse', 'irfse', 'se')
            target_size: Output image size (H, W)
            split: 'train' or 'test'
            preload: Whether to preload all data into memory
            return_kspace: Whether to return k-space data
        """
        self.data_root = data_root
        self.sequence_type = sequence_type
        self.target_size = target_size
        self.split = split
        self.preload = preload
        self.return_kspace = return_kspace
        
        # Build paths
        split_path = os.path.join(data_root, split)
        self.path_0_5 = os.path.join(split_path, f'Raw_0_5_{sequence_type}')
        self.path_1_5 = os.path.join(split_path, f'Raw_1_5_{sequence_type}')
        
        # Get paired files
        files_0_5 = set([f for f in os.listdir(self.path_0_5) if f.endswith('.mat')])
        files_1_5 = set([f for f in os.listdir(self.path_1_5) if f.endswith('.mat')])
        
        # Find common files (paired)
        common_files = sorted(files_0_5 & files_1_5)
        self.file_list = common_files
        
        # Key names in .mat files
        self.key_0_5 = sequence_type  # 'fse', 'irfse', 'se'
        # Fix key name for 1.5T data
        if sequence_type == 'fse':
            self.key_1_5 = 'Highfse'
        elif sequence_type == 'irfse':
            self.key_1_5 = 'Highirfse'
        elif sequence_type == 'se':
            self.key_1_5 = 'Highse'
        else:
            self.key_1_5 = f'High{sequence_type}'
        
        # Build sample index (file_idx, slice_idx)
        self.samples = []
        for file_idx, filename in enumerate(self.file_list):
            # Load to get number of slices
            mat_0_5 = sio.loadmat(os.path.join(self.path_0_5, filename))
            num_slices = mat_0_5[self.key_0_5].shape[2]
            
            for slice_idx in range(num_slices):
                self.samples.append((file_idx, slice_idx))
        
        # Preload data if requested
        self.data_cache = {}
        # Simple cache for non-preload mode
        self.last_loaded_file_idx = -1
        self.last_loaded_data = None
        
        if preload:
            self._preload_all()
        
        print(f"[{split.upper()}] Loaded {len(self.file_list)} subjects, "
              f"{len(self.samples)} slices for sequence '{sequence_type}'")
    
    def _preload_all(self):
        """Preload all data into memory."""
        print("Preloading data...")
        for file_idx, filename in enumerate(self.file_list):
            # Load 0.5T
            mat_0_5 = sio.loadmat(os.path.join(self.path_0_5, filename))
            kspace_0_5 = mat_0_5[self.key_0_5]
            
            # Load 1.5T
            mat_1_5 = sio.loadmat(os.path.join(self.path_1_5, filename))
            kspace_1_5 = mat_1_5[self.key_1_5]
            
            # Reconstruct images
            img_0_5 = kspace_to_image_rss(kspace_0_5)
            img_1_5 = kspace_to_image_rss(kspace_1_5)
            
            self.data_cache[file_idx] = {
                'img_0_5': img_0_5,
                'img_1_5': img_1_5,
                'kspace_0_5': kspace_0_5 if self.return_kspace else None,
                'kspace_1_5': kspace_1_5 if self.return_kspace else None,
                'is_preloaded_image': True
            }
        print("Preloading complete.")
    
    def _load_file(self, file_idx: int) -> Dict:
        """Load raw k-space data for a single file."""
        if file_idx in self.data_cache:
            return self.data_cache[file_idx]
            
        # Check simple LRU cache (for non-preload mode)
        if self.last_loaded_file_idx == file_idx and self.last_loaded_data is not None:
             return self.last_loaded_data
        
        filename = self.file_list[file_idx]
        
        # Load 0.5T
        mat_0_5 = sio.loadmat(os.path.join(self.path_0_5, filename))
        kspace_0_5 = mat_0_5[self.key_0_5]
        
        # Load 1.5T
        mat_1_5 = sio.loadmat(os.path.join(self.path_1_5, filename))
        kspace_1_5 = mat_1_5[self.key_1_5]
        
        # Do NOT reconstruct images here for non-preload mode
        # Just return k-space
        
        data = {
            'kspace_0_5': kspace_0_5,
            'kspace_1_5': kspace_1_5,
            'is_preloaded_image': False
        }
        
        # Update simple cache
        self.last_loaded_file_idx = file_idx
        self.last_loaded_data = data
        
        return data
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, slice_idx = self.samples[idx]
        data = self._load_file(file_idx)
        
        if data.get('is_preloaded_image', False):
            # Data is already reconstructed images (from preload)
            img_0_5 = data['img_0_5'][slice_idx]  # (H, W)
            img_1_5 = data['img_1_5'][slice_idx]  # (H, W)
        else:
            # Data is raw k-space, reconstruct only this slice
            # kspace shape: (H, W, slices, coils)
            kspace_slice_0_5 = data['kspace_0_5'][:, :, slice_idx, :]
            kspace_slice_1_5 = data['kspace_1_5'][:, :, slice_idx, :]
            
            img_0_5 = kspace_to_image_rss(kspace_slice_0_5)
            img_1_5 = kspace_to_image_rss(kspace_slice_1_5)
        
        # Center crop to common size
        img_0_5 = center_crop(img_0_5, self.target_size)
        img_1_5 = center_crop(img_1_5, self.target_size)
        
        # Normalize
        img_0_5 = normalize_image(img_0_5)
        img_1_5 = normalize_image(img_1_5)
        
        # Convert to tensor (add channel dimension)
        img_0_5 = torch.from_numpy(img_0_5).unsqueeze(0)  # (1, H, W)
        img_1_5 = torch.from_numpy(img_1_5).unsqueeze(0)  # (1, H, W)
        
        # Domain labels: 0 for 0.5T (low-field), 1 for 1.5T (high-field)
        label_0_5 = torch.tensor([0.0], dtype=torch.float32)
        label_1_5 = torch.tensor([1.0], dtype=torch.float32)
        
        result = {
            'img_low': img_0_5,      # 0.5T image
            'img_high': img_1_5,     # 1.5T image
            'label_low': label_0_5,  # 0.5T label (0)
            'label_high': label_1_5, # 1.5T label (1)
            'file_idx': file_idx,
            'slice_idx': slice_idx
        }
        
        if self.return_kspace:
            # If return_kspace is requested, we need to handle it
            # For preload mode, it might be in data
            # For non-preload mode, it is in data
            if data.get('kspace_0_5') is not None:
                 result['kspace_low'] = torch.from_numpy(data['kspace_0_5'][:, :, slice_idx, :])
                 result['kspace_high'] = torch.from_numpy(data['kspace_1_5'][:, :, slice_idx, :])
        
        return result


class CombinedMRIDataset(Dataset):
    """
    Combined dataset for all sequence types (FSE, IRFSE, SE).
    """
    
    def __init__(
        self,
        data_root: str,
        sequence_types: List[str] = ['fse', 'irfse', 'se'],
        target_size: Tuple[int, int] = (256, 256),
        split: str = 'train',
        preload: bool = False
    ):
        self.datasets = []
        self.cumulative_lengths = [0]
        
        for seq_type in sequence_types:
            try:
                ds = MRIPairedDataset(
                    data_root=data_root,
                    sequence_type=seq_type,
                    target_size=target_size,
                    split=split,
                    preload=preload
                )
                self.datasets.append(ds)
                self.cumulative_lengths.append(
                    self.cumulative_lengths[-1] + len(ds)
                )
            except Exception as e:
                print(f"Warning: Could not load {seq_type} dataset: {e}")
        
        print(f"[{split.upper()}] Combined dataset: {len(self)} total slices")
    
    def __len__(self) -> int:
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        for i, (start, end) in enumerate(zip(
            self.cumulative_lengths[:-1], 
            self.cumulative_lengths[1:]
        )):
            if start <= idx < end:
                return self.datasets[i][idx - start]
        
        raise IndexError(f"Index {idx} out of range")


def get_dataloader(
    data_root: str,
    sequence_type: str = 'fse',
    target_size: Tuple[int, int] = (256, 256),
    split: str = 'train',
    batch_size: int = 4,
    num_workers: int = 4,
    preload: bool = False,
    shuffle: bool = None
) -> DataLoader:
    """
    Create a DataLoader for the MRI dataset.
    
    Args:
        data_root: Root path containing train/test folders
        sequence_type: 'fse', 'irfse', 'se', or 'all'
        target_size: Output image size
        split: 'train' or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        preload: Whether to preload data
        shuffle: Whether to shuffle (default: True for train, False for test)
    
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    if sequence_type == 'all':
        dataset = CombinedMRIDataset(
            data_root=data_root,
            target_size=target_size,
            split=split,
            preload=preload
        )
    else:
        dataset = MRIPairedDataset(
            data_root=data_root,
            sequence_type=sequence_type,
            target_size=target_size,
            split=split,
            preload=preload
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )


if __name__ == '__main__':
    # Test the dataset
    data_root = '/data/data54/wanghaobo/Counterfactual-LatentSpaceSharing-LFRecon/Third_full'
    
    # Test single sequence
    loader = get_dataloader(
        data_root=data_root,
        sequence_type='fse',
        target_size=(256, 256),
        split='train',
        batch_size=2,
        num_workers=0
    )
    
    batch = next(iter(loader))
    print("\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
