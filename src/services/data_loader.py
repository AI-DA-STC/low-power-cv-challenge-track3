import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional, List, Dict, Any

class Places365Dataset(Dataset):
    """Dataset class for loading Places365 images."""
    
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'train', 
                 transform: Optional[transforms.Compose] = None,
                 image_size: Tuple[int, int] = (384, 384)):
        """
        Args:
            root_dir (str): Path to the Places365 dataset
            split (str): 'train' or 'val'
            transform (torchvision.transforms.Compose, optional): Transformations to apply
            image_size (tuple): Height and width to resize images to
        """
        self.root_dir = os.path.join(root_dir, f'places365_standard/{split}')
        self.transform = transform
        self.image_size = image_size
        
        # If no transform is provided, create a default one
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])
        
        # Get all image paths
        self.image_paths = []
        for category in os.listdir(self.root_dir):
            category_path = os.path.join(self.root_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(category_path, img_name))
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return {"pixel_values": image}


def create_data_loaders(
    data_dir: str,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (384, 384),
    num_workers: int = 4,
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders for Places365 dataset.
    
    Args:
        data_dir (str): Path to Places365 data directory
        batch_size (int): Batch size for training and validation
        image_size (tuple): Height and width to resize images to
        num_workers (int): Number of workers for data loading
        train_transform (torchvision.transforms.Compose, optional): Custom transforms for training
        val_transform (torchvision.transforms.Compose, optional): Custom transforms for validation
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = Places365Dataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform,
        image_size=image_size
    )
    
    val_dataset = Places365Dataset(
        root_dir=data_dir,
        split='val',
        transform=val_transform,
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader