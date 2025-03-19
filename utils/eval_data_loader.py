import os
from PIL import Image
from typing import Optional, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class IndoorOutdoorDataset(Dataset):
    """
    Custom dataset for loading indoor and outdoor images along with their labels.
    Expected folder structure:
    
        root_dir/
            indoor/              # .jpg images for indoor scenes
            indoor_labels/       # .pt label tensors (shape: 384x384) matching indoor images
            outdoor/             # .jpg images for outdoor scenes
            outdoor_labels/      # .pt label tensors (shape: 384x384) matching outdoor images
    """
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[transforms.Compose] = None, 
                 image_size: Tuple[int, int] = (384, 384)):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            transform (torchvision.transforms.Compose, optional): Transformations to apply to images.
            image_size (tuple): Desired image size (height, width).
        """
        self.root_dir = root_dir
        # Use a default transform if none is provided.
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.image_size = image_size
        
        # List to store tuples of (image_path, label_path)
        self.data = []
        
        # Process indoor images and their labels.
        indoor_dir = os.path.join(root_dir, "indoor")
        indoor_labels_dir = os.path.join(root_dir, "indoor_labels")
        if os.path.exists(indoor_dir) and os.path.exists(indoor_labels_dir):
            for img_name in os.listdir(indoor_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(indoor_dir, img_name)
                    # Construct the label file name by replacing the image extension with .pt
                    label_name = os.path.splitext(img_name)[0] + ".pt"
                    label_path = os.path.join(indoor_labels_dir, label_name)
                    if os.path.exists(label_path):
                        self.data.append((img_path, label_path))
        
        # Process outdoor images and their labels.
        outdoor_dir = os.path.join(root_dir, "outdoor")
        outdoor_labels_dir = os.path.join(root_dir, "outdoor_labels")
        if os.path.exists(outdoor_dir) and os.path.exists(outdoor_labels_dir):
            for img_name in os.listdir(outdoor_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(outdoor_dir, img_name)
                    label_name = os.path.splitext(img_name)[0] + ".pt"
                    label_path = os.path.join(outdoor_labels_dir, label_name)
                    if os.path.exists(label_path):
                        self.data.append((img_path, label_path))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label_path = self.data[idx]
        
        # Load the image and convert it to RGB.
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load the label tensor.
        label = torch.load(label_path,map_location=torch.device('cpu'))
        
        # Return dictionary uses key "label" to match your DataLoader usage.
        return {"pixel_values": image, "label": label}
    
class IndoorOutdoorDatasetUnlabelled(Dataset):
    """
    Custom dataset for loading indoor and outdoor images without the labels.
    Expected folder structure:
    
        root_dir/
            indoor/              # .jpg images for indoor scenes
            outdoor/             # .jpg images for outdoor scenes
    """
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[transforms.Compose] = None, 
                 image_size: Tuple[int, int] = (384, 384)):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            transform (torchvision.transforms.Compose, optional): Transformations to apply to images.
            image_size (tuple): Desired image size (height, width).
        """
        self.root_dir = root_dir
        # Use a default transform if none is provided.
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.image_size = image_size
        
        # List to store tuples of (image_path, label_path)
        self.data = []
        
        # Process indoor images and their labels.
        indoor_dir = os.path.join(root_dir, "indoor")
        if os.path.exists(indoor_dir):
            for img_name in os.listdir(indoor_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(indoor_dir, img_name)
                    self.data.append(img_path)
        
        # Process outdoor images and their labels.
        outdoor_dir = os.path.join(root_dir, "outdoor")
        if os.path.exists(outdoor_dir):
            for img_name in os.listdir(outdoor_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(outdoor_dir, img_name)
                    self.data.append(img_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.data[idx]
        
        # Load the image and convert it to RGB.
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Return dictionary uses key "label" to match your DataLoader usage.
        return {"pixel_values": image}

def create_data_loader(
    data_dir: str,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (384, 384),
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
    labelled: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the combined indoor-outdoor dataset.
    
    Args:
        data_dir (str): Path to the data directory containing 'indoor', 'indoor_labels', 'outdoor', and 'outdoor_labels' subfolders.
        batch_size (int): Batch size for the DataLoader.
        image_size (tuple): Desired image size for resizing.
        num_workers (int): Number of worker processes for data loading.
        transform (torchvision.transforms.Compose, optional): Custom transformations for images.
        
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    if labelled:
        dataset = IndoorOutdoorDataset(
            root_dir=data_dir,
            transform=transform,
            image_size=image_size
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        dataset = IndoorOutdoorDatasetUnlabelled(
            root_dir=data_dir,
            transform=transform,
            image_size=image_size
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
    return data_loader
