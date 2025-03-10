import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
from typing import Dict, Tuple, Any, Optional

class TeacherModel(nn.Module):
    """
    Teacher model for depth estimation.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the teacher model.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        super(TeacherModel, self).__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name,output_hidden_states=True)
        
        # Freeze all parameters of the teacher model
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the teacher model.
        
        Args:
            pixel_values (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            dict: Dictionary containing logits and pseudo labels
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values)
            
            # Extract pseudo labels (predicted depth)
            pseudo_labels = outputs.predicted_depth

            #Extract logits of last hidden layer
            logits = outputs.hidden_states[-1]
            
        return {
            "logits": logits,
            "pseudo_labels": pseudo_labels
        }
    
    @torch.no_grad()
    def generate_pseudo_labels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Generate pseudo labels for the given images.
        
        Args:
            pixel_values (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Pseudo labels
        """
        outputs = self.forward(pixel_values)
        return outputs["pseudo_labels"]