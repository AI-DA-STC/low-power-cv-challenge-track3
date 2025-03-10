import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
from typing import Dict, Tuple, Any, Optional

class StudentModel(nn.Module):
    """
    Student model for knowledge distillation.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the student model.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        super(StudentModel, self).__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name,output_hidden_states=True)
            
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the student model.
        
        Args:
            pixel_values (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            dict: Dictionary containing logits and predicted labels
        """
        # Forward pass
        outputs = self.model(pixel_values)
        
        # Extract pseudo labels (predicted depth)
        predicted_labels = outputs.predicted_depth

        #Extract logits of last hidden layer
        logits = outputs.hidden_states[-1]
        
        return {
            "logits": logits,
            "labels": predicted_labels
        }