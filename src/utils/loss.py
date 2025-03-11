import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

class DistillationLoss(nn.Module):
    """
    Distillation loss for knowledge distillation.
    Loss = CE(teacher_pseudo_labels, student_labels) + gamma * KL(teacher_logits, student_logits)
    """
    
    def __init__(self, gamma: float = 0.5, temperature: float = 1.0):
        """
        Initialize the distillation loss.
        
        Args:
            gamma (float): Weight for the KL divergence term
            temperature (float): Temperature for softening the logits
        """
        super(DistillationLoss, self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        
    def forward(
        self, 
        teacher_outputs: Dict[str, torch.Tensor], 
        student_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate the distillation loss.
        
        Args:
            teacher_outputs (dict): Outputs from the teacher model
            student_outputs (dict): Outputs from the student model
            
        Returns:
            torch.Tensor: The calculated loss
        """
        # Extract outputs
        teacher_logits = teacher_outputs["logits"]
        teacher_pseudo_labels = teacher_outputs["pseudo_labels"]
        
        student_logits = student_outputs["logits"]
        student_labels = student_outputs["labels"]

        projection = nn.Linear(teacher_logits.shape[-1], student_logits.shape[-1]).to(teacher_logits.device)
        teacher_logits_projected = projection(teacher_logits)

        mse_loss = self.mse_loss(teacher_pseudo_labels, student_labels)
        
        # Normalize logits for KL divergence
        teacher_logits_norm = (teacher_logits_projected - teacher_logits_projected.min()) / (teacher_logits_projected.max() - teacher_logits_projected.min() + 1e-6)
        student_logits_norm = (student_logits - student_logits.min()) / (student_logits.max() - student_logits.min() + 1e-6)
        
        # Apply temperature scaling for KL divergence
        teacher_logits_scaled = teacher_logits_norm / self.temperature
        student_logits_scaled = student_logits_norm / self.temperature

        # Calculate KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=1),
            F.softmax(teacher_logits_scaled, dim=1),
            reduction='batchmean'
        )
        
        # Combine losses
        total_loss = mse_loss + self.gamma * kl_loss
        
        return total_loss
