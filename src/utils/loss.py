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
        self.ce_loss = nn.CrossEntropyLoss()
        
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
        
        # Normalize logits for KL divergence
        teacher_logits_norm = (teacher_logits - teacher_logits.min()) / (teacher_logits.max() - teacher_logits.min() + 1e-6)
        student_logits_norm = (student_logits - student_logits.min()) / (student_logits.max() - student_logits.min() + 1e-6)
        
        # Apply temperature scaling for KL divergence
        teacher_logits_scaled = teacher_logits_norm / self.temperature
        student_logits_scaled = student_logits_norm / self.temperature
        
        # Calculate cross entropy loss
        # First, reshape for cross entropy loss
        b, c, h, w = teacher_pseudo_labels.shape
        teacher_pseudo_labels_flat = teacher_pseudo_labels.reshape(b, c, h * w).permute(0, 2, 1)
        student_labels_flat = student_labels.reshape(b, c, h * w).permute(0, 2, 1)
        
        ce_loss = self.ce_loss(student_labels_flat, teacher_pseudo_labels_flat)
        
        # Calculate KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=1),
            F.softmax(teacher_logits_scaled, dim=1),
            reduction='batchmean'
        )
        
        # Combine losses
        total_loss = ce_loss + self.gamma * kl_loss
        
        return total_loss


class DepthL1Loss(nn.Module):
    """
    L1 loss for depth estimation.
    """
    
    def __init__(self, scale_invariant: bool = True, lambda_scale: float = 0.5):
        """
        Initialize the depth L1 loss.
        
        Args:
            scale_invariant (bool): Whether to use scale-invariant L1 loss
            lambda_scale (float): Weight for the scale-invariant term
        """
        super(DepthL1Loss, self).__init__()
        self.scale_invariant = scale_invariant
        self.lambda_scale = lambda_scale
        
    def forward(
        self, 
        pred_depth: torch.Tensor, 
        target_depth: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the depth L1 loss.
        
        Args:
            pred_depth (torch.Tensor): Predicted depth
            target_depth (torch.Tensor): Target depth
            
        Returns:
            torch.Tensor: The calculated loss
        """
        # Calculate standard L1 loss
        l1_loss = torch.abs(pred_depth - target_depth).mean()
        
        if self.scale_invariant:
            # Calculate scale-invariant term
            log_pred = torch.log(pred_depth + 1e-8)
            log_target = torch.log(target_depth + 1e-8)
            
            # Calculate difference
            diff = log_pred - log_target
            
            # Calculate scale-invariant term
            si_term = torch.pow(diff.mean(), 2)
            
            # Combine losses
            loss = l1_loss + self.lambda_scale * si_term
            return loss
        
        return l1_loss