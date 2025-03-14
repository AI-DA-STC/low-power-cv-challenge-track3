import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, TrainerCallback
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

from ..modules.student import StudentModel
from ..modules.teacher import TeacherModel
from ..utils.loss import DistillationLoss

class KnowledgeDistillationTrainer(Trainer):
    """
    Custom Trainer class for knowledge distillation.
    """
    
    def __init__(
        self,
        student_model: StudentModel,
        teacher_model: TeacherModel,
        args: TrainingArguments,
        distillation_loss: Optional[nn.Module] = None,
        optimizers: Tuple[optim.Optimizer, Any] = (None, None),
        device: Optional[torch.device] = "mps",
        **kwargs
    ):
        """
        Initialize the knowledge distillation trainer.
        
        Args:
            student_model (StudentModel): The student model to train
            teacher_model (TeacherModel): The teacher model to distill from
            args (TrainingArguments): Training arguments
            train_dataloader (DataLoader): Training data loader
            eval_dataloader (DataLoader, optional): Evaluation data loader
            distillation_loss (nn.Module, optional): Loss function for distillation
            optimizers (tuple, optional): Tuple of (optimizer, scheduler)
            gamma (float): Weight for the KL divergence term
            temperature (float): Temperature for softening the logits
            **kwargs: Additional arguments for the Trainer
        """
        # Initialize the base Trainer with the student model
        super().__init__(
            model=student_model,
            args=args,
            **kwargs
        )
        
        # Set attributes
        self.student_model = student_model
        self.teacher_model = teacher_model
        
        self.distillation_loss = distillation_loss 
        
        # Move models to device
        self.student_model.to(device)
        self.teacher_model.to(device)
        
        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Set teacher to evaluation mode
        self.teacher_model.eval()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to implement knowledge distillation.
        
        Args:
            model (nn.Module): The student model
            inputs (Dict): Input data            
        Returns:
            tuple or torch.Tensor: The loss or (loss, outputs)
        """
        # Get pixel values
        pixel_values = inputs["pixel_values"]
        
        # Forward pass through teacher model
        with torch.no_grad():
            teacher_outputs = self.teacher_model(pixel_values)
            
        # Forward pass through student model
        student_outputs = self.student_model(pixel_values)
        
        # Calculate distillation loss
        loss = self.distillation_loss(teacher_outputs, student_outputs)
        
        if return_outputs:
            return loss, student_outputs
        
        return loss
    
    '''def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and return metrics.
        """
        # Run standard evaluation
        output = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Make sure metrics dictionary exists
        if not hasattr(output, "metrics"):
            output.metrics = {}
        
        # Add 'eval_loss' to the metrics if not already present
        if f"{metric_key_prefix}_loss" not in output.metrics:
            # Calculate average loss over the evaluation dataset
            self.model.eval()
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
            eval_losses = []
            for inputs in eval_dataloader:
                # Move inputs to the correct device
                inputs = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                with torch.no_grad():
                    loss = self.compute_loss(self.model, inputs)
                eval_losses.append(loss.detach().cpu().item())
            
            # Add the loss to the metrics
            output.metrics[f"{metric_key_prefix}_loss"] = np.mean(eval_losses)
        
        return output'''