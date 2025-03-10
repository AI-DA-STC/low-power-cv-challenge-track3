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
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        distillation_loss: Optional[nn.Module] = None,
        optimizers: Tuple[optim.Optimizer, Any] = (None, None),
        gamma: float = 0.5,
        temperature: float = 1.0,
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
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            optimizers=optimizers,
            **kwargs
        )
        
        # Set attributes
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Initialize distillation loss if not provided
        self.distillation_loss = distillation_loss or DistillationLoss(
            gamma=gamma, 
            temperature=temperature
        )
        
        # Move models to device
        self.student_model.to(device)
        self.teacher_model.to(device)
        
        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Set teacher to evaluation mode
        self.teacher_model.eval()
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Override get_train_dataloader to use our custom dataloader.
        """
        return self.train_dataloader
    
    def get_eval_dataloader(self) -> DataLoader:
        """
        Override get_eval_dataloader to use our custom dataloader.
        """
        return self.eval_dataloader
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to implement knowledge distillation.
        
        Args:
            model (nn.Module): The student model
            inputs (Dict): Input data
            return_outputs (bool): Whether to return outputs
            
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

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, **kwargs):
        """
        Override the train method to implement custom validation frequency.
        
        Args:
            resume_from_checkpoint (str or bool, optional): Path to checkpoint
            **kwargs: Additional arguments for training
            
        Returns:
            TrainOutput: Training output
        """
        # Initialize training arguments
        args = self.args
        
        # Add validation callback
        class ValidationCallback(TrainerCallback):
            def __init__(self, trainer, eval_steps=10):
                self.trainer = trainer
                self.eval_steps = eval_steps
                self.global_step = 0
                
            def on_step_end(self, args, state, control, **kwargs):
                self.global_step += 1
                if self.global_step % self.eval_steps == 0:
                    self.trainer.evaluate()
        
        # Add validation callback
        validation_callback = ValidationCallback(self, eval_steps=10)
        self.add_callback(validation_callback)
        
        # Call parent train method
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)