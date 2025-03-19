import os
import wandb
import torch
import argparse
import transformers
from transformers import TrainingArguments
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir("config"))
sys.path.append(str(root))

import logging
logger = logging.getLogger(__name__)

from config import settings
from src.modules.student import StudentModel
from src.modules.teacher import TeacherModel
from utils.places365_data_loader import create_data_loaders
from src.services.trainer_KD import KnowledgeDistillationTrainer
from src.utils.loss import DistillationLoss
from utils import logger as logging

logging.setup_logging()

def parse_args():
    """
    Parse command line arguments with defaults from settings.
    Command line arguments override the settings.
    """
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Depth Estimation")
    
    # Data args
    parser.add_argument("--data_dir", type=str, 
                        default=str(settings.DATA_DIR), 
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, 
                        default=str(settings.OUTPUT_DIR), 
                        help="Output directory")

    # Multi-GPU training args
    parser.add_argument("--use_data_parallel", type=bool, 
                        default=settings.USE_DATA_PARALLEL, 
                        help="Use DataParallel for multi GPU training")
    parser.add_argument("--device", type=str, 
                        default=settings.DEVICE, 
                        help="Device to use (e.g., 'cuda' or 'cpu')")
    
    # Training args
    parser.add_argument("--batch_size", type=int, 
                        default=settings.BATCH_SIZE, 
                        help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, 
                        default=settings.VAL_BATCH_SIZE, 
                        help="Batch size for validation")
    parser.add_argument("--num_workers", type=int, 
                        default=settings.NUM_WORKERS, 
                        help="Number of workers for data loading")
    parser.add_argument("--learning_rate", type=float, 
                        default=settings.LEARNING_RATE, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, 
                        default=settings.WEIGHT_DECAY, 
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, 
                        default=settings.NUM_EPOCHS, 
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int,
                        default=settings.MAX_STEPS,
                        help="Maximum number of training steps")
    parser.add_argument("--image_size", type=int, 
                        default=settings.IMAGE_SIZE, 
                        help="Size of input images")
    
    # Distillation args
    parser.add_argument("--gamma", type=float, 
                        default=settings.GAMMA, 
                        help="Weight for KL divergence loss")
    parser.add_argument("--temperature", type=float, 
                        default=settings.TEMPERATURE, 
                        help="Temperature for softening logits")
    
    # Model args
    parser.add_argument("--teacher_model", type=str, 
                        default=settings.TEACHER_MODEL, 
                        help="Teacher model name")
    parser.add_argument("--student_model", type=str, 
                        default=settings.STUDENT_MODEL, 
                        help="Student model name")
    
    # Other args
    parser.add_argument("--seed", type=int, 
                        default=settings.SEED, 
                        help="Random seed")
    parser.add_argument("--save_steps", type=int, 
                        default=settings.SAVE_STEPS, 
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, 
                        default=settings.EVAL_STEPS, 
                        help="Evaluate every X steps")
    parser.add_argument("--logging_steps", type=int, 
                        default=settings.LOGGING_STEPS, 
                        help="Log every X steps")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    transformers.set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers
    )
    
    # Initialize models
    teacher_model = TeacherModel(model_name=args.teacher_model)
    student_model = StudentModel(model_name=args.student_model)

    # Wrap models with DataParallel if requested and if multiple GPUs are available
    if args.use_data_parallel and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        teacher_model = torch.nn.DataParallel(teacher_model)
        student_model = torch.nn.DataParallel(student_model)
    
    # Initialize distillation loss
    distillation_loss = DistillationLoss(gamma=args.gamma, temperature=args.temperature)
    run = wandb.init(project=settings.WANDB_PROJECT, name=f"{settings.RUN_NAME}_{args.student_model}")
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        warmup_steps=settings.WARMUP_STEPS,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type=settings.LR_SCHEDULER_TYPE,
        max_steps=args.max_steps
    )
    
    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        distillation_loss=distillation_loss,
        device=args.device
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    
    logger.info("Training completed!")
    run.finish()

if __name__ == "__main__":
    main()