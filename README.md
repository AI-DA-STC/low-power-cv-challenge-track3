## TO DO
- Accelerate training using distributed gpu-cuda and run for ~10 epochs (with and without augmentation)
- Use scale & shift invariant loss instead of mse for depth estimation loss (as used in depthanythingv2)
- Use cos_sim instead of KL_div (as used in depthanythingv1)
- Include contrastive_loss to see if any difference in loss function

## Backlog


## Done

- Run the depthanythingv2 on Qualcomm ai hub 
- Data creation 
    - Dataset extraction : Places365 dataset. Outdoor : 56%, Indoor : 44%
    - Augmentation : Apply color jitter to balance distribution
- Init repo
    - Teacher model : depthAnythingv2-large and student model : dpt-small-dinov2-kitti 
    - Script for training a baseline KD pipeline with distill loss = MSE + alpha * KL_div


# Setup

For linux, in addition to installing the requirements.txt

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```