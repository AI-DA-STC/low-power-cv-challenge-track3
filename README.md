## TO DO (Implementation)
- Propose a hardware aware pruning strategy 
    - Construct latency table and simple NN to predict latency from input pruning ratios
    - Use gradient descent to update the pruning ratio 

Experiments to perform
- Exp 0 : KD
- Exp 1 : KD + Post training quantization (4/8bit) for Weights and activation
- Exp 2 : KD + quantization aware training (4/8bit) for weights and activations
- Exp 3 : 


## Backlog

- skipped : Pruning using https://github.com/quic/aimet since library only supports CNN based models

- Exp 2 : With and without augmentation
- Exp 3 : Use scale & shift invariant loss instead of mse for depth estimation loss (as used in depthanythingv2)
- Exp 4 : Use cos_sim instead of KL_div (as used in depthanythingv1)


## Done
- Accelerate training using distributed gpu-cuda and run for ~10 epochs (with and without augmentation)
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