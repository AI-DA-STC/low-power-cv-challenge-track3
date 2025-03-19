import torch
import torch.nn as nn
import numpy as np
import logging
from transformers import AutoModel
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

class LatencyTable:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "mps"):
        self.device = device
        self.model = self.import_model(model_path)
    
    def import_model(self, model_path):
        """Load the HF model from safetensors format."""
        model = DPTForDepthEstimation.from_pretrained(model_path)
        try:
            state_dict = load_file(f"{model_path}/model.safetensors")
        except Exception as e:
            logger.error("model.safetensors file not found!")
        new_state_dict = {k.replace("module.model.", ""): v for k, v in state_dict.items()}  # Remove the prefix
        model.load_state_dict(new_state_dict) 
        model.to("cuda")
        model.eval()
        return model

    def compute_layer_importance(self, dataloader, num_batches=5):
        """Compute layer importance using gradients."""
        importance_scores = {}
        criterion = nn.CrossEntropyLoss()
        layers = list(self.model.named_modules())  # Get all model layers

        for i, (inputs, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()

            for name, module in layers:
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):  # Focus on key layers
                    grad = module.weight.grad
                    if grad is not None:
                        importance_scores[name] = importance_scores.get(name, 0) + grad.abs().mean().item()
        
        # Normalize importance scores
        max_val = max(importance_scores.values())
        importance_scores = {k: v / max_val for k, v in importance_scores.items()}

        return importance_scores

    def sample_pruning_ratio(self, importance_scores, N=10, alpha=0.8):
        """Sample pruning ratios based on importance scores."""
        pruning_sets = []
        for _ in range(N):
            pruning_ratios = []
            for layer, importance in importance_scores.items():
                norm_importance = importance  # Assume importance is already normalized
                p_max = max(0.5, 1 - alpha * norm_importance)
                p_min = 0 if p_max == 0.5 else 1 - alpha * norm_importance
                pruning_ratios.append(np.random.uniform(p_min, p_max))
            pruning_sets.append(pruning_ratios)
        return np.array(pruning_sets)

    def sample_latency(self, pruning_ratios):
        """Perform pruning and get latency values from Qualcomm AI Hub"""
        dummy_input = torch.randn(1, *self.model.config.to_dict()["input_size"]).to(self.device)

        # Apply pruning
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)) and name in pruning_ratios:
                    pruning_ratio = pruning_ratios[name]
                    weight = module.weight.data
                    threshold = torch.quantile(weight.abs(), pruning_ratio)
                    mask = torch.abs(weight) > threshold
                    module.weight.data.mul_(mask)  # Apply pruning
        
        # Convert to ONNX format
        traced_model = torch.jit.trace(self.model, (torch.randn(1, *self.model.config.hidden_size).to(self.device)))
        
        # Compile the pruned model
        compile_job = hub.submit_compile_job(
            name="pruned_model",
            model=traced_torch_model,
            device=hub.Device("Samsung Galaxy S24 (Family)"),
            input_specs=dict(image=(1, 3, 384, 384)),  # Modify based on model input shape
        )
        
        # Download profile and extract latency
        profile_job = compile_job.download_profile()
        latency = profile["execution_summary"]["estimated_inference_time"]
        
        return latency

