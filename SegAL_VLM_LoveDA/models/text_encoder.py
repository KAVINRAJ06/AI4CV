import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze=True):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embed_dim = int(getattr(getattr(self.model, "config", None), "projection_dim", 512))
        self._cache_key = None
        self._cache_value = None
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, text_prompts, device):
        """
        Args:
            text_prompts: List of strings.
            device: torch device.
        Returns:
            text_features: Tensor of shape (num_prompts, embed_dim)
        """
        key = (tuple(text_prompts), str(device))
        if self._cache_key == key and self._cache_value is not None:
            return self._cache_value

        inputs = self.processor(text=list(text_prompts), return_tensors="pt", padding=True).to(device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features.detach()
        self._cache_key = key
        self._cache_value = text_features
        return text_features
