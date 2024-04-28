#!/usr/bin/env python3
import os
import re
import json
import torch
import logging
import safetensors

from nano_llm.utils import AttributeDict, download_model


class MMProjector():
    """
    Multimodal projector MLP used by Llava and other Vision-Language Models
    to map from CLIP vision embedding space to the LLM's word embedding space.
    """
    @staticmethod
    def from_pretrained(model, dtype=torch.float16):
        """
        Load the projector from the HuggingFace Transformers model (Llava)
        
        If the model directory doesn't already have mm_projector.bin, its
        weights will be extracted from the main model (and saved there)
        
        Parameters:
        
          model (str) -- either the path to the model, or HuggingFace model repo/name
                         (e.g. liuhaotian/llava-v1.5-13b)
                         
          dtype (dtype) -- use either torch.float32 or torch.float16 weights
        """
        from nano_llm import NanoLLM
        
        if isinstance(model, NanoLLM):
            return MMProjector(model.config.mm_projector_path, model.config, dtype)
        elif isinstance(model, str):
            if not os.path.isdir(model):
                model = download_model(model)
            return MMProjector(model)
        else:
            raise ValueError(f"model should either be a string containing the path or name of the HuggingFace model, or a NanoLLM model instance")
            
    def __init__(self, model_path, config=None, dtype=torch.float16):
        """
        Create the mm_projector network and load its weights
        """
        if config:
            self.config = config
        else:
            config_path = os.path.join(model_path, 'config.json')
            
            if not os.path.isfile(config_path):
                raise IOError("couldn't find mm_projector config file {config_path}")
                
            with open(config_path) as config_file:
                self.config = AttributeDict(json.load(config_file))

        self.model_path = model_path
        self.type = 'linear'
        self.dtype = dtype
        
        if hasattr(self.config, 'mm_projector_type'):
            self.type = self.config.mm_projector_type

        # create different types of projector models
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.type)
        
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(torch.nn.GELU())
                modules.append(torch.nn.Linear(self.config.hidden_size, self.config.hidden_size))
            self.model = torch.nn.Sequential(*modules)
        elif self.type == 'mlp_downsample':
            self.model = torch.nn.Sequential(
                DownSampleBlock(),
                torch.nn.LayerNorm(self.config.mm_hidden_size * 4),
                torch.nn.Linear(self.config.mm_hidden_size * 4, self.config.hidden_size),
                torch.nn.GELU(),
                torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)
            )
        elif self.type == 'linear':
            self.model = torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)
        else:
            raise RuntimeError(f"Unknown vision mm_projector type: {self.type}")
           
        # load projector weights, extracting from the original model if needed
        weights = self.load_torch(os.path.join(self.model_path, 'mm_projector.bin'))
        
        if not weights:
            weights = self.load_safetensors(os.path.join(self.model_path, 'model.safetensors'))
        
        if not weights:
            weights = self.load_sharded(self.model_path)
            
        if not weights:
            raise IOError(f"could not mm_projector weights under {self.model_path}")

        weights = {k.replace('model.mm_projector.', ''):v for k,v in weights.items()}  
        weights = {k.replace('layers.', ''):v for k,v in weights.items()}  
        
        print(f"mm_projector ({self.type})", self.model)
        print("mm_projector weights", weights.keys())
        
        self.model.load_state_dict(weights)
        self.model.to(dtype=self.dtype, device='cuda:0').eval()

    def __call__(self, *args, **kwargs):
        """
        Forward-pass call to the model
        """
        with torch.inference_mode():
            return self.model(*args, **kwargs)
            
    @staticmethod
    def load_torch(filename):
        if not os.path.isfile(filename):
            return None

        return torch.load(filename, map_location='cpu')
    
    @staticmethod
    def load_safetensors(filename):
        if not os.path.isfile(filename):
            return None
            
        return safetensors.torch.load_file(filename, device='cpu')

    @staticmethod
    def load_sharded(model_path, save='mm_projector.bin'):
        weight_indexes = [
            os.path.join(model_path, 'pytorch_model.bin.index.json'),
            os.path.join(model_path, 'model.safetensors.index.json'),
        ]
        
        for weight_index in weight_indexes:
            if os.path.isfile(weight_index):
                break

        if not os.path.isfile(weight_index):
            logging.error(f"could not find sharded weight index at these locations:  {weight_indexes}")
            return False
            
        with open(weight_index, 'r') as file:
            weight_map = json.load(file)['weight_map']
            
        weights_path = None
        
        for key, value in weight_map.items():
            if 'mm_projector' in key:
                weights_path = os.path.join(model_path, value)
                break
         
        if not weights_path:
            logging.error(f"could not find mm_projector weights in sharded weight index {weight_index}")
            return False
             
        logging.debug(f"extracting mm_projector weights from {weights_path}")
        
        if 'safetensors' in weight_index:
            weights = safetensors.torch.load_file(weights_path, device='cpu')
        else:
            weights = torch.load(weights_path, map_location='cpu')
            
        weights = {k : v for k, v in weights.items() if 'mm_projector' in k}
        
        if save:
            save_path = os.path.join(model_path, save)
            logging.debug(f"saving mm_projector weights to {save_path}")
            torch.save(weights, save_path)
            
        return weights

            
            
class DownSampleBlock(torch.nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        return x

