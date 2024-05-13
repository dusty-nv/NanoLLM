#!/usr/bin/env python3
import os
import json
import time
import psutil
import logging
import traceback

import cv2
import PIL
import torch
import torch2trt
import tensorrt

import numpy as np
import torchvision.transforms as T

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor, SiglipVisionModel
from ..utils import AttributeDict, load_image, torch_image, image_size, convert_tensor, download_model, print_table

_clip_model_cache = dict(image={}, text={})

class CLIPImageEmbedding():
    """
    CLIP feature extractor and projector for generating image embeddings.
    """
    @staticmethod
    def from_pretrained(model="openai/clip-vit-large-patch14-336", dtype=torch.float16, crop=False, 
                        use_cache=True, use_tensorrt=True, **kwargs):
        """
        Load a CLIP or SigLIP vision encoder model from HuggingFace Hub or a local checkpoint.
        """                
        global _clip_model_cache
        
        if use_cache and model in _clip_model_cache['image']:
            return _clip_model_cache['image'][model]
            
        inst = CLIPImageEmbedding(model, dtype=dtype, crop=crop, use_tensorrt=use_tensorrt, **kwargs)
        
        if use_cache:
            _clip_model_cache['image'][model] = inst
            
        return inst
    
    def __init__(self, model, dtype=torch.float16, crop=False, use_tensorrt=True, **kwargs):
        self.stats = AttributeDict()
        self.config = AttributeDict(name=model, crop=crop)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.stream = None
        
        self.dtype = torch.float32 if use_tensorrt else dtype # TRT handles FP16 internally
        self.output_dtype = dtype  # still output the embeddings with the requested dtype
        
        self.model_types = {
            'clip':  dict(preprocessor=CLIPImageProcessor, model=CLIPVisionModelWithProjection, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            'siglip': dict(preprocessor=SiglipImageProcessor, model=SiglipVisionModel, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        }
        
        def check_model_type(model_name):
            for key in self.model_types.keys():
                if key in model_name.lower():
                    return key 
            return None
            
        self.model_type = check_model_type(model)  # for model names, or paths containing the name
       
        if not self.model_type and os.path.isdir(model):  # for paths without, check the config.json
            try:
                config_path = os.path.join(model, 'config.json')
                with open(config_path) as config_file:
                    model_type = json.load(config_file)['model_type']
                    self.model_type = check_model_type(model_type)
            except Exception as error:
                logging.error(f"failed to get vision encoder type from local model config under {model} ({error})")
 
        if not self.model_type:
            raise ValueError(f"tried loading unrecognized vision encoder from {model} - supported model types are CLIP and SigLIP")
            
        logging.info(f'loading {self.model_type} vision model {model}')

        factory = self.model_types[self.model_type]
        
        self.model = factory['model'].from_pretrained(model, torch_dtype=self.dtype)#.to(self.device).eval()
        self.config.input_shape = (self.model.config.image_size, self.model.config.image_size)
        
        #self.preprocessor = model_type['preprocessor'].from_pretrained(model, torch_dtype=self.dtype)#.to(self.device)
        
        # Pre-processing is able to use GPU with torchvision (cropping is optional)
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L79
        self.preprocessor = torch.nn.Sequential()

        self.preprocessor.append(
            T.Resize(
                self.config.input_shape[0] if crop else self.config.input_shape, 
                interpolation=T.InterpolationMode.BICUBIC# BILINEAR
            )
        )
        
        if crop:
            self.preprocessor.append(T.CenterCrop(self.config.input_shape[0]))
   
        self.preprocessor.append(T.Normalize(factory['mean'], factory['std']))
        self.preprocessor.append(T.ConvertImageDtype(self.dtype))

        class VisionEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.config = model.config

            def forward(self, image):
                return self.model(
                    image, 
                    output_attentions=False, 
                    output_hidden_states=True, 
                    return_dict=True
                )
         
        self.model = VisionEncoder(self.model)
        self.model.to(dtype=self.dtype, device=self.device).eval()
        
        print(type(self.model), model, self.model)

        logging.debug(f'{self.config.name} warmup')
        self(PIL.Image.new('RGB', self.config.input_shape, (255,255,255)))
        print_table(self.config)
        
        if use_tensorrt:
            try:
                self.init_trt()
            except Exception as error:
                logging.error(f"Exception occurred trying to use TensorRT for {self.model_type} model ({self.config.name})\n\n{traceback.format_exc()}")

    def init_trt(self, trt_cache="/data/models/clip"): 
        if psutil.virtual_memory().total < 20 * (1024 ** 3):
            logging.warning(f"disabling CLIP TensorRT due to limited memory (falling back to --vision-api=hf)")
            return
         
        trt_path = os.path.join(trt_cache, self.config.name.replace('/','-').replace('@','-') + '-trt.pt')
        test_model_inputs = torch.ones(1, 3, *self.config.input_shape, dtype=self.dtype, device='cuda')

        if os.path.isfile(trt_path):
            logging.info(f"loading TensorRT model from {trt_path}")
            trt_model = torch2trt.TRTModule()
            trt_model.load_state_dict(torch.load(trt_path))
        else:
            logging.info(f"optimizing {self.config.name} with TensorRT...")
        
            trt_model = torch2trt.torch2trt(
                self.model,
                [test_model_inputs],
                fp16_mode=True,#(self.config.dtype == torch.float16),
                log_level=tensorrt.Logger.VERBOSE,
                max_workspace_size=(1024**3) * 3,
                use_onnx=True,
            )
        
            logging.info(f"saving TensorRT model for {self.config.name} to {trt_path}")
            
            os.makedirs(trt_cache, exist_ok=True)
            torch.save(trt_model.state_dict(), trt_path)
        
        def profile_model(model, inputs, runs=3):
            for i in range(runs+1):
                if i == 1:
                    time_begin = time.perf_counter()
                output = model(inputs)
            torch.cuda.synchronize()
            return (time.perf_counter() - time_begin) * 1000 / runs
            
        logging.info(f"torch time: {profile_model(self.model, test_model_inputs)} ms")
        logging.info(f"trt time:   {profile_model(trt_model, test_model_inputs)} ms")
          
        trt_model.config = self.model.config
        self.model = trt_model
        
    def embed_image(self, image, hidden_state=None, return_tensors='pt', return_dict=False, stream=None, **kwargs):
        """
        Return the encoded features from the given image in the embedding (or whatever the model output is).
        """
        if isinstance(image, str):
            image = load_image(image)
        
        def _convert_tensor(x):
            return convert_tensor(x, return_tensors=return_tensors, device=self.device, dtype=self.output_dtype)
            
        output = AttributeDict() if return_dict else None
        
        with torch.cuda.StreamContext(stream), torch.inference_mode():
            time_begin_enc = time.perf_counter()
            
            image = torch_image(image, dtype=self.dtype, device=self.device)
            ndims = len(image.shape)

            if ndims != 3 and ndims != 4:
                raise ValueError(f"image with dims {image.shape} was not in NCHW or NHWC format")
            
            if ndims == 3:
                image = image.unsqueeze(0)
                
            if image.shape[3] <= 4:
                image = image.permute(0, 3, 1, 2)
                
            image = self.preprocessor(image)
            model_output = self.model(image) #, output_hidden_states=hidden_state is not None)   #.pooler_output  .last_hidden_state

            if self.model_type == 'clip':
                output_embeds = model_output['image_embeds']
            elif self.model_type == 'siglip':
                output_embeds = model_output['pooler_output']
                
            if hidden_state is not None:
                hidden_tensor = _convert_tensor(model_output['hidden_states'][hidden_state])
                if return_dict:
                    output.hidden_state = hidden_tensor
                else:
                    output = hidden_tensor
                self.config.output_shape = hidden_tensor.shape
            else:
                self.config.output_shape = output_embeds.shape
                
            if return_dict:
                output.image_embeds = _convert_tensor(output_embeds) 
            elif hidden_state is None:
                output = _convert_tensor(output_embeds) 

        time_end_enc = time.perf_counter()
        
        self.stats.clip_time = time_end_enc - time_begin_enc
        self.stats.clip_rate = 1.0 / self.stats.clip_time
        self.stats.input_shape = f"{image_size(image)} -> {self.model.config.image_size}x{self.model.config.image_size}"
        self.stats.output_shape = self.config.output_shape

        return output
        
    def __call__(self, image, hidden_state=None, return_tensors='pt', **kwargs):
        return self.embed_image(image, hidden_state=hidden_state, return_tensors='pt', **kwargs)
        
