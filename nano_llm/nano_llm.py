#!/usr/bin/env python3
import os
import re
import time
import json
import shutil
import logging

import torch
import numpy as np

from transformers import AutoConfig

from .vision import CLIPImageEmbedding, MMProjector
from .utils import AttributeDict, convert_tensor, download_model, default_model_api, print_table


class NanoLLM():
    @staticmethod
    def from_pretrained(model, api=None, **kwargs):
        """
        Load a model from the given path or download it from HuggingFace Hub.
        Various inference and quantization APIs are supported, such as MLC and AWQ.
        If the API isn't explicitly specified, it will be inferred from the type of model.
        
        Base class for local LLM APIs. It defines common Huggingface-like interfaces for
        model loading, text generation, tokenization, embeddings, and streaming.
        It also supports multimodal vision models like Llava and generating image embeddings with CLIP.
    
        Args:
          model (str): either the path to the model, or HuggingFace model repo/name.
          api (str): the model backend API to use:  'auto_gptq', 'awq', 'mlc', or 'hf'
                       if left as None, it will attempt to be automatically determined.

          quantization (str): for AWQ or MLC, either specify the quantization method,
                              or the path to the quantized model (AWQ and MLC API's only)

          vision_model (str): for VLMs, override the vision embedding model 
                              (typically `openai/clip-vit-large-patch14-336 <https://huggingface.co/openai/clip-vit-large-patch14-336>`_).
                              Otherwise, it will use the CLIP variant from the config.
                                
        Returns:
          A loaded `NanoLLM` model instance using the determined API.
        """
        if os.path.isdir(model) or os.path.isfile(model):
            model_path = model
            model_name = os.path.basename(model_path)
        else:
            model_path = download_model(model, **kwargs)
            model_name = os.path.basename(model)
            
        if not api:
            api = default_model_api(model_path, kwargs.get('quantization'))
        
        kwargs['name'] = model_name
        kwargs['api'] = api
        
        logging.info(f"loading {model_path} with {api.upper()}")
        load_begin = time.perf_counter()
        
        # doing this imports here avoid circular import, and makes it so these
        # dependencies are only needed if they are actually used to load a model
        if api == 'auto_gptq':
            from nano_llm.models import AutoGPTQModel
            model = AutoGPTQModel(model_path, **kwargs)
        elif api == 'awq':
            from nano_llm.models import AWQModel
            model = AWQModel(model_path, **kwargs)
        elif api == 'mlc':
            from nano_llm.models import MLCModel
            model = MLCModel(model_path, **kwargs)
        elif api == 'hf':
            from nano_llm.models import HFModel
            model = HFModel(model_path, **kwargs)
        else:
            raise ValueError(f"invalid API: {api}")

        # moved CLIP to after LLM is loaded because of MLC CUDA errors when running in subprocess
        model.init_vision(**kwargs)  
        model.config.load_time = time.perf_counter() - load_begin
        
        print_table(model.config)
        print('')
        
        return model
     
    def generate(self, inputs, streaming=True, **kwargs):
        """
        Generate output from input text, tokens, or an embedding.
        For detailed kwarg descriptions, see `transformers.GenerationConfig <https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig>`_.
        
        Args:
          inputs (str|list[int]|torch.Tensor|np.ndarray): the prompt string, token IDs, or embedding.
          streaming (bool): if true (default), an iterator will be returned that outputs
                              one token at a time.  Otherwise, return the full response.
          max_new_tokens (int): the number of tokens to output in addition to the prompt (default: 128)
          min_new_tokens (int): force the model to generate a set number of output tokens (default: -1)
          do_sample (bool): if True, temperature/top_p will be used.  Otherwise, greedy search (default: False)
          repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty (default: 1.0)  
          temperature (float): randomness token sampling parameter (default=0.7, only used if ``do_sample=True``)
          top_p (float): if set to float < 1 and do_sample=True, only the smallest set of most probable tokens
                           with probabilities that add up to top_p or higher are kept for generation (default 0.95)
          stop_tokens (list[int]): defaults to EOS token ID
          kv_cache (np.ndarray): previous kv_cache that the inputs will be appended to.  By default, a blank kv_cache 
                                will be created for each generation (i.e. a new chat).  This generation's kv_cache
                                will be set in the returned :class:`StreamingResponse` iterator after the request is complete.

        Returns:
          An asynchronous :class:`StreamingResponse` iterator (when ``streaming=True``) that outputs one decoded token string at a time.
          Otherwise, this function blocks and a string containing the full reply is returned after it's been completed.
        """
        raise NotImplementedError("use LLM.from_pretrained() as opposed to instantiating an LLM object directly")

    def tokenize(self, text, add_special_tokens=False, dtype=np.int32, return_tensors='np', **kwargs):
        """
        Tokenize the given string and return the encoded token ID's.
        
        Args:
          text (str): the text to tokenize.
          add_special_tokens (str): if BOS/EOS tokens (like ``<s>`` or ``<|endoftext|>``) should automatically be added (default False)
          dtype (type): the numpy or torch datatype of the tensor to return.
          return_tensors (str): ``'np'`` to return a `np.ndarray` or ``'pt'`` to return a `torch.Tensor`
          kwargs:  additional arguments forwarded to the HuggingFace `transformers.AutoTokenizer <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_ encode function.
          
        Returns:
          The token ID's with the tensor type as indicated by `return_tensors` (either `'np'` for `np.ndarray`
          or `'pt'` for `torch.Tensor`) and datatype as indicated by `dtype` (by default ``int32``)
        """
        return self.tokenizer(
            text, 
            add_special_tokens=add_special_tokens, 
            return_tensors=return_tensors,
            **kwargs
        ).input_ids.astype(dtype, copy=False)
    
    def detokenize(self, tokens, skip_special_tokens=False, **kwargs) -> str:
        """
        Detokenize the given token ID's and return the decoded string.
        
        Args:
          tokens (list[int], np.ndarray, torch.Tensor): the array of token ID's
          skip_special_tokens (bool): if special tokens (like BOS/EOS) should be supressed from the output or not (default false)
          kwargs:  additional arguments forwarded to the HuggingFace `transformers.AutoTokenizer <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_ decode function.
          
        Returns:
          The string containing the decoded text.
        """
        return self.model.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens, **kwargs)
        
    def embed_text(self, text, add_special_tokens=False, use_cache=False, return_tensors='np', **kwargs):
        """
        Tokenize the string with :meth:`NanoLLM.tokenize` and return its embedding as computed by :meth:`NanoLLM.embed_tokens`
        
        Args:
          text (str): the text to tokenize and embed.
          add_special_tokens (str): if BOS/EOS tokens (like ``<s>``, ``<|endoftext|>``) should automatically be added (default False)
          use_cache (bool): if True, the text embedding will be cached and returned without additional computation if
                            the same string was already embedded previously.  This is useful for things like the system prompt
                            that are relatively static, but probably shouldn't be used for dynamic user inputs that are unlikely
                            to be re-used again (leading to unnecessarily increased memory usage).  The default is false.
          return_tensors (str): ``'np'`` to return a `np.ndarray` or ``'pt'`` to return a `torch.Tensor`
          kwargs:  additional arguments forwarded to :meth:`NanoLLM.tokenize` and the HuggingFace `transformers.AutoTokenizer <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_ 
          
        Returns:
          The embedding with the tensor type as indicated by `return_tensors` (either `'np'` for `np.ndarray`
          or `'pt'` for `torch.Tensor`) with ``float32`` data.
        """
        # TODO migrate this from models.MLC and have the text cache in this class for all model APIs
        raise NotImplementedError("embed_text() not implemented for this model")
        
    def embed_tokens(self, tokens, return_tensors='np', **kwargs):
        """
        Compute the token embedding and return its tensor.
        
        Args:
          tokens (list[int], np.ndarray, torch.Tensor): the array of token ID's
          return_tensors (str): ``'np'`` to return a `np.ndarray` or ``'pt'`` to return a `torch.Tensor`
          
        Returns:
          The embedding with the tensor type as indicated by `return_tensors` (either `'np'` for `np.ndarray`
          or `'pt'` for `torch.Tensor`) with ``float32`` data.
        """
        raise NotImplementedError("embed_tokens() not implemented for this model")
       
    def embed_image(self, image, crop=None, return_tensors='pt', return_dict=False, **kwargs):
        """
        Compute the embedding of an image (for multimodel models with a vision encoder like CLIP),
        and apply any additional projection layers as specified by the model.
        
        Args:
          image (pil.Image, np.ndarray, torch.Tensor, jetson.utils.cudaImage, __cuda_array_interface__): the image
          crop (bool): center-crop the image to square resolution instead of resizing the aspect ratio
          return_tensors (str): ``'np'`` to return a `np.ndarray` or ``'pt'`` to return a `torch.Tensor` (on the GPU)
          return_dict (bool): if true, return a dict including the vision encoder's `hidden_state` and `embedding`
          kwargs: additional arguments forwarded to the vision encoder (`nano_llm.vision.CLIPImageEmbedding`)
        
        Returns:
          The embedding with the tensor type as indicated by `return_tensors` (either `'np'` for `np.ndarray`
          or `'pt'` for `torch.Tensor`), or a dict containing the embedding and vision encoder's `hidden_state`
          if ``return_dict=True``.
        """  
        assert(self.has_vision)
        
        if crop is None:
            crop = (self.vision_scaling == 'crop')

        output = self.vision(image, crop=crop, hidden_state=self.config.mm_vision_select_layer, return_dict=return_dict)
        
        embedding = output.hidden_state if return_dict else output
        embedding = self.mm_projector(embedding[:, 1:])

        logging.debug(f"image_embedding  shape={embedding.shape}  dtype={embedding.dtype}  device={embedding.device}")
        
        if return_dict:
            output.embedding = embedding
            for key in output:
                output[key] = convert_tensor(output[key], return_tensors=return_tensors)
            return output
        else:
            return convert_tensor(embedding, return_tensors=return_tensors)
        
    def __init__(self, model_path, **kwargs):
        #: HuggingFace `transformers.AutoTokenizer <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_ instance used for tokenization/detokenization.
        self.tokenizer = None
        
        #: Dict containing the model configuration (inspect it on the HuggingFace model card)
        self.config = AttributeDict()
        
        #: The local path to the model config file (``config.json``)
        self.config_path = os.path.join(model_path, 'config.json')
        
        #: The local path to the model checkpoint/weights in HuggingFace format.
        self.model_path = model_path

        # load the config file
        if os.path.isfile(self.config_path):
            with open(self.config_path) as config_file:
                self.config = AttributeDict(json.load(config_file))
        else:
            logging.warning(f"could not find model config file at {self.config_path}")
            self.config = AttributeDict()

        self.config.name = kwargs.get('name')
        self.config.api = kwargs.get('api')
        
        model_type = self.config.model_type.lower()
        
        #: Dict containing the latest generation performance statistics.
        self.stats = AttributeDict()
        
        #: True if this is a multimodal vision/language model.
        self.has_vision = 'llava' in model_type
        
        # patch the config to change llava to llama so the quant tools handle it
        if self.has_vision:
            if 'stablelm' in model_type:
                self.patch_config(model_type='stablelm_epoch')
            elif 'phi' in model_type:
                self.patch_config(model_type='phi')
            else:
                self.patch_config(model_type='llama')
        else:
            name_or_path = self.config.get('_name_or_path')
            if name_or_path:
                self.has_vision = 'llava' in name_or_path.lower()

        for arch in self.config.get('architectures', []):
            if 'llava' in arch.lower() or 'bunny' in arch.lower():
                self.has_vision = True

        if self.config.model_type == 'bunny-stablelm':
            self.patch_config(model_type='stablelm_epoch')
        elif self.config.model_type == 'bunny-phi':
            self.patch_config(model_type='phi')
     
    def patch_config(self, **kwargs):
        # Update the original HF model's config.json with different settings from the provided kwargs.
        # The original will be saved under the same directory to 'config.json.backup'
        backup_path = self.config_path + '.backup'
        
        if not os.path.isfile(backup_path):
            logging.info(f"backing up original model config to {backup_path}")
            shutil.copyfile(self.config_path, backup_path)
            
        logging.info(f"patching model config with {kwargs}")
        
        patched_config = self.config #.copy()
        patched_config.update(kwargs)

        with open(self.config_path, 'w') as config_file:
            json.dump(patched_config, config_file, indent=2)
                
    def init_vision(self, **kwargs):
        # Init vision embedding/projection models for VLMs like llava, MiniGPT-4, ect.
        if not self.has_vision:
            return
           
        # load the image embedding model
        self.vision = CLIPImageEmbedding.from_pretrained(
            kwargs.get('vision_model') if kwargs.get('vision_model')
            else self.config.mm_vision_tower,
            dtype=torch.float16,
        ) 
        
        # create image embedding projection model
        self.mm_projector = MMProjector.from_pretrained(self, self.vision.dtype)
        
        # default to cropping enabled
        self.vision_scaling = kwargs.get('vision_scaling')
        
        if self.vision_scaling is None:
            self.vision_scaling = 'crop'
            
        
