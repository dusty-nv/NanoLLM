#!/usr/bin/env python3
#
# Example for vision/language action (VLA) model inference on video streams,
# with benchmarking and accuracy evaluation on Open-X Embodiment datasets.
#
# You can run it on video files or devices like this:
#
#    huggingface-cli download --local-dir /data/models/openvla-7b openvla/openvla-7b
#
#    python3 -m nano_llm.vision.vla \
#      --model openvla/openvla-7b \
#      --eval-model /data/models/openvla-7b \
#      --dataset bridge_orig \
#      --max-episodes 1
#
import os
import sys
import time
import json
import pprint
import logging
import subprocess

import PIL
import torch
import numpy as np

from transformers import AutoModelForVision2Seq, AutoProcessor

from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import AttributeDict, convert_tensor, print_table


class VLAModel:
    """
    Extension to NanoLLM for Vision/Language Action models.
    
    To get the :class:`NanoLLM`, use the ``vla.model`` property.
    To get the :class:`VLAModel`, use the ``model.vla`` property.
    """
    def __init__(self, model="openvla/openvla-7b", action_space={}, max_images=1, **kwargs):
        """
        VLAModel can either be instantiated directly, or through :func:`NanoLLM.from_pretrained()` -
        in which case it can be accessed through the returned ``model.vla`` property.
        """  
        if isinstance(model, str):
            self.model = NanoLLM.from_pretrained(model, **kwargs)
        elif isinstance(model, NanoLLM):
            self.model = model
        else:
            raise TypeError(f"expected model as str or NanoLLM (was {type(model)})")
            
        self.chat = ChatHistory(model, **kwargs)
        self.instruction = kwargs.get('instruction', 'stop')
        self.prompt_template = "In: What action should the robot take to ${INSTRUCTION}?\nOut:▁" 
        
        self.num_images = 0
        self.max_images = max_images

        assert(model.has_vision)
        
        # setup action configs
        if action_space is None:
            action_space = {}
            
        if not isinstance(action_space, dict):
            raise TypeError(f"expected action_space to be dict (was {type(action_space)})")
          
        action_spaces = {}
        
        if len(action_space) > 0:
            action = action_space[list(action_space.keys())[0]]
            if isinstance(action, dict) and 'action' in action:
                action_spaces = action_space
                action_space = 'normalized'
             
        action_spaces['normalized'] = dict(action=dict(
            normalized=True,
            mask=[False] * 7,
            q01=[-1.0] * 7,
            q99=[1.0] * 7,
        ))
        
        for key, space in action_spaces.items():
            action = AttributeDict(space['action'])
            action.name = key
            action_spaces[key] = action
            for stat_key, stat in action.items():
                if isinstance(stat, list):
                    action[stat_key] = np.array(stat)

        self.action_spaces = action_spaces
        self.action_space = action_space
        
        # map the tokenizer vocab range to discrete action bins
        self.bins = np.linspace(-1, 1, self.model.config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = self.config.vocab_size - self.config.pad_to_multiple_of
        
        # LLM warmup
        self.chat.append(role='user', text='What is 2+2?')
        logging.info(f"Warmup response:  '{self.model.generate(self.chat.embed_chat()[0], streaming=False, max_new_tokens=8)}'".replace('\n','\\n'))
        self.chat.reset()

    @property
    def config(self):
        """
        Return the base model's config dict from :attr:`NanoLLM.config`
        """
        return self.model.config
        
    @property
    def action_space(self):
        """
        Returns a dict defining the action space and normalization coefficients.
        """
        return self._action_space
        
    @action_space.setter
    def action_space(self, key):
        """
        Set the default action space to a dict or name defined by the model.
        """
        if isinstance(key, str):
            self._action_space = self.action_spaces[key]
        elif isinstance(key, dict):
            if not isinstance(key, AttributeDict):
                key = AttributeDict(key)
                
            key.setdefault('q01', key.min)
            key.setdefault('q99', key.max)
            key.setdefault('mask', [True] * (len(key.min)-1) + [False])

            for stat_key, stat in key.items():
                if isinstance(stat, list):
                    key[stat_key] = np.array(stat)
                    
            self._action_space = key
        else:
            raise TypeError(f"actions can be set to str or dict (was {type(key)})")
     
    @property
    def dof(self):
        """
        Degrees of freedom, or the number of action dimensions.
        """
        return len(self.action_space['q01'])
      
    def __call__(self, image, **kwargs):
        """
        Function-calling overload for predict_action()
        """
        return self.predict_action(image, **kwargs)
            
    def predict_action(self, image, instruction='', action_space={}, streaming=False, **kwargs):
        """
        Predict the actions for a given image using the latest prompt.
        
        If streaming=True, then a :class:`StreamingResponse` iterator
        will be returned, and the tokens need decoded into actions::
        
            stream = vla.predict_action(image, instruction)
            
            for token in stream:
                actions = vla.decode_actions(stream.tokens, action_space)
        """
        if not action_space:
            action_space = self.action_space
        elif isinstance(action_space, str):
            action_space = self.action_spaces[action_space]

        '''
        for i in range(self.dof-1, 0, -1):
            if actions.mask[i]:
                break
            else:
                num_actions -= 1
        '''
        
        stream = self.model.generate(
            self.embed(image, instruction=instruction, **kwargs),
            kv_cache=self.chat.kv_cache,
            detokenize=False,
            min_new_tokens=self.dof,
            max_new_tokens=self.dof,
            action_space=action_space,
            **kwargs,
        )

        if streaming:
            return stream
        else:
            return stream.wait()    

    def decode_action(self, tokens, action_space={}, return_tensors='np', **kwargs):
        """
        Map tokens into action space, denormalizing them if needed.
        """
        if not action_space:
            action_space = self.action_space
        elif isinstance(action_space, str):
            action_space = self.action_spaces[actions]
          
        if not tokens:
            return None
        
        num_tokens = min(len(tokens), self.dof) 
        tokens = tokens[:num_tokens]
 
        # map from vocab bins back into action space (-1,1)
        pred_actions = self.vocab_size - convert_tensor(tokens, return_tensors='np')
        pred_actions = np.clip(pred_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        pred_actions = self.bin_centers[pred_actions]
        
        # denormalize the actions using robot coefficients
        if not action_space.get('normalized', False):
            action_high, action_low = action_space['q99'][:num_tokens], action_space['q01'][:num_tokens]
            pred_actions = np.where(
                action_space['mask'][:num_tokens],
                0.5 * (pred_actions + 1) * (action_high - action_low) + action_low,
                pred_actions,
            )
        #else:
        #    logging.warning('NanoLLM skipping denormalization')
            
        return convert_tensor(pred_actions, return_tensors=return_tensors, **kwargs)
           
    def embed(self, image, instruction='', prompt_template=None, **kwargs):
        """
        Embed the image and instruction into a chat generation using the prompt template.
        """
        if not instruction:
            instruction = self.instruction
         
        if not prompt_template:
            prompt_template = self.prompt_template
        
        prompt = prompt_template.replace('${INSTRUCTION}', instruction)
        
        logging.debug(f"{self.model.config.name} prompt:  `{prompt}`".replace('\n', '\\n'))

        self.chat.reset()
        
        self.chat.append('user', image=image)
        self.chat.append('user', text=prompt)
        
        return self.chat.embed_chat()[0]

           
                
if __name__ == "__main__":

    from nano_llm.utils import ArgParser
    from nano_llm.datasets import DatasetTypes, load_dataset

    parser = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input', 'video_output'])
    
    parser.add_argument("--eval-model", type=str, default=None, help="path to the original HuggingFace model to enable error comparison")
    parser.add_argument("--dataset", type=str, default=None, required=True, help=f"path or name of the dataset to load")
    parser.add_argument("--dataset-type", type=str, default=None, choices=list(DatasetTypes.keys()), help=f"type of the dataset to load")
    parser.add_argument("--dump", type=str, default=None, help="dump the OpenX dataset to a directory of individual image files")
    parser.add_argument("--max-images", type=int, default=1, help="the number of video frames to keep in the history")
    parser.add_argument("--max-episodes", type=int, default=None, help="the maximum number of episodes from the dataset to process")
    parser.add_argument("--max-steps", type=int, default=None, help="the maximum number of frames to process across all episodes")
    parser.add_argument("--normalized", action='store_true', help="disable denormalization of the output actions (will output [-1,1])")
    parser.add_argument("--save-stats", type=str, default=None, help="path to json file for saving performance and accuracy measurements")
    
    args = parser.parse_args()

    if not args.model:
        args.model = "openvla/openvla-7b"

    if args.dataset:
        dataset = load_dataset(**vars(args))
        
    if args.dump:
        dataset.dump(args.dump)
        sys.exit(0)
   
    np.set_printoptions(floatmode='fixed', precision=5, linewidth=1000, edgeitems=30) 
      
    # load vision-language-action model
    model = NanoLLM.from_pretrained(**vars(args))
    vla = model.vla
    
    assert(vla)
    assert(args.dataset)

    # gather performance and accuracy measurements
    stats = AttributeDict(
        latency = [],
        eval_latency = [],
        eval_error = [],
        quant_error = [],
        error = [],
    )
       
    if args.normalized:
        vla.action_space = 'normalized'
    elif hasattr(dataset, 'action_space'):
        vla.action_space = dataset.action_space
    elif 'bridge' in dataset.config.name:
        vla.action_space = 'bridge_orig'
    else:
        vla.action_space = dataset.config.name

    action_range = vla.action_space.q99 - vla.action_space.q01

    logging.info(f"Action space:\n{vla.action_space}")
    logging.info(f"Action range:\n{action_range}")
    
    def rmspe(y_true, y_pred):
        # https://stackoverflow.com/questions/53165807/how-to-calculate-rmspe-in-python-using-numpy
        return np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_true))))
    
    def nrmse(y_true, y_pred, y_range=None):
        # https://en.wikipedia.org/wiki/Root_mean_square_deviation#Normalization
        if args.normalized:
            y_range = 2.0
        elif y_range is None:
            y_range = np.max(y_true) - np.min(y_true)
        
        if isinstance(y_range, (list, np.ndarray)):
            return np.sqrt(np.nanmean(np.square((y_true - y_pred) / y_range)))  # y_range[y_range == 0.0] = 1.0
        else:    
            return np.sqrt(np.nanmean(np.square(y_true - y_pred))) / y_range
    
    def mean_stats(stats):
        mean = AttributeDict(timesteps=len(stats.latency))
        for key, samples in stats.items():
            if not samples or not isinstance(samples, list):
                continue
            mean[key] = np.mean(samples)
            if 'latency' in key:
                mean[key.replace('latency', 'fps')] = 1 / mean[key]
        return mean
        
    # load original unquantized model for eval
    if args.eval_model:
        eval_dtype = torch.float16 #torch.bfloat16
        logging.info(f"loading eval model with HF Transformers from {args.eval_model}  ({eval_dtype})")
        eval_processor = AutoProcessor.from_pretrained(args.eval_model, trust_remote_code=True) 
        eval_model = AutoModelForVision2Seq.from_pretrained(
            args.eval_model,
            #attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=eval_dtype, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        ).to("cuda:0")
        logging.success(f"loaded eval model {eval_model.__class__.__name__} from {args.eval_model}  ({eval_dtype})")
        logging.debug(f"action bins difference:  {np.sum(np.abs(model.vla.bins - eval_model.bins))}")
        logging.debug(f"action bin centers diff: {np.sum(np.abs(model.vla.bin_centers - eval_model.bin_centers))}")
        assert(model.vla.vocab_size == eval_model.vocab_size)
        
    def eval(step, i, actions, gt):
        if not args.eval_model:
            return
        prompt = f"In: What action should the robot take to {step.instruction}?\nOut:" 
        time_begin = time.perf_counter()
        inputs = eval_processor(prompt, PIL.Image.fromarray(step.images[0])).to("cuda:0", dtype=eval_dtype)
        eval_actions = eval_model.predict_action(**inputs, unnorm_key=vla.action_space, do_sample=False)
        time_elapsed = time.perf_counter() - time_begin
        stats.quant_error.append(nrmse(eval_actions, actions, y_range=action_range))
        stats.eval_error.append(nrmse(gt, eval_actions, y_range=action_range))
        stats.eval_latency.append(time_elapsed)
        print(f"eval {i}  {time_elapsed*1000:.1f} ms  {1/time_elapsed:.2f} FPS  ~{1/np.mean(stats.eval_latency):.2f} FPS  {eval_actions}  error={stats.eval_error[-1]:.4f} ~{np.mean(stats.eval_error):.4f}  q_err={stats.quant_error[-1]:.4f} ~{np.mean(stats.quant_error):.4f}")
        
    # process the dataset
    for i, step in enumerate(dataset):
        time_begin = time.perf_counter()
        actions = vla.predict_action(step.images[0], instruction=step.instruction, streaming=False)
        time_elapsed = time.perf_counter() - time_begin
        print_table(model.stats)
        stats.latency.append(time_elapsed)
        stats.error.append(nrmse(step.action, actions, y_range=action_range))
        print(f"step {i}  {time_elapsed*1000:.1f} ms  {1/time_elapsed:.2f} FPS  ~{1/np.mean(stats.latency):.2f} FPS  {actions}  error={stats.error[-1]:.4f} ~{np.mean(stats.error):.4f}")
        eval(step, i, actions, step.action)
        print(f"gt   {i}                                {step.action}")  
        stats.mean = mean_stats(stats)
        if args.save_stats:
            with open(args.save_stats, 'w') as file:
                json.dump(stats, file, indent=2)
                
    logging.success(f"Done processing {dataset.config.name} with {model.config.name}\n")
    print_table(stats.mean)
     
