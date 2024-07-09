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
#      --dataset jaco_play \
#      --max-episodes 1 \
#      --max-steps 100
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

from nano_llm import NanoLLM, ChatHistory, remove_special_tokens
from nano_llm.utils import AttributeDict, convert_tensor, print_table
from nano_llm.vision import DatasetTypes, load_dataset


class VLAModel:
    def __init__(self, model="openvla/openvla-7b", actions={}, max_images=1, **kwargs):
        if isinstance(model, str):
            self.model = NanoLLM.from_pretrained(model, **kwargs)
        elif isinstance(model, NanoLLM):
            self.model = model
        else:
            raise TypeError(f"expected model as str or NanoLLM (was {type(model)})")
            
        self.chat = ChatHistory(model, **kwargs)
        self.prompt = "In: What action should the robot take to ${INSTRUCTION}?\nOut:▁" 
        self.instruction = kwargs.get('instruction', 'stop')
        
        self.num_images = 0
        self.max_images = max_images

        assert(model.has_vision)
        
        # setup action configs
        actions['normalized'] = dict(action=dict(
            normalized=True,
            q01=[-1.0] * 7,
            q99=[1.0] * 7,
        ))
        
        for key, scene in actions.items():
            action = scene['action']
            action['name'] = key
            actions[key] = action
            for stat_key, stat in action.items():
                if isinstance(stat, list):
                    action[stat_key] = np.array(stat)

        self.action_configs = actions
        self.actions = 'normalized'
        
        # map the tokenizer vocab range to discrete action bins
        self.bins = np.linspace(-1, 1, self.model.config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = self.config.vocab_size - self.config.pad_to_multiple_of
        
        # LLM warmup
        self.chat.append(role='user', text='What is 2+2?')
        logging.info(f"Warmup response:  '{self.model.generate(self.chat.embed_chat()[0], streaming=False)}'".replace('\n','\\n'))
        self.chat.reset()

    @property
    def config(self):
        return self.model.config
        
    @property
    def actions(self):
        return self._actions
        
    @actions.setter
    def actions(self, key):
        if isinstance(key, str):
            self._actions = self.action_configs[key]
        elif isinstance(key, dict):
            self._actions = key
        else:
            raise TypeError(f"actions can be set to str or dict (was {type(key)})")
            
    def embed(self, image, instruction='', prompt=None, **kwargs):
        if not instruction:
            instruction = self.instruction
         
        if not prompt:
            prompt = self.prompt
        
        prompt = prompt.replace('${INSTRUCTION}', instruction)
        
        logging.debug(f"{self.model.config.name} prompt:  `{prompt}`".replace('\n', '\\n'))

        self.chat.reset()
        
        self.chat.append('user', image=image)
        self.chat.append('user', text=prompt)
        
        return self.chat.embed_chat()[0]
        
    def predict_actions(self, image, actions={}, **kwargs):
        if not actions:
            actions = self.actions
        elif isinstance(actions, str):
            actions = self.action_configs[actions]
            
        num_actions = len(actions['q01'])

        reply = self.model.generate(
            self.embed(image, **kwargs),
            kv_cache=self.chat.kv_cache,
            detokenize=False,
            min_new_tokens=num_actions,
            max_new_tokens=num_actions,
            **kwargs,
        ).wait()
        
        return self.decode_actions(reply.tokens[:num_actions], actions, **kwargs)
        
    def decode_actions(self, tokens, actions={}, return_tensors='np', **kwargs):
        if not actions:
            actions = self.actions
        elif isinstance(actions, str):
            actions = self.action_configs[actions]
            
        #print(f'NanoLLM tokens ({len(tokens)}) ', tokens)
        
        # map from vocab bins back into action space (-1,1)
        pred_actions = self.vocab_size - convert_tensor(tokens, return_tensors='np')
        pred_actions = np.clip(pred_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        pred_actions = self.bin_centers[pred_actions]
        
        # denormalize the actions using robot coefficients
        if not actions.get('normalized', False):
            action_high, action_low = actions['q99'], actions['q01']
            pred_actions = np.where(
                actions['mask'],
                0.5 * (pred_actions + 1) * (action_high - action_low) + action_low,
                pred_actions,
            )
        #else:
        #    logging.warning('NanoLLM skipping denormalization')
            
        return convert_tensor(pred_actions, return_tensors=return_tensors, **kwargs)
     
    def __call__(self, image, **kwargs):
        return self.predict_actions(image, **kwargs)
           

                
if __name__ == "__main__":

    from nano_llm.utils import ArgParser, load_prompts, wrap_text
    from nano_llm.plugins import VideoSource, VideoOutput

    from termcolor import cprint
    from jetson_utils import cudaMemcpy, cudaToNumpy, cudaFont

    # parse args and set some defaults
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

    print(args)

    if args.dataset:
        dataset = load_dataset(args.dataset, **vars(args))
        
    if args.dump:
        dataset.dump(args.dump)
        sys.exit(0)
      
    if args.normalized:
        actions_key = 'normalized'
    else:
        actions_key = dataset.name
        
    np.set_printoptions(precision=4, linewidth=1000, edgeitems=30) 
      
    # load vision-language-action model
    model = NanoLLM.from_pretrained(**vars(args))

    assert(model.vla)
    assert(args.dataset)

    # gather performance and accuracy measurements
    stats = AttributeDict(
        latency = [],
        eval_latency = [],
        error = [],
    )
       
    def rmspe(y_true, y_pred):
        # https://stackoverflow.com/questions/53165807/how-to-calculate-rmspe-in-python-using-numpy
        return np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_true))))
    
    def nrmse(y_true, y_pred):
        # https://en.wikipedia.org/wiki/Root_mean_square_deviation#Normalization
        if args.normalized:
            y_range = 2.0
        else:
            y_range = np.max(y_true) - np.min(y_true)
        
        return np.sqrt(np.mean(np.square(y_true - y_pred))) / y_range
    
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
        
    def eval(step, i, actions):
        if not args.eval_model:
            return
        prompt = f"In: What action should the robot take to {step.instruction}?\nOut:" 
        time_begin = time.perf_counter()
        inputs = eval_processor(prompt, PIL.Image.fromarray(step.image)).to("cuda:0", dtype=eval_dtype)
        eval_actions = eval_model.predict_action(**inputs, unnorm_key=actions_key, do_sample=False)
        time_elapsed = time.perf_counter() - time_begin
        stats.error.append(nrmse(eval_actions, actions))
        stats.eval_latency.append(time_elapsed)
        print(f"eval {i}/{len(dataset)}  {time_elapsed*1000:.1f} ms  {1/time_elapsed:.2f} FPS  ~{1/np.mean(stats.eval_latency):.2f} FPS  {eval_actions}  error={stats.error[-1]:.4f} ~error={np.mean(stats.error):.4f}")
        
    # process the dataset
    for i, step in enumerate(dataset):
        time_begin = time.perf_counter()
        actions = model.vla.predict_actions(step.image, actions=actions_key, instruction=step.instruction)
        time_elapsed = time.perf_counter() - time_begin
        print_table(model.stats)
        stats.latency.append(time_elapsed)
        print(f"step {i}/{len(dataset)}  {time_elapsed*1000:.1f} ms  {1/time_elapsed:.2f} FPS  ~{1/np.mean(stats.latency):.2f} FPS  {actions}")
        eval(step, i, actions)
        stats.mean = mean_stats(stats)
        if args.save_stats:
            with open(args.save_stats, 'w') as file:
                json.dump(stats, file, indent=2)
                
    logging.success(f"Done processing {dataset.name} with {model.config.name}\n")
    print_table(stats.mean)
     
