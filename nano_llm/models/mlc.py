#!/usr/bin/env python3
import os
import sys
import tvm
import time
import math
import json
import glob
import shutil
import queue
import threading
import subprocess
import random
import logging

import torch
import numpy as np

from tvm.runtime.relax_vm import VirtualMachine
from transformers import AutoTokenizer, AutoConfig
from cuda.cudart import cudaMemcpy, cudaMemcpyKind, cudaDeviceSynchronize

from nano_llm import NanoLLM, StreamingResponse, KVCache
from nano_llm.utils import AttributeDict, ends_with_token


class MLCModel(NanoLLM):
    """
    MLC model (https://github.com/mlc-ai/mlc-llm)
    """
    def __init__(self, model_path, quantization='q4f16_ft', max_context_len=None, **kwargs):
        """
        Parameters:
        
          model_path (str) -- the original model on HuggingFace - used for getting
                              the tokenizer and original model config.  If this is a 
                              Llama2 model, it will be automatically set if Llama/ect
                              is in the path. Otherwise, this should be set to the name.
                              
          quantization (str) -- either a directory path containing the mlc_llm quantized model,
                         or the quantization method to use (q4f16_1, q4f16_ft, ect)          
                         If a path, there should be a .so in this dir, with params/ 
                         directory under it containing the weights and MLC config.
                         
          max_context_len (int) -- override the model's default context window length (in tokens)
                                   This should include space for model output (up to --max-new-tokens)
                                   Lowering this from the default (e.g. 4096 for Llama) will reduce 
                                   memory usage. If None, it's inherited from the model's max length.
        """
        super(MLCModel, self).__init__(model_path, **kwargs)

        # 20240223: the 'stablelm_epoch' model type was re-named in transformers to 'stablelm'
        if self.config.model_type == 'stablelm':
            self.patch_config(model_type='stablelm_epoch', norm_eps=1e-05, rope_pct=0.25)
            
        # perform quantization if needed
        if not quantization:
            quantization = 'q4f16_ft'
            
        quant = MLCModel.quantize(self.model_path, self.config, method=quantization, max_context_len=max_context_len, **kwargs)
            
        self.config.quant = quant.split('-')[-1]  # recover the quant method        
        self.quant_path = quant
        
        # the weights location used to be under 'params', but then moved to the model dir
        self.weight_path = os.path.join(self.quant_path, 'params')
        
        if not os.path.isdir(self.weight_path):
            self.weight_path = self.quant_path

        # initialize tvm device
        self.device = tvm.runtime.cuda(0)  # tvm.runtime.Device(tvm.runtime.Device.kDLCUDAManaged, 0)
        assert(self.device.exist) # this is needed to initialize CUDA?
        logging.info(f"device={self.device}, name={self.device.device_name}, compute={self.device.compute_version}, max_clocks={self.device.max_clock_rate}, multiprocessors={self.device.multi_processor_count}, max_thread_dims={self.device.max_thread_dimensions}, api_version={self.device.api_version}, driver_version={self.device.driver_version}")
        
        self.cuda_stream = self.device.create_raw_stream()
        self.device.set_raw_stream(self.cuda_stream)
        
        # load model config
        with open(os.path.join(self.weight_path, 'mlc-chat-config.json'), 'r') as file:
            config = json.load(file)
        
        #self.config.name = config['local_id']  # model_name
        self.config.type = config.get('model_category', config.get('model_type'))  # 'conv_template'
        self.config.max_length = config.get('max_window_size', config.get('context_window_size'))
        self.config.prefill_chunk_size = config.get('prefill_chunk_size', -1)
        self.config.vocab_size = config['vocab_size']
        
        # load model's dynamic library
        def find_module():
            module_name = os.path.basename(quant) + '-cuda.so'
            module_paths = [
                os.path.join(self.quant_path, module_name),
                os.path.join(self.weight_path, module_name),
            ]
            for module_path in module_paths:
                if os.path.isfile(module_path):
                    return module_path
            raise IOError(f"MLC couldn't find {module_path}")
            
        self.module_path = find_module()
        logging.info(f"loading {self.config.name} from {self.module_path}")
        load_time_begin = time.perf_counter()
        self.module = tvm.runtime.load_module(self.module_path)
        
        self.vm = self.module['vm_load_executable']()
        
        self.vm['vm_initialization'](
            self.device.device_type, self.device.device_id,
            VirtualMachine.POOLED_ALLOCATOR,
            tvm.runtime.Device.kDLCPU, 0,  # kDLCUDAManaged kDLCUDAHost  kDLCUDA
            VirtualMachine.POOLED_ALLOCATOR
        )    
        
        # model metadata (optional)
        self.metadata = None
        
        if self.vm.implements_function('_metadata'):
            self.metadata = json.loads(self.vm['_metadata']())
            logging.debug(f"{self.config.name} memory usage:  {self.metadata['memory_usage']}")
        else:
            logging.warning(f"model library {self.module_path} was missing metadata")
        
        # embedding/generation functions
        if self.vm.implements_function('embed'):
            self._embed = self.vm['embed']
            self.has_embed = True
        else:
            self.has_embed = False
            logging.warning(f"{self.config.name} is missing embed() function in {self.module_path}")
            
        self._decode = self.vm['decode']
        
        # prefill functions
        prefill_functions = [
            'prefill_with_embed',
            'prefill',
        ]
        
        for prefill_func in prefill_functions:
            if self.vm.implements_function(prefill_func):
                self._prefill = self.vm[prefill_func]
                self.prefill_type = prefill_func
                break
                
        if not hasattr(self, '_prefill'):
            raise RuntimeError(f"couldn't find any of the following functions in {self.module_path} - {prefill_functions}")
        
        logging.debug(f"using {self.prefill_type}() from {self.module_path}")

        # KV cache manipulation functions
        create_kv_cache_functions = [
            'create_kv_cache',
            'create_flashinfer_paged_kv_cache',
            'create_tir_paged_kv_cache',
            '_initialize_effect'
        ]
        
        for kv_cache_func in create_kv_cache_functions:
            if self.vm.implements_function(kv_cache_func):
                self._kv_cache_create = self.vm[kv_cache_func]
                self.kv_cache_type = kv_cache_func #[len('create_'):]
                self.kv_cache_paged = 'paged' in self.kv_cache_type
                break
        
        if not hasattr(self, '_kv_cache_create'):
            raise RuntimeError(f"couldn't find any of the following functions in {self.module_path} - {create_kv_cache_functions}")

        logging.debug(f"using {self.kv_cache_type}() from {self.module_path}")
 
        if self.kv_cache_paged:
            self._kv_cache_clear = tvm.get_global_func('vm.builtin.attention_kv_cache_array_clear')
            self._kv_cache_pop = tvm.get_global_func('vm.builtin.paged_attention_kv_cache_popn')  # 'vm.builtin.kv_state_popn')
            self._kv_cache_add_sequence = tvm.get_global_func('vm.builtin.paged_attention_kv_cache_add_sequence')  # 'vm.builtin.kv_state_add_sequence') 
            self._kv_cache_remove_sequence = tvm.get_global_func('vm.builtin.paged_attention_kv_cache_remove_sequence')  # 'vm.builtin.kv_state_remove_sequence')    
            self._kv_cache_begin_forward = tvm.get_global_func('vm.builtin.paged_attention_kv_cache_begin_forward')  # 'vm.builtin.kv_state_begin_forward')
            self._kv_cache_end_forward = tvm.get_global_func('vm.builtin.paged_attention_kv_cache_end_forward')  # 'vm.builtin.kv_state_end_forward')
            self.backtracking_kv = True
        else:
            if self.vm.implements_function('reset_kv_cache'):
                self._kv_cache_clear = self.vm['reset_kv_cache']
                self.backtracking_kv = False
            else:
                self._kv_cache_clear = tvm.get_global_func('vm.builtin.attention_kv_cache_array_clear')
                self.backtracking_kv = True
                
            self._kv_cache_pop = tvm.get_global_func('vm.builtin.attention_kv_cache_array_popn')
            self._kv_cache_append = tvm.get_global_func('vm.builtin.attention_kv_cache_append')
            self._kv_cache_update = tvm.get_global_func('vm.builtin.attention_kv_cache_update')
            self._kv_cache_view = tvm.get_global_func('vm.builtin.attention_kv_cache_view')

        self._sample_top_p_from_prob = tvm.get_global_func('vm.builtin.sample_top_p_from_prob')
        self._sample_top_p_from_logits = tvm.get_global_func('vm.builtin.sample_top_p_from_logits')
        
        self._apply_repetition_penalty = tvm.get_global_func('vm.builtin.apply_repetition_penalty')
        self._apply_softmax_with_temperature = tvm.get_global_func('vm.builtin.apply_softmax_with_temperature')

        self.kv_caches = []  # cache of KV caches to reuse because they take ~100ms to allocate
        
        # param loading functions
        self._load_cache = tvm.get_global_func('vm.builtin.ndarray_cache.load')
        self._load_params = tvm.get_global_func('vm.builtin.param_array_from_cache')
        self._clear_cache = tvm.get_global_func('vm.builtin.ndarray_cache.clear')

        # load model weights
        self._load_cache(self.weight_path, self.device.device_type, self.device.device_id)
        
        if self.metadata:
            self._load_params_by_name = tvm.get_global_func('vm.builtin.param_array_from_cache_by_name')
            self.params = self._load_params_by_name([param['name'] for param in self.metadata['params']])
        else:
            self.params = self._load_params('param', -1)
            
        self._clear_cache() # after we get params, it is safe to simply clear the cached version
        self.config.load_time = time.perf_counter() - load_time_begin
        
        # add up the total weight size
        self.params_size = 0

        for param in self.params:
            self.params_size += math.prod(param.shape) * np.dtype(param.dtype).itemsize

        self.config.params_size = self.params_size / (1024 * 1024)
        
        # create multithreading
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True).start()        
    
    @staticmethod
    def quantize(model, config, method='q4f16_ft', max_context_len=None, output='/data/models/mlc/dist', **kwargs):
        """
        Quantize a model with the given method.  It will be saved under the output directory,
        in a subdirectory based on the model name and quant method (Llama-2-7b-chat-hf-q4f16_ft)
        """
        if not max_context_len:
            max_context_len = config.max_position_embeddings

        model_name = kwargs.get('name', os.path.basename(model))
        model_path = os.path.join(output, 'models', model_name)
        quant_path = os.path.join(output, f"{model_name}-ctx{max_context_len}", f"{model_name}-{method}")

        config_paths = [
            os.path.join(quant_path, 'mlc-chat-config.json'),
            os.path.join(quant_path, 'params/mlc-chat-config.json')
        ]
        
        for config_path in config_paths:
            if os.path.isfile(config_path):
                try:
                    with open(config_path) as config_file:
                        config_json = json.load(config_file)
                        return quant_path
                except Exception as err:
                    logging.warning(f"Rebuilding {model_name} after exception occurred trying to load {config_path}\n{err}")
                    pass
        
        if not os.path.isdir(model_path):
            os.symlink(model, model_path, target_is_directory=True)
                
        if config.model_type == 'phi' or config.model_type == 'gemma':
            cmd = f"mlc_chat convert_weight {model} --quantization {method} --output {quant_path} && "
            cmd += f"mlc_chat gen_config {model} --quantization {method} --conv-template LM "
            cmd += f"--max-batch-size 1 --context-window-size {max_context_len} --output {quant_path} "
            cmd += f"&& mlc_chat compile {quant_path} --device cuda --opt O3 --output {quant_path}/{model_name + '-' + method}-cuda.so"
        else:
            cmd = f"python3 -m mlc_llm.build --model {model_path} --quantization {method} "
            cmd += f"--target cuda --use-cuda-graph --use-flash-attn-mqa --sep-embed " 
            cmd += f"--max-seq-len {max_context_len} --artifact-path {os.path.join(output,f'{model_name}-ctx{max_context_len}')} "

            if len(glob.glob(os.path.join(model_path, '*.safetensors'))) > 0:
                cmd += "--use-safetensors "

        logging.info(f"running MLC quantization:\n\n{cmd}\n\n")
        subprocess.run(cmd, executable='/bin/bash', shell=True, check=True)  
        
        return quant_path
        
    def embed_tokens(self, tokens, return_tensors='np', **kwargs):  # pt, np, tvm
        """
        Generate embedding from token IDs
        """
        if not self.has_embed:
            raise RuntimeError(f"{self.config.name} does not have embed() in {self.module_path}")
            
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
        
        if not isinstance(input, tvm.runtime.ndarray.NDArray):
            if isinstance(tokens, list):
                tokens = np.asarray([tokens], dtype=np.int32)
            elif not isinstance(tokens, np.ndarray):
                raise TypeError(f"NanoLLM.embed_tokens() expects a list, np.ndarray, tvm.ndarray, or torch.Tensor (was {type(input)})")
                
        if isinstance(tokens, np.ndarray):
            tokens = tvm.nd.array(tokens.astype(np.int32), self.device)

        time_begin = time.perf_counter()
        embedding = self._embed(tokens, self.params)
        self.stats.embed_time = time.perf_counter() - time_begin
        #self.device = embedding.device
        
        if return_tensors == 'np':
            embedding = embedding.numpy()
            
        return embedding
        
    def generate(self, inputs, streaming=True, detokenize=True, **kwargs):
        """
        Generate output from input text, tokens, or an embedding.
        For detailed kwarg descriptions, see `transformers.GenerationConfig <https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig>`_.
        
        Args:
        
          inputs (str|ndarray): Text or embedding inputs to the model/
          
          streaming (bool): If True, an iterator will be returned that returns text chunks.
                            Otherwise, this function will block and return the generated text.
                              
          functions(list[callable]): Dynamic functions or plugins to run inline with token generation 
                                     for things like function calling, guidance, token healing, ect.
                                     These will be passed the text generated by the LLM so far, and any
                                     additional text that these return will be added to the chat.

          max_new_tokens (int): The number of tokens to output in addition to the prompt (default: 128)
          min_new_tokens (int): Force the model to generate a set number of output tokens (default: -1)
          detokenize (bool): If ``True`` (the default), text will be returned (otherwise ``list[int]`` of token ID's)
          do_sample (bool): If ``True``, temperature/top_p will be used.  Otherwise, greedy search (default: ``False``)
          repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty (default: 1.0)
          temperature (float): Randomness token sampling parameter (default=0.7, only used if ``do_sample=True``)
          top_p (float): If set to float < 1 and ``do_sample=True``, only the smallest set of most probable tokens.
                           with probabilities that add up to top_p or higher are kept for generation (default 0.95)
          stop_tokens (list[int]|list[str]): Stop generation if the bot produces tokens or text from this list (defaults to EOS token ID)
          kv_cache (np.ndarray): Previous kv_cache that the inputs will be appended to.  By default, a blank kv_cache 
                                will be created for each generation (i.e. a new chat).  This generation's kv_cache
                                will be set in the returned :class:`StreamingResponse` iterator after the request is complete.

        Returns:
          An asynchronous :class:`StreamingResponse` iterator (when ``streaming=True``) that outputs one decoded token string at a time.
          Otherwise, this function blocks and a string containing the full reply is returned after it's been completed.
        """
        
        ''' this should not actually be necessary because only plugin(text) is called (i.e. the same for both callable and Plugin)
        for idx, function in enumerate(functions):
            if not isinstance(function, Plugin):
                if not callable(function):
                    raise TypeError(f"functions passed to NanoLLM.generate() should be either a callable function, or a Plugin instance (was {type(function)})")
            functions[idx] = Callback(function)
        '''

        stream = StreamingResponse(self, inputs, detokenize=detokenize, **kwargs)
        
        self.queue.put(stream)
        
        if not streaming:
            stream.wait()
            if detokenize:
                return stream.text
            else:
                return stream.tokens

        return stream

    def _generate(self, stream):
        """
        Process a generation request in model's inference thread.
        """
        functions = stream.kwargs.get('functions', [])
        
        if functions is None:
            functions = []
        elif not isinstance(functions, list):
            functions = [functions]

        max_new_tokens = stream.kwargs.get('max_new_tokens', 128)
        min_new_tokens = stream.kwargs.get('min_new_tokens', -1)
        
        do_sample = stream.kwargs.get('do_sample', False)
        temperature = stream.kwargs.get('temperature', 0.7)
        top_p = stream.kwargs.get('top_p', 0.95)
        repetition_penalty = stream.kwargs.get('repetition_penalty', 1.0)

        self.device.set_raw_stream(self.cuda_stream)
        
        # if the stop tokens are strings, tokenize them
        stop_tokens = stream.kwargs.get('stop_tokens', [self.tokenizer.eos_token_id])
        
        if isinstance(stop_tokens, int):
            stop_tokens = [stop_tokens]

        for i, stop in enumerate(stop_tokens):
            if isinstance(stop, str):
                stop_tokens[i] = self.tokenize(stop).squeeze().tolist()
                    
        # convert inputs to tokens or embeddings
        if isinstance(stream.input, str):
            if self.has_embed:
                input = self.embed_text(stream.input, return_tensors='tvm')
            else:
                input = self.tokenize(stream.input)
        else:
            input = stream.input
        
        if not isinstance(input, tvm.runtime.ndarray.NDArray):
            input = tvm.nd.array(input, device=self.device)

        self.stats.input_tokens = input.shape[1]
        self.stats.output_tokens = 0
        
        if input.dtype.startswith('int'):
            if self.has_embed:
                input = self.embed_tokens(input, return_tensors='tvm')
        elif input.dtype.startswith('float'):
            if not self.has_embed:
                raise RuntimeError(f"{self.config.name} doesn't have embedding support in {self.module_path} and was passed {input.dtype} input, but can only process int token inputs")
        else:
            raise TypeError(f"expected input of type int or float, but got {input.dtype}")
        
        # create a kv_cache if needed
        if stream.kv_cache is None:
            stream.kv_cache = MLCKVCache(self)

        stream.kv_cache.num_tokens += input.shape[1]

        # returns a list[logits, [kv_cache]] - logits are [1,1,32000] float32 (on the GPU)
        time_begin_prefill = time.perf_counter()
        
        def prefill(input, kv_cache):
            if self.kv_cache_paged:
                self._kv_cache_begin_forward(
                    kv_cache.state, 
                    tvm.runtime.ShapeTuple([0]),  # sequence ID
                    tvm.runtime.ShapeTuple([input.shape[1]]),  # input length
                )
                output = self._prefill(input, kv_cache.state, self.params)
                self._kv_cache_end_forward(kv_cache.state)
            else:  
                output = self._prefill(input,  # prefill_with_embed
                    tvm.runtime.ShapeTuple([kv_cache.num_tokens]), 
                    kv_cache.state, self.params
                )
            return output
            
        output = prefill(input, stream.kv_cache)

        # decode until EOS or max_new_tokens
        time_begin_decode = time.perf_counter()

        self.stats.prefill_time = time_begin_decode - time_begin_prefill
        self.stats.prefill_rate = self.stats.input_tokens / self.stats.prefill_time
        
        while True:
            tokens = [self._sample(output[0], do_sample, temperature, top_p, repetition_penalty)]
            stream.add_tokens(tokens)
            
            # TODO change `token` above to a list and then add any tokens made by functions to it
            # (and adapt the code below to handle > 1 token at a time)
            function_text = ''
            
            for function in functions:
                function_next = function(stream.text)
                if not function_next:
                    continue
                function_text = ' ' + function_next
                
            if function_text:
                function_tokens = self.tokenize(function_text)[0].tolist()
                tokens.extend(function_tokens)
                stream.add_tokens(function_tokens)
                stream.kv_cache.num_tokens += len(tokens)
                #self.stats.output_tokens += len(tokens)
                output = prefill(self.embed_tokens(tokens, return_tensors='tvm'), stream.kv_cache)
                continue
                
            stream.kv_cache.num_tokens += 1
            self.stats.output_tokens += 1

            stream.event.set()

            # decode the next token
            if self.kv_cache_paged:
                self._kv_cache_begin_forward(
                    stream.kv_cache.state, 
                    tvm.runtime.ShapeTuple([0]),  # sequence ID (always 0)
                    tvm.runtime.ShapeTuple([1]),  # number of tokens added
                )
                embedding = self._embed(tvm.nd.array(np.array([[tokens[-1]]], dtype=np.int32), self.device), self.params)
                
                output = self._decode(
                    embedding,
                    stream.kv_cache.state, self.params
                )
                self._kv_cache_end_forward(stream.kv_cache.state)
            else:
                output = self._decode(
                    tvm.nd.array(np.array([[tokens[-1]]], dtype=np.int32), self.device),
                    tvm.runtime.ShapeTuple([stream.kv_cache.num_tokens]), stream.kv_cache.state, self.params
                )

            self.stats.decode_time = time.perf_counter() - time_begin_decode
            self.stats.decode_rate = self.stats.output_tokens / self.stats.decode_time
        
            # stop generation on EOS tokens
            if len(stream.tokens) >= min_new_tokens and ends_with_token(stream.tokens, stop_tokens, self.tokenizer):
                break

            # add EOS token on early stop
            if len(stream.tokens) >= max_new_tokens - 1 or stream.stopping or stream.stopped:
                stream.add_tokens(self.tokenizer.eos_token_id)
                stream.kv_cache.num_tokens += 1
                self.stats.output_tokens += 1
                prefill(self.embed_tokens([self.tokenizer.eos_token_id], return_tensors='tvm'), stream.kv_cache)
                break

        stream.stopped = True
        stream.event.set()
        
    def _sample(self, logits, do_sample, temperature, top_p, repetition_penalty):
        """
        Perform token sampling (TODO implement repetition penalty)
        https://github.com/mlc-ai/mlc-llm/blob/6e40c21fb6433aeffe50ee321f1b589ef846b6fb/cpp/llm_chat.cc#L1044
        """
        if do_sample:
            return self._sample_top_p_from_logits(logits, top_p, temperature, random.random())
        else:
            return np.argmax(logits.numpy()) #, axis=-1)

    def _run(self):
        """
        Run the generation requests thread.
        """
        while True:
            stream = self.queue.get()
            self._generate(stream)
        
    def _create_kv_cache(self, use_cache=True):
        """
        Allocate or return a free KV cache available to use. If use_cache is true, then an existing
        KV cache that isn't in use is returned, because KV caches take ~100ms to allocate.
        Otherwise, a new cache is allocated and returned (which can take considerable time)
        """
        if use_cache:
            for kv_cache in self.kv_caches:
                if sys.getrefcount(kv_cache) <= 3:  # existing references from this call, loop, and list
                    self._kv_cache_clear(kv_cache)  # clearing is much faster than allocating (<0.1ms)
                    return kv_cache

        time_begin = time.perf_counter()
        
        if self.kv_cache_paged:
            kv_cache = self._kv_cache_create(
                tvm.runtime.ShapeTuple([1]),  # max sequences
                tvm.runtime.ShapeTuple([self.config.max_length]),  # max window size
                tvm.runtime.ShapeTuple([self.config.prefill_chunk_size]),  # prefill chunk size
                tvm.runtime.ShapeTuple([16])  # page size
            )
            self._kv_cache_add_sequence(kv_cache, 0)
        else:
            kv_cache = self._kv_cache_create()
        
        logging.debug(f"allocated new KV cache in {(time.perf_counter()-time_begin)*1000:.1f} ms  (existing cache refcounts={[sys.getrefcount(k) for k in self.kv_caches]})")
        if use_cache:
            self.kv_caches.append(kv_cache)
        return kv_cache    


class MLCKVCache(KVCache):
    """
    Interface for storing & manipulating the chat's KV cache.
    """
    def __init__(self, model, state=None):
        super().__init__()
        
        self.model = model
        self.state = state
        
        if not self.state:
            self.state = self.model._create_kv_cache()

        self.tvm = None    # list of tvm.ndarray
        self.torch = None  # list of torch.cuda.Tensor
        self.cuda = None   # list of cuda pointers

    def pop(self, tokens):
        """
        Remove the given number of tokens from the end of the cache.
        """
        if self.num_tokens - tokens < 0:
            raise ValueError(f"tried to pop {tokens} from the KV cache, but the cache only had {self.num_tokens} in it")
            
        logging.debug(f"popping {tokens} tokens from KV cache (num_tokens={self.num_tokens})")
        
        self.model._kv_cache_pop(self.state, tokens)
        self.num_tokens = self.num_tokens - tokens
    
    def remove(self, start, stop, sync=True):
        """
        Remove a range of tokens from the cache, from the start index (inclusive) to the stop index (exclusive)
        """
        self.map(cuda=True)
        
        remove_tokens = stop - start
        
        logging.debug(f"removing {remove_tokens} tokens from KV cache at [{start}, {stop})  (num_tokens={self.num_tokens})")
        
        for n in range(len(self.torch)):
            ptr = self.cuda[n]
            tvm = self.tvm[n]
            cache_size = tvm.shape[1] * tvm.shape[2] * np.dtype(tvm.dtype).itemsize
            dst_tokens = start
            src_tokens = stop
            
            #if n == 0:
            #    print(f" layer {n} - shape={tvm.shape}  size={cache_size}")
            
            while src_tokens < self.num_tokens:
                num_tokens = min(remove_tokens, self.num_tokens - src_tokens)

                #if n == 0:
                #    print(f"   dst_tokens={dst_tokens}  src_tokens={src_tokens}  copy_tokens={num_tokens}")
                
                cudaMemcpy(
                    ptr + dst_tokens * cache_size,
                    ptr + src_tokens * cache_size,
                    num_tokens * cache_size,
                    cudaMemcpyKind.cudaMemcpyDeviceToDevice
                )
                
                src_tokens = src_tokens + num_tokens
                dst_tokens = dst_tokens + num_tokens
                
        self.pop(remove_tokens)
        
        if sync:
            cudaDeviceSynchronize()
               
    def get_size(self, length=None):
        """
        Calculate the size (in bytes) that the KV cache consumes for the given token length
        (or by default for the maximum window size if length is None)
        """
        self.map(tvm=True)
        size = 0
        for x in self.tvm:
            if length is None:
                length = x.shape[0]
            size += length * x.shape[1] * x.shape[2] * np.dtype(x.dtype).itemsize
        return size
        
    def map(self, tvm=False, pytorch=False, cuda=False):
        """
        Allocate the mappings needed to access the KV cache for performing various manipulations.
        """
        if cuda:
            pytorch = True
            
        if pytorch:
            tvm = True
            
        if tvm and not self.tvm:
            self.tvm = [self.model._kv_cache_view(x) for x in self.state]
            
        if pytorch and not self.torch:
            self.torch = [torch.utils.dlpack.from_dlpack(x.to_dlpack()) for x in self.tvm]
            
        if cuda and not self.cuda:
            self.cuda = [x.data_ptr() for x in self.torch]
            
