#!/usr/bin/env python3
import os
import time
import queue
import threading
import logging

import torch
import numpy as np

# https://github.com/mit-han-lab/llm-awq/blob/main/tinychat/utils/constants.py
#tinychat.utils.constants.max_batch_size = args.max_batch_size
#tinychat.utils.constants.max_seq_len = args.max_seq_len

from awq.quantize.quantizer import real_quantize_model_weight
from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp
from tinychat.models import FalconForCausalLM, LlamaForCausalLM, MPTForCausalLM
from tinychat.utils.load_quant import load_awq_model, load_awq_llama_fast

from accelerate import load_checkpoint_and_dispatch
from transformers import AutoConfig, modeling_utils

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from nano_llm import NanoLLM, StreamingResponse
from nano_llm.utils import convert_tensor, ends_with_token


class AWQModel(NanoLLM):
    """
    AWQ model (https://github.com/mit-han-lab/llm-awq)
    """
    def __init__(self, model_path, quantization=None, w_bit=4, q_group_size=128, zero_point=True, **kwargs):
        super(AWQModel, self).__init__(model_path, **kwargs)

        if not quantization:
            raise ValueError(f"AWQ model needs to have the --quantization argument provided, with the path to the quantized model")
            
        if not os.path.isfile(quantization):
            raise ValueError(f"AWQ quantized model not found: {quantization}")
        
        self.quant_path = quantization
        self.config.quant = os.path.basename(quantization)
        self.config.precision = w_bit
        
        self.q_config = {
            'zero_point': zero_point,
            'q_group_size': q_group_size,
        }
        
        self.model_types = {
            "llama": LlamaForCausalLM,
            "falcon": FalconForCausalLM,
            "mpt": MPTForCausalLM,
        }

        def skip(*args, **kwargs):
            pass
    
        modeling_utils._init_weights = False
        
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.kaiming_normal_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
    
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)   
             
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cuda_stream = None #torch.cuda.Stream(self.device)
        
        if True: # args.model_type.lower() == "llama":
            self.model = self.model_types["llama"](config).half()
            self.model = load_awq_llama_fast(
                self.model, self.quant_path, w_bit, q_group_size, self.device
            )
        else:
            self.model = self.model_types[args.model_type.lower()](config).half()
            self.model = load_awq_model(
                self.model, self.quant_path, w_bit, q_group_size, self.device
            )
            
        self.model = self.model.to(self.device)

        make_quant_attn(self.model, self.device)
        make_quant_norm(self.model)
        #make_fused_mlp(self.model)
        
        self.model.eval()
        
        self.has_embed = True
        self.config.type = self.model.config.model_type
        self.config.max_length = self.model.config.max_length
        self.config.vocab_size = self.model.config.vocab_size
        
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True).start()  

        self.benchmark()
     
    def embed_tokens(self, tokens, return_tensors='np', **kwargs):
        """
        Generate embedding from token IDs
        """
        if not self.has_embed:
            raise RuntimeError(f"{self.config.name} does not have embed() in {self.module_path}")
        
        with torch.cuda.StreamContext(self.cuda_stream), torch.inference_mode():
            tokens = convert_tensor(tokens, return_tensors='pt', device=self.device)
            
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)

            return convert_tensor(self.model.model.embed_tokens(tokens), return_tensors=return_tensors)
       
    def generate(self, inputs, streaming=True, functions=None, **kwargs):
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
        if functions is None:
            functions = []
        elif not isinstance(functions, list):
            functions = [functions]

        stream = StreamingResponse(self, inputs, functions=functions, **kwargs)
        self.queue.put(stream)
        
        if not streaming:
            text = ''
            for token in stream:
                text += token
            return stream.text  # return the fully-detokenized copy
        
        return stream
    
    def _generate(self, stream):
        """
        Process a generation request in model's inference thread.
        """
        max_new_tokens = stream.kwargs.get('max_new_tokens', 128)
        min_new_tokens = stream.kwargs.get('min_new_tokens', -1)

        logits_processor = self.prepare_logits_processor(
            do_sample=stream.kwargs.get('do_sample', False), 
            temperature=stream.kwargs.get('temperature', 0.7),
            repetition_penalty=stream.kwargs.get('repetition_penalty', 1.0),
            top_p=stream.kwargs.get('top_p', 0.95),
            top_k=stream.kwargs.get('top_k', 40),
        )
        
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
                inputs = self.embed_text(stream.input, return_tensors='pt')
            else:
                inputs = self.tokenize(stream.input, return_tensors='pt', dtype=torch.int64)
        else:
            inputs = stream.input
        
        inputs = convert_tensor(inputs, return_tensors='pt', device=self.device)

        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)
            
        # there are tinychat errors if input_len<8 (except 4)
        if inputs.shape[1] < 8 and inputs.shape[1] != 4:
            if torch.is_floating_point(inputs):
                assert(self.has_embed)
            else:
                logging.warning(f"AWQ padding {inputs.shape[1]} token input up to 8 tokens")
                inputs = torch.cat([
                    inputs,
                    torch.full((1, 8 - inputs.shape[1]), self.tokenize(' ', return_tensors='np').item(), dtype=inputs.dtype, device=self.device),
                ], dim=1)

        self.stats.input_tokens = inputs.shape[1]
        self.stats.output_tokens = 0
        
        time_begin_prefill = time.perf_counter()
        cache_position = stream.kwargs.get('cache_position', 0)
        
        with torch.cuda.StreamContext(self.cuda_stream), torch.inference_mode():
            for i in range(max_new_tokens):
                if i > 0:
                    if self.has_embed:
                        inputs = self.embed_tokens([[token]], return_tensors='pt')
                    else:
                        inputs = torch.as_tensor([[token]], device=self.device)

                logits = self.model(
                    None if self.has_embed else inputs,
                    start_pos=cache_position,
                    inputs_embeds=inputs if self.has_embed else None,
                )

                if logits_processor:
                    if logits_processor.needs_past_tokens:
                        token_ids = torch.as_tensor([stream.tokens], device=self.device) # TODO this should include the input_id's, but we might not have them if has_embed
                    else:
                        token_ids = None
                        
                    logit = logits_processor(token_ids, logits[:, -1, :])[0]
                    probs = torch.softmax(logit, dim=-1)
                    token = int(torch.multinomial(probs, num_samples=1))
                else:
                    token = int(torch.argmax(logits[0, -1, :]))
                
                if i == 0:
                    time_begin_decode = time.perf_counter()

                cache_position += logits.shape[1]
                self.stats.output_tokens += 1

                stream.add_tokens(token)
                stream.event.set()
                
                # stop generation on EOS tokens
                if len(stream.tokens) >= min_new_tokens and ends_with_token(stream.tokens, stop_tokens, self.tokenizer):
                    break
                    
                # add EOS token on early stop
                if len(stream.tokens) >= max_new_tokens - 1 or stream.stopping or stream.stopped:
                    stream.add_tokens(self.tokenizer.eos_token_id)
                    self.stats.output_tokens += 1
                    break
                
        time_end_decode = time.perf_counter()
        
        self.stats.prefill_time = time_begin_decode - time_begin_prefill
        self.stats.prefill_rate = self.stats.input_tokens / self.stats.prefill_time
        self.stats.decode_time = time_end_decode - time_begin_decode
        self.stats.decode_rate = self.stats.output_tokens / self.stats.decode_time

        logging.debug(f"AWQ ending cache position:  {cache_position}")
        
        stream.stopped = True
        stream.event.set()  
        
    def _run(self):
        """
        Run the generation requests thread.
        """     
        while True:
            stream = self.queue.get()
            self._generate(stream)
    
    def prepare_logits_processor(self, do_sample=False, temperature=1.0, repetition_penalty=1.0, top_p=1.0, top_k=0):
        """
        Create the logits post-processors based on the sampling options.
        """
        if not do_sample:
            return None
            
        processor_list = LogitsProcessorList()
        processor_list.needs_past_tokens = False
        
        # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
            processor_list.needs_past_tokens = True
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
            
        return processor_list
                
    def benchmark(self, context_length=8, generation_length=128):
        """
        Run performance benchmark
        """
        logging.info(f"Benchmarking AWQ model {self.config.name} (input={context_length} output={generation_length})")
        start_pos = 0
        time_lis = []
        input_ids = [1 for _ in range(context_length)]

        with torch.inference_mode():
            for i in range(generation_length):
                torch.cuda.synchronize()
                t_st = time.time()

                if i == 0:
                    inputs = torch.as_tensor([input_ids], device=self.device)
                else:
                    inputs = torch.as_tensor([[token]], device=self.device)

                out = self.model(inputs, start_pos=start_pos)
                start_pos += out.shape[1]

                torch.cuda.synchronize()
                t_ed = time.time()
                time_lis.append(t_ed - t_st)
                token = out[:, -1].max(1)[1].unsqueeze(1)
                #print(i, np.median(time_lis))

        logging.info(f"AWQ {self.config.name} generation: {1 / np.median(time_lis):.2f} tokens/second")
        
'''
class AWQModel(HFModel):
    """
    AWQ model (https://github.com/mit-han-lab/llm-awq)
    """
    def __init__(self, model_path, quantization=None, w_bit=4, q_group_size=128, zero_point=True, **kwargs):
        super(AWQModel, self).__init__(model_path, init_empty_weights=True, **kwargs)

        if not quantization:
            raise ValueError(f"AWQ model needs to have the --quantization argument provided, with the path to the quantized model")
            
        if not os.path.isfile(quantization):
            raise ValueError(f"AWQ quantized model not found: {quantization}")
        
        self.quant_path = quantization
        self.config.quant = os.path.basename(quantization)
        self.config.precision = w_bit
        
        self.q_config = {
            'zero_point': zero_point,
            'q_group_size': q_group_size,
        }

        real_quantize_model_weight(self.model, w_bit=w_bit, q_config=self.q_config, init_only=True)

        self.model = load_checkpoint_and_dispatch(
            self.model, self.quant_path, device_map='balanced', 
            no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer"]
        )
        
        #make_quant_attn(self.model, self.device)
        #make_quant_norm(self.model)
        #make_fused_mlp(self.model)
        
        self.model.eval()
        self.has_embed = False
'''



        
