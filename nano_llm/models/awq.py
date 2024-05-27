#!/usr/bin/env python3
import os
import time
import torch
import numpy as np

# https://github.com/mit-han-lab/llm-awq/blob/main/tinychat/utils/constants.py
#tinychat.utils.constants.max_batch_size = args.max_batch_size
#tinychat.utils.constants.max_seq_len = args.max_seq_len

from awq.quantize.quantizer import real_quantize_model_weight
from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp
from tinychat.models import FalconForCausalLM, LlamaForCausalLM, MPTForCausalLM
from tinychat.utils.load_quant import load_awq_model, load_awq_llama_fast

from transformers import AutoConfig, modeling_utils
from accelerate import load_checkpoint_and_dispatch

from nano_llm import NanoLLM, StreamingResponse
#from .hf import HFModel


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
            
        #self.model = self.model_types['llama'](config).half()
        
        #real_quantize_model_weight(self.model, w_bit=w_bit, q_config=self.q_config, init_only=True)

        #self.model = load_checkpoint_and_dispatch(
        #    self.model, self.quant_path, device_map='balanced', 
        #    no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer"]
        #)
        
        
        self.model = self.model.to(self.device)
        

            
        make_quant_attn(self.model, self.device)
        make_quant_norm(self.model)
        make_fused_mlp(self.model)
        
        self.model.eval()
        self.has_embed = False
        
        print("Benchmarking...")
        context_length = 4
        gen_length = 200
        start_pos = 0
        time_lis = []
        input_ids = [1 for _ in range(context_length)]

        with torch.inference_mode():
            for i in range(gen_length):
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
                print(i, np.median(time_lis))

        print(f"Speed: {1 / np.median(time_lis)} tokens per second.")
        
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



        
