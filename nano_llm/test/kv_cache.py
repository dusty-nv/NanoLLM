#!/usr/bin/env python3
import argparse
import termcolor
import pprint
import time
import tvm
import torch

from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import LogFormatter, print_table

DEFAULT_MODEL="princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT"  #"meta-llama/Llama-2-7b-chat-hf"   

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help="path to the model, or HuggingFace model repo")
parser.add_argument('--max-new-tokens', type=int, default=128, help="the maximum response length for each bot reply")
parser.add_argument('--modify-cache', type=str, default='remove', choices=['none', 'remove', 'pop'])

args = parser.parse_args()
LogFormatter.config(level='debug')

# load model
model = NanoLLM.from_pretrained(
    model=args.model, 
    quantization='q4f16_ft', 
    api='mlc'
)

# create the chat history
chat_history = ChatHistory(model, system_prompt="You are a helpful and friendly AI assistant.")

#system_embedding, _ = chat_history.embed_chat()

prompts = [
    "Hi, my name is Dusty!",
    "What is 2+2?",
    "What's my name?",
    "What color is the sky?",
    "How many feet are in a mile?",
    "How cold is it at the North Pole?",
]

"""
def reset(chat_history, kv_cache):
    time_begin = time.perf_counter()
    chat_history.entries = chat_history.entries[1:]
    pop_tokens = model._kv_cache_view(kv_cache.state[0]).shape[0] - system_embedding.shape[1]
    kv_cache.num_tokens = kv_cache.num_tokens - pop_tokens
    model._kv_cache_pop(kv_cache.state, pop_tokens)

    for kv in kv_cache.state:
        view = model._kv_cache_view(kv)#.numpy()
        '''
        model._kv_cache_update(kv,
            tvm.nd.array(
                view[:system_embedding.shape[1]+1,:,:], 
                device=model.device
        ))
        '''
        tensor = torch.utils.dlpack.from_dlpack(kv_cache_view.to_dlpack())
        kv_size = tensor.shape[1] * tensor.shape[2] * 2
        begin_ptr = tensor.data_ptr() + (system_embedding.shape[1] * kv_size)
        cudaMemcpy(begin_ptr, begin_ptr + 185 * kv_size, 185 * kv_size, cudaMemcpyKind.cudaMemcpyDeviceToDevice)

    cudaDeviceSynchronize()               
    print(f'KV_CACHE RESET  {(time.perf_counter() - time_begin)*1000} ms')
"""


    
for p, prompt in enumerate(prompts):
    print(prompt)
    
    # add user prompt and generate chat tokens/embeddings
    chat_history.append(role='user', msg=prompt)
    embedding, position = chat_history.embed_chat()

    print(f'chat embedding={embedding.shape}  position={position}')
    
    # generate bot reply
    reply = model.generate(
        embedding, 
        streaming=True, 
        kv_cache=chat_history.kv_cache,
        stop_tokens=chat_history.template.stop,
        max_new_tokens=args.max_new_tokens,
    )
        
    # stream the output
    for token in reply:
        termcolor.cprint(token, 'blue', end='\n\n' if reply.eos else '', flush=True)

    chat_history.kv_cache = reply.kv_cache
    
    # save the final output
    bot_entry = chat_history.append(role='bot', text=reply.text, tokens=reply.tokens)

    #chat_history.kv_cache.map(tvm=True)
    print(f"cache_num_tokens={chat_history.kv_cache.num_tokens}  chat_tokens={chat_history.num_tokens}  reply_tokens={len(reply.tokens)}")  # kv_cache={chat_history.kv_cache.tvm[0].shape}  
    
    #print(f'RESPONSE TOKENS      ', reply.tokens)
    #print(f'DETOKENIZED RESPONSE ```{model.detokenize(reply.tokens)}```')
    #print(f'RE-TOKENIZED RESPONSE ', model.tokenize(model.detokenize(reply.tokens)))
    #print(f'SPECIAL TOKEN        ```{model.detokenize([29871])}```')
    
    print_table(model.stats)
    
    if args.modify_cache == 'remove' and p > 0:
        #remove_tokens = chat_history[1].text_embedding.shape[1] + chat_history[2].text_embedding.shape[1]
        #system_embedding = chat_history.entries[0].text_embedding
        
        #time_begin = time.perf_counter()
        #chat_history.kv_cache.remove(system_embedding.shape[1], system_embedding.shape[1] + remove_tokens)
        #print(f'KV_CACHE REMOVE  {(time.perf_counter() - time_begin)*1000} ms   num_tokens={chat_history.kv_cache.num_tokens}')
        print(f'CHAT ENTRIES BEFORE REMOVAL {len(chat_history)}')
        
        del chat_history[1:3]

        print(f'CHAT ENTRIES AFTER REMOVAL {len(chat_history)}')
           
    elif args.modify_cache == 'pop':
        #pop_tokens = chat_history.entries[-1].text_embedding.shape[1] + chat_history.entries[-2].text_embedding.shape[1] 
        #tokens_before = chat_history.kv_cache.num_tokens
        #time_begin = time.perf_counter()
        #chat_history.kv_cache.pop(pop_tokens)
        #print(f'KV_CACHE POP {pop_tokens}  {(time.perf_counter() - time_begin)*1000} ms   tokens_before={tokens_before}  tokens_after={chat_history.kv_cache.num_tokens}')
        print(f'CHAT ENTRIES BEFORE REMOVAL {len(chat_history)}')
        
        del chat_history[-2:]
        #del chat_history.entries[-1]
        print(f'CHAT ENTRIES AFTER REMOVAL {len(chat_history)}')
        
    print(f"cache_num_tokens={chat_history.kv_cache.num_tokens}  chat_tokens={chat_history.num_tokens}  reply_tokens={len(reply.tokens)}")  # kv_cache={chat_history.kv_cache.tvm[0].shape}  
       
