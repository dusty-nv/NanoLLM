#!/usr/bin/env python3
import argparse
import termcolor

from nano_llm import NanoLLM, ChatHistory

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help="path to the model, or HuggingFace model repo")
parser.add_argument('--max-new-tokens', type=int, default=256, help="the maximum response length for each bot reply")
args = parser.parse_args()

# load model
model = NanoLLM.from_pretrained(
    model=args.model, 
    quantization='q4f16_ft', 
    api='mlc'
)

# create the chat history
chat_history = ChatHistory(model, system_prompt="You are a helpful and friendly AI assistant.")

while True:
    # enter the user query from terminal
    print('>> ', end='', flush=True)
    prompt = input().strip()

    # add user prompt and generate chat tokens/embeddings
    chat_history.append(role='user', msg=prompt)
    embedding, position = chat_history.embed_chat()

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

    # save the final output
    chat_history.append(role='bot', text=reply.text)
    chat_history.kv_cache = reply.kv_cache
