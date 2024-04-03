# Chat

This page includes information about managing multi-turn chat sessions, templating, and maintaining the embedding history.  Here's how to run it interactively from the terminal:

```bash
python3 -m nano_llm.chat --api mlc \
  --model meta-llama/Llama-2-7b-chat-hf \
  --quantization q4f16_ft
```

If you load a multimodal model (like `liuhaotian/llava-v1.6-vicuna-7b`), you can enter image filenames or URLs followed by a query to chat about images.  Enter `/reset` to reset the chat history.

## Code Example

```python
from nano_llm import NanoLLM, ChatHistory

# load model
model = NanoLLM.from_pretrained(
   "meta-llama/Llama-2-7b-hf",  # HuggingFace repo/model name, or path to HF model checkpoint
   api='mlc',                   # supported APIs are: mlc, awq, hf
   api_token='hf_abc123def',    # HuggingFace API key for authenticated models ($HUGGINGFACE_TOKEN)
   quantization='q4f16_ft'      # q4f16_ft, q4f16_1, q8f16_0 for MLC, or path to AWQ weights
)

# create the chat history
chat_history = ChatHistory(model, system_prompt="You are a helpful and friendly AI assistant.")

while True:
    # enter the user query from terminal
    print('>> ', end='', flush=True)
    prompt = input().strip()

    # add user prompt and generate chat tokens/embedding
    chat_history.append(role='user', msg=prompt)
    embedding, position = chat_history.embed_chat()

    # generate bot reply
    reply = model.generate(
        embedding, 
        streaming=True, 
        kv_cache=chat_history.kv_cache,
        stop_tokens=chat_history.template.stop,
        max_new_tokens=256,
    )
        
    # append the output stream to the chat history
    bot_reply = chat_history.append(role='bot', text='')
    
    for token in reply:
        bot_reply.text += token
        print(token, end='', flush=True)
            
    print('\n')

    # save the inter-request KV cache 
    chat_history.kv_cache = reply.kv_cache
```

## Templates

These are the built-in chat templates that are automatically determined from the model type, or settable with the ``--chat-template`` command-line argument:

```
* llama-2
* vicuna-v0, vicuna-v1
* stablelm-zephyr
* chat-ml
* sheared-llama
* nous-obsidian
* phi-2-chat, phi-2-instruct
* gemma
```

See `nano_llm/chat/templates.py` for them.  You can also specify a JSON file containing the template.

## Chat History

```{eval-rst}
.. autoclass:: nano_llm.ChatHistory
   :members:
```
