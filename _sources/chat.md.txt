# Chat

This page includes information about managing multi-turn chat sessions, templating, and maintaining the embedding history.  Here's how to run it interactively from the terminal:

```bash
python3 -m nano_llm.chat --api mlc \
  --model meta-llama/Llama-2-7b-chat-hf \
  --quantization q4f16_ft
```

If you load a multimodal model (like `liuhaotian/llava-v1.6-vicuna-7b`), you can enter image filenames or URLs followed by a query to chat about images.  Enter `/reset` to reset the chat history.

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
