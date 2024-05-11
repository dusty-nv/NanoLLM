# Chat

This page includes information about managing multi-turn chat sessions, templating, and maintaining the embedding history.  Here's how to run it interactively from the terminal:

```bash
python3 -m nano_llm.chat --api mlc \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --quantization q4f16_ft
```

If you load a multimodal model (like `liuhaotian/llava-v1.6-vicuna-7b`), you can enter image filenames or URLs and a query to chat about images.  Enter `/reset` to reset the chat history.

## Code Example

```{eval-rst}
.. literalinclude:: ../nano_llm/chat/example.py
```

## Templates

These are the built-in chat templates that are automatically determined from the model type, or settable with the ``--chat-template`` command-line argument:

```
* llama-2, llama-3
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
   :special-members: __len__, __getitem__, __delitem__
```

## Chat Message

```{eval-rst}
.. autoclass:: nano_llm.ChatMessage
   :members:
```

## Function Calling

You can expose Python functions that the model is able to invoke using its code generation abilities, should you so instruct it to.  A list of functions can be provided to [`NanoLLM.generate()`](#model-api) that will be called inline with the generation, and recieve the output produced so far by the model.  

```{eval-rst}
.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/7lKBJPpasAQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  
  <br/>
  <br/>
```

These functions can then parse the text from the bot to determine if it was called, and execute it accordingly.  Any text returned by these functions will be added to the chat before resuming generation, so the bot is able to utilize them the rest of its reply.

The `bot_function()` decorator automatically wraps Python functions, performs regex matching on the model output, runs them if they were called using Python `eval()`, and returns any results:

```python
from nano_llm import NanoLLM, ChatHistory, BotFunctions, bot_function
from datetime import datetime

@bot_function
def DATE():
    """ Returns the current date. """
    return datetime.now().strftime("%A, %B %-m %Y")
   
@bot_function
def TIME():
    """ Returns the current time. """
    return datetime.now().strftime("%-I:%M %p")
          
# load the model   
model = NanoLLM.from_pretrained(
    model="meta-llama/Meta-Llama-3-8B-Instruct", 
    quantization='q4f16_ft', 
    api='mlc'
)

# create the chat history
system_prompt = "You are a helpful and friendly AI assistant." + BotFunctions.generate_docs()
chat_history = ChatHistory(model, system_prompt=system_prompt)

while True:
    # enter the user query from terminal
    print('>> ', end='', flush=True)
    prompt = input().strip()

    # add user prompt and generate chat tokens/embeddings
    chat_history.append(role='user', msg=prompt)
    embedding, position = chat_history.embed_chat()

    # generate bot reply (give it function access)
    reply = model.generate(
        embedding, 
        streaming=True, 
        functions=BotFunctions(),
        kv_cache=chat_history.kv_cache,
        stop_tokens=chat_history.template.stop
    )
        
    # stream the output
    for token in reply:
        print(token, end='\n\n' if reply.eos else '', flush=True)

    # save the final output
    chat_history.append(role='bot', text=reply.text, tokens=reply.tokens)
    chat_history.kv_cache = reply.kv_cache
```
   

```{eval-rst}
.. autofunction:: nano_llm.bot_function
```

```{eval-rst}
.. autoclass:: nano_llm.BotFunctions
   :members:
   :special-members: __new__
```
