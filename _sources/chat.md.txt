# Chat

This page includes information about managing multi-turn chat sessions, templating, and maintaining the embedding history.

## Templates

Built-in chat templates that are automatically determined from the model type, or settable with the ``--chat-template`` command-line argument:

* llama2
* vicuna
* ChatML

See `nano_llm/chat/templates.py` for the template definitions.

## Chat History

```{eval-rst}
.. autoclass:: nano_llm.ChatHistory
   :members:
```
