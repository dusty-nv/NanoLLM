# Plugins

Plugins are wrappers around models (like LLM, ASR, TTS), post-processors, and I/O (such as video and audio streams) that can be connected together to form pipelines in tandem with other plugin nodes.  These are designed to reduce boilerplate code and quickly build up more complex agents.

A plugin recieves input into a processing queue, processes it, and then outputs the results across its output channels.  Each output channel signifies a type of output the model has (for example, the ``ChatQuery`` plugin exposes outputs at the token, word, and sentence level) and each output channel can be connected to an arbitrary number of other plugin nodes. 

By default plugins are threaded and run off their own queue, but they can be configured to run inline (unthreaded) by passing ``threaded=False`` to the plugin's initializer.  They can also be interrupted with the `interrupt()` function to abandon the current request and any remaining data in the input queue (for example, if you wanted to stop LLM generation early, or mute TTS output)

When creating new plugin types, implement the `process()` function to handle incoming data, and then return the outgoing data.  You can also use simple callback functions to recieve data instead of needing to define your own Plugin class (like `chat_plugin.add(my_function)` to recieve chat output)

## Plugin API

All plugins derive from the shared ``Plugin`` interface:

```{eval-rst}
.. autoclass:: nano_llm.Plugin
   :members:
   :special-members: __call__
```
