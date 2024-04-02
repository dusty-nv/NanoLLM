# Plugins

Plugins are wrappers around models (like LLM, ASR, TTS), post-processors, and I/O (such as video and audio streams) that can be connected together to form pipelines in tandem with other plugin nodes.  These are designed to reduce boilerplate code and quickly build up more complex agents.

A plugin recieves input into a processing queue, processes it, and then outputs the results across its output channels.  Each output channel signifies a type of output the model has (for example, the ``ChatQuery`` plugin exposes outputs at the token, word, and sentence level) and each output channel can be connected to an arbitrary number of other plugin nodes. 

By default plugins are threaded and run off their own queue, but they can be configured to run inline (unthreaded) as well by passing ``threaded=False`` to the plugin's initializer.  You can also use simple callback functions that gets wrapped as a plugin instance without needing to define a new Plugin class.

## Plugin API

All plugins derive from the shared ``Plugin`` interface:

```{eval-rst}
.. autoclass:: nano_llm.Plugin
   :members:
```
