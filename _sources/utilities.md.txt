# Utilities

Documented on this page are various utility functions that NanoLLM provides, including audio/image manipulation, tensor format conversion, argument parsing, ect.

To use these, import them from **nano_llm.utils** (like `from nano_llm.utils import convert_tensor`)

## Tensor Conversion

```{eval-rst}
.. autofunction:: nano_llm.utils.convert_dtype
.. autofunction:: nano_llm.utils.convert_tensor
```

## Audio

```{eval-rst}
.. autofunction:: nano_llm.utils.convert_audio
.. autofunction:: nano_llm.utils.audio_rms
.. autofunction:: nano_llm.utils.audio_silent
```

## Images

```{eval-rst}
.. autofunction:: nano_llm.utils.load_image
.. autofunction:: nano_llm.utils.is_image
.. autofunction:: nano_llm.utils.cuda_image
.. autofunction:: nano_llm.utils.torch_image
.. autofunction:: nano_llm.utils.image_size
```

## Argument Parsing

```{eval-rst}
.. autoclass:: nano_llm.utils.ArgParser
   :members:
   :special-members: __init__
```