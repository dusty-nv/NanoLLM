# Models

The [`NanoLLM`](#model-api) interface provides model loading, quantization, embeddings, and inference.

```python
from nano_llm import NanoLLM

model = NanoLLM.from_pretrained(
   "meta-llama/Llama-2-7b-hf",  # HuggingFace repo/model name, or path to HF model checkpoint
   api='mlc',                   # supported APIs are: mlc, awq, hf
   api_token='abc123xyz456',    # HuggingFace API key for authenticated models ($HUGGINGFACE_TOKEN)
   quantization='q4f16_ft'      # q4f16_ft, q4f16_1, q8f16_0 for MLC, or path to AWQ weights
)

response = model.generate("Once upon a time,", max_new_tokens=128)

for token in response:
   print(token, end='', flush=True)
```

## Supported Architectures

* Llama
* Llava
* StableLM
* Phi-2
* Gemma
* Mistral
* GPT-Neox

These include fine-tuned derivatives that share the same network architecture as above (for example, [`lmsys/vicuna-7b-v1.5`](https://huggingface.co/lmsys/vicuna-7b-v1.5) is a Llama model).  Others model types are supported via the various quantization APIs well - check the associated library documentation for details.

## Model API

```{eval-rst}
.. autoclass:: nano_llm.NanoLLM
   :members:
```

## Streaming

```{eval-rst}
.. autoclass:: nano_llm.StreamingResponse
   :members:
   :special-members: __next__
```
