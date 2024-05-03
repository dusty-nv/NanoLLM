# Models

The [`NanoLLM`](#model-api) interface provides model loading, quantization, embeddings, and inference.

```python
from nano_llm import NanoLLM

model = NanoLLM.from_pretrained(
   "meta-llama/Llama-3-8b-hf",  # HuggingFace repo/model name, or path to HF model checkpoint
   api='mlc',                   # supported APIs are: mlc, awq, hf
   api_token='hf_abc123def',    # HuggingFace API key for authenticated models ($HUGGINGFACE_TOKEN)
   quantization='q4f16_ft'      # q4f16_ft, q4f16_1, q8f16_0 for MLC, or path to AWQ weights
)

response = model.generate("Once upon a time,", max_new_tokens=128)

for token in response:
   print(token, end='', flush=True)
```

You can run text completion from the command-line like this:

```bash
python3 -m nano_llm.completion --api=mlc \
  --model meta-llama/Llama-3-8b-chat-hf \
  --quantization q4f16_ft \
  --prompt 'Once upon a time,'
```

See the [Chat](chat.md) section for examples of running multi-turn chat and [function calling](chat.md#function-calling).

## Supported Architectures

* Llama
* Llava
* StableLM
* Phi-2
* Gemma
* Mistral
* GPT-Neox

These include fine-tuned derivatives that share the same network architecture as above (for example, [`lmsys/vicuna-7b-v1.5`](https://huggingface.co/lmsys/vicuna-7b-v1.5) is a Llama model).  Others model types are supported via the various quantization APIs well - check the associated library documentation for details.

## Tested Models

```{eval-rst}
.. admonition:: Access to Gated Models from HuggingFace Hub

   To download models requiring authentication, generate an `API key <https://huggingface.co/docs/api-inference/en/quicktour#get-your-api-token>`_ and `request access <https://huggingface.co/meta-llama>`_ (Llama)
```

**Large Language Models**

* [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
* [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
* [`meta-llama/Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
* [`meta-llama/Llama-2-70b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)

**Small Language Models** ([SLM](https://www.jetson-ai-lab.com/tutorial_slm.html))

* [`stabilityai/stablelm-2-zephyr-1_6b`](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)
* [`stabilityai/stablelm-zephyr-3b`](https://huggingface.co/stabilityai/stablelm-zephyr-3b)
* [`NousResearch/Nous-Capybara-3B-V1.9`](https://huggingface.co/NousResearch/Nous-Capybara-3B-V1.9)
* [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
* [`princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT`](https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT)
* [`google/gemma-2b-it`](https://huggingface.co/google/gemma-2b-it)
* [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2)

**Vision Language Models** ([VLM](https://www.jetson-ai-lab.com/tutorial_llava.html))

* [`liuhaotian/llava-v1.5-7b`](https://huggingface.co/liuhaotian/llava-v1.5-7b)
* [`liuhaotian/llava-v1.5-13b`](https://huggingface.co/liuhaotian/llava-v1.5-13b)
* [`liuhaotian/llava-v1.6-vicuna-7b`](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b)
* [`liuhaotian/llava-v1.6-vicuna-13b`](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b)
* [`NousResearch/Obsidian-3B-V0.5`](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
* [`Efficient-Large-Model/VILA-2.7b`](https://huggingface.co/Efficient-Large-Model/VILA-2.7b)
* [`Efficient-Large-Model/VILA-7b`](https://huggingface.co/Efficient-Large-Model/VILA-7b)
* [`Efficient-Large-Model/VILA-13b`](https://huggingface.co/Efficient-Large-Model/VILA-13b)
* [`Efficient-Large-Model/VILA1.5-3b`](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b)
* [`Efficient-Large-Model/Llama-3-VILA1.5-8B`](https://huggingface.co/Efficient-Large-Model/Llama-3-VILA1.5-8b)
* [`Efficient-Large-Model/VILA1.5-13b`](https://huggingface.co/Efficient-Large-Model/VILA1.5-13b)

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

