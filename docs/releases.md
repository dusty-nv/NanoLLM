# Release Notes

Each release has a corresponding branch in the NanoLLM [GitHub repository](https://github.com/dusty-nv/NanoLLM) and container images on [DockerHub](https://hub.docker.com/r/dustynv/nano_llm/tags).  For more info about running these, see the [Installation Guide](install.md).

```{eval-rst}
.. admonition:: Container Versions

   The latest builds following the main branch are ``dustynv/nano_llm:r35.4.1`` for JetPack 5 and ``dustynv/nano_llm:r36.2.0`` for JetPack 6.  Check the tags on `DockerHub <https://hub.docker.com/r/dustynv/nano_llm/tags>`_ for the versions below.
```
  
## 24.7

* Initial release of [Agent Studio](https://www.jetson-ai-lab.com/agent_studio.html)

## 24.6

* Added LLM backends for AWQ (`--api=awq`) and HuggingFace Transformers (`--api=hf`)
* Support for OpenAI-style tool calling and `NousResearch/Hermes-2-Pro-Llama-3-8B`
* Plugin for [`whisper_trt`](https://github.com/NVIDIA-AI-IOT/whisper_trt) and VAD

## 24.5.1

* Added [Video VILA](multimodal.md#video-sequences) on image sequences
* Added [`KVCache`](models.md#kv-cache) interface for KV cache manipulation
* Updated [`ChatHistory`](chat.md#chat-history) and [`ChatMessage`](chat.md#chat-history) with deletion/removal operators

## 24.5

* Added [function calling](chat.md#function-calling) with Llama 3
* Added [VILA-1.5](https://developer.nvidia.com/blog/visual-language-intelligence-and-edge-ai-2-0/) with TensorRT for CLIP/SigLIP

## 24.4.1

* Added chat templates for Llama 3
* Added [NanoLLM page](https://www.jetson-ai-lab.com/tutorial_nano-llm.html) to Jetson AI Lab
* Added simplified 'Live Llava' code example under [`nano_llm/vision/example.py`](https://github.com/dusty-nv/NanoLLM/blob/main/nano_llm/vision/example.py)
* Changed over some lingering references from `local_llm`

## 24.4

* Migration from `local_llm` in jetson-containers
* Generated docs site ([dusty-nv.github.io/NanoLLM](https://dusty-nv.github.io/NanoLLM))
* Year/month/date versioning with stable releases being cut for each month

