# Multimodality

```{eval-rst}
.. admonition:: Multimodal Agents on Jetson AI Lab

   Refer to these guides and tutorials on Jetson AI Lab: `llamaspeak <https://www.jetson-ai-lab.com/tutorial_llamaspeak.html>`_ | `Live Llava <https://www.jetson-ai-lab.com/tutorial_live-llava.html>`_ | `NanoVLM <https://www.jetson-ai-lab.com/tutorial_nano-vlm.html>`_
```

NanoLLM provides optimized multimodal pipelines, including vision/language models (VLM), vector databases ([NanoDB](https://www.jetson-ai-lab.com/tutorial_nanodb.html)), and speech services that can be integrated into interactive agents.  

```{eval-rst}
.. raw:: html

  <iframe width="637" height="400" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vTJ9lFqOIZSfrdnS_0sa2WahzLbpbAbBCTlS049jpOchMCum1hIk-wE_lcNAmLkrZd0OQrI9IkKBfGp/pubchart?oid=1784494314&amp;format=interactive"></iframe>

  <a href="https://www.jetson-ai-lab.com/tutorial_live-llava.html" target="_blank"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava.gif"></img></a>
```

These are implemented through the [model](model.md) and [chat interfaces](chat.md) covered in the previous sections:

```bash
python3 -m nano_llm.chat --api=mlc \
  --model Efficient-Large-Model/VILA-2.7b \
  --prompt '/data/images/lake.jpg' \
  --prompt 'please describe the scene.' \
  --prompt 'are there any hazards to be aware of?'
```

## Code Example

```{eval-rst}
.. literalinclude:: ../nano_llm/vision/example.py
```

## Multimodal Demos

[**llamaspeak**](https://www.jetson-ai-lab.com/tutorial_llamaspeak.html) - Talk live with Llama using Riva ASR/TTS, and chat about images with VLMs.

```{eval-rst}
.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/UOjqF3YCGkY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  
  <br/>
  <br/>
```

[**Live Llava**](https://www.jetson-ai-lab.com/tutorial_live-llava.html) - Run multimodal models on live video streams over a repeating set of prompts.

```{eval-rst}
.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/8Eu6zG0eEGY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    
  <br/>
  <br/>
```
