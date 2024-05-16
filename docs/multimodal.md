# Multimodal

```{eval-rst}
.. admonition:: Multimodal Agents on Jetson AI Lab

   Refer to these guides and tutorials on Jetson AI Lab: `llamaspeak <https://www.jetson-ai-lab.com/tutorial_llamaspeak.html>`_ | `Live Llava <https://www.jetson-ai-lab.com/tutorial_live-llava.html>`_ | `NanoVLM <https://www.jetson-ai-lab.com/tutorial_nano-vlm.html>`_
```

NanoLLM provides optimized multimodal pipelines, including vision/language models (VLM), vector databases ([NanoDB](https://www.jetson-ai-lab.com/tutorial_nanodb.html)), and speech services that can be integrated into interactive agents.  

```{eval-rst}
.. raw:: html

  <iframe width="719" height="446" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vTJ9lFqOIZSfrdnS_0sa2WahzLbpbAbBCTlS049jpOchMCum1hIk-wE_lcNAmLkrZd0OQrI9IkKBfGp/pubchart?oid=88720541&amp;format=interactive"></iframe>

  <a href="https://www.jetson-ai-lab.com/tutorial_live-llava.html" target="_blank"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava.gif"></img></a>
```

These are implemented through the [model](model.md) and [chat interfaces](chat.md) covered in the previous sections:

```bash
python3 -m nano_llm.chat --api=mlc \
  --model Efficient-Large-Model/VILA1.5-3b \
  --prompt '/data/images/lake.jpg' \
  --prompt 'please describe the scene.' \
  --prompt 'are there any hazards to be aware of?'
```

See the [Tested Models](models.md#tested-models) section for the list of multimodal models that are supported in NanoLLM.

## Image Messages

You can get a vision/language model to respond about an image by adding it to the [chat history](chat.md), and then asking a query about it:

```python
chat_history.append(role='user', image=img) # np.ndarray, torch.Tensor, PIL.Image, cudaImage
chat_history.append(role='user', text='Describe the image.')

print(model.generate(chat_history.embed_chat()[0], streaming=False))
```

Image messages will be embedded into the chat using the model's CLIP/SigLIP vision encoder and multimodal projector.  Supported image types are `np.ndarray`, `torch.Tensor`, `PIL.Image`, `jetson_utils.cudaImage`, and URLs or local paths to image files (*jpg, png, tga, bmp, gif*)

## Code Example

```{eval-rst}
.. literalinclude:: ../nano_llm/vision/example.py
```

## Video Sequences


The code in [`vision/video.py`](https://github.com/dusty-nv/NanoLLM/blob/main/nano_llm/vision/video.py) keeps a rolling history of image frames and can be used with models that were trained to understand video (like [VILA-1.5](https://github.com/Efficient-Large-Model/VILA)) to apply video summarization, action & behavior analysis, change detection, and other temporal-based vision functions:

```
  python3 -m nano_llm.vision.video \
    --model Efficient-Large-Model/VILA1.5-3b \
    --max-images 8 \
    --max-new-tokens 48 \
    --video-input /data/my_video.mp4 \
    --video-output /data/my_output.mp4 \
    --prompt 'What changes occurred in the video?'
``` 

<a href="https://youtu.be/_7gughth8C0" target="_blank"><img src="https://jetson-ai-lab.com/images/video_vila_wildfire.gif" title="Link to YouTube video of more clips (Realtime Video Vision/Language Model with VILA1.5-3b and Jetson Orin)"></a>


## Multimodal Agents

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

[**Video VILA**](https://www.jetson-ai-lab.com/tutorial_nano-vlm.html#video-sequences) - Process multiple images per query for temporal understanding of video sequences.

```{eval-rst}
.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/_7gughth8C0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    
  <br/>
  <br/>
```
