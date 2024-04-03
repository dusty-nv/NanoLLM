# Agents

Agents are just plugins that create or connect pipelines of other nested plugins, for implementing higher-level behaviors with more advanced control flow.  They are designed to be layered on top of each other, so that you can combine capabilities of different agents together.  

## Chat Agent

```{eval-rst}
.. autoclass:: nano_llm.agents.chat.ChatAgent
   :members:
   :special-members: __init__
```

## Voice Chat

```{eval-rst}
.. autoclass:: nano_llm.agents.voice_chat.VoiceChat
   :members:
   :special-members: __init__
```

## Web Chat

```{eval-rst}
.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/UOjqF3YCGkY" style="margin-bottom: 1em;" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  
.. autoclass:: nano_llm.agents.web_chat.WebChat
   :members:
   :special-members: __init__
```

## Video Stream

```{eval-rst}
.. autoclass:: nano_llm.agents.video_stream.VideoStream
   :members:
   :special-members: __init__
```

## Video Query

```{eval-rst}
.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/8Eu6zG0eEGY" style="margin-bottom: 1em;" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  
.. autoclass:: nano_llm.agents.video_query.VideoQuery
   :members:
   :special-members: __init__
```
