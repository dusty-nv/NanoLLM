====================
Welcome to NanoLLM!
====================

`NanoLLM <https://www.github.com/dusty-nv/NanoLLM>`_ is a lightweight, high-performance library using optimized inferencing APIs for quantized LLM's, multimodality, speech services, vector databases with RAG, and web frontends.  It can be used to build responsive, low-latency interactive agents that can be deployed on Jetson.

----------
Benchmarks
----------

.. raw:: html

  <iframe width="637" height="400" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vTJ9lFqOIZSfrdnS_0sa2WahzLbpbAbBCTlS049jpOchMCum1hIk-wE_lcNAmLkrZd0OQrI9IkKBfGp/pubchart?oid=1393594867&amp;format=interactive"></iframe>
  
.. raw:: html

  <iframe width="637" height="400" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vTJ9lFqOIZSfrdnS_0sa2WahzLbpbAbBCTlS049jpOchMCum1hIk-wE_lcNAmLkrZd0OQrI9IkKBfGp/pubchart?oid=1784494314&amp;format=interactive"></iframe>

For more info, see the `Benchmarks <https://www.jetson-ai-lab.com/benchmarks.html>`_ on Jetson AI Lab.

-------------
Model Support
-------------

* LLM

  * Llama
  * Mistral
  * Mixtral
  * GPT-2
  * GPT-NeoX
  
* SLM

  * StableLM
  * Phi-2
  * Gemma
  * TinyLlama
  * ShearedLlama
  * OpenLLama
  
* VLM

  * Llava
  * VILA
  * NousHermes/Obsidian
  
* Speech

  * Riva ASR
  * Riva TTS
  * XTTS

See the :ref:`Models` section for more info and API documentation.

-------------
Platform Support
-------------

Currently built for Jetson Orin and JetPack 6.  Containers are provided by `jetson-containers <https://www.github.com/dusty-nv/jetson-containers>`_.

----------
Videos
----------

.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/UOjqF3YCGkY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  
.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/OJT-Ax0CkhU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  
.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/8Eu6zG0eEGY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For more background on generative AI applications in edge devices, visit the `Jetson AI Lab <https://www.jetson-ai-lab.com>`_.





.. toctree::
   :maxdepth: 3
   :caption: Documentation:

   install.md
   models.md
   chat.md
   multimodal.md
   plugins.md
   agents.md
   webserver.md
   utilities.md
   releases.md
