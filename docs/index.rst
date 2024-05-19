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

  <img width="637" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vTJ9lFqOIZSfrdnS_0sa2WahzLbpbAbBCTlS049jpOchMCum1hIk-wE_lcNAmLkrZd0OQrI9IkKBfGp/pubchart?oid=88720541&format=image"></img>

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
  * Piper TTS
  * XTTS

See the :ref:`Models` section for more info and API documentation.

-------------
Containers
-------------

Currently supported on Jetson Orin and JetPack 5/6.  Containers are built by `jetson-containers <https://www.github.com/dusty-nv/jetson-containers>`_ with images available on `DockerHub <https://hub.docker.com/r/dustynv/nano_llm/tags>`_.  These are the monthly releases (there are also point releases):

.. list-table:: Container Images
   :header-rows: 1
   
   * - Version
     - JetPack 5
     - JetPack 6
   * - main
     - ``dustynv/nano_llm:r35.4.1``
     - ``dustynv/nano_llm:r36.2.0``
   * - 24.5
     - ``dustynv/nano_llm:24.5-r35.4.1``
     - ``dustynv/nano_llm:24.5-r36.2.0``
   * - 24.4
     - ``dustynv/nano_llm:24.4-r35.4.1``
     - ``dustynv/nano_llm:24.4-r36.2.0``
     
See the :ref:`Release Notes` and :ref:`Installation Guide <Installation>` for info about running the containers and samples.


----------
Videos
----------

.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/UOjqF3YCGkY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  
.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/OJT-Ax0CkhU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  
.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/8Eu6zG0eEGY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/7lKBJPpasAQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

.. raw:: html

  <iframe width="720" height="405" src="https://www.youtube.com/embed/_7gughth8C0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
   
  <br/>
  <br/>

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
