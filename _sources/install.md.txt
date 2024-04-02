# Installation

To use the optimized API's like MLC and AWQ built with CUDA, the recommended installation method is by running the Docker container image built by [jetson-containers](https://github.com/dusty-nv/jetson-containers).  First, clone that repo:

```bash
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
pip3 install -r requirements.txt
```

Then you can start `nano_llm` container like this:

```bash
./run.sh $(./autotag nano_llm)
```

This will automatically pull/run the container image compatible with your version of JetPack-L4T (e.g. `dustynv/nano_llm:r36.2.0` for JetPack 6.0 DP)

Once in the container, you should be able to `import nano_llm` in a Python3 interpreter, and run the various example commands shown on this page like:

```bash
python3 -m nano_llm.chat --model meta-llama/Llama-2-7b-chat-hf --api=mlc --quantization q4f16_ft
```

Or you can run the container & chat command in one go like this:

```bash
./run.sh --env HUGGINGFACE_TOKEN=hf_abc123def \
  $(./autotag nano_llm) \
  python3 -m nano_llm.chat --api=mlc \
    --model meta-llama/Llama-2-7b-chat-hf \
    --quantization q4f16_ft
```

Setting your `$HUGGINGFACE_TOKEN` is for models requiring authentication to download (like Llama-2)