# Installation

Having a complex set of dependencies, currently the recommended installation method is by running the Docker container image built by [jetson-containers](https://github.com/dusty-nv/jetson-containers).  First, clone and install that repo:

```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```

Then you can start the `nano_llm` container like this:

```bash
jetson-containers run $(autotag nano_llm)
```

This will automatically pull/run the container image compatible with your version of JetPack-L4T (e.g. `dustynv/nano_llm:r36.2.0` for JetPack 6.0)

### Running Models

Once in the container, you should be able to `import nano_llm` in a Python3 interpreter, and run the various example commands from the docs like:

```bash
python3 -m nano_llm.chat --model meta-llama/Llama-2-7b-chat-hf --api=mlc --quantization q4f16_ft
```

Or you can run the container & chat command in one go like this:

```bash
jetson-containers run \
  --env HUGGINGFACE_TOKEN=hf_abc123def \
  $(./autotag nano_llm) \
  python3 -m nano_llm.chat --api=mlc \
    --model meta-llama/Llama-2-7b-chat-hf \
    --quantization q4f16_ft
```

Setting your `$HUGGINGFACE_TOKEN` is for models requiring authentication to download (like Llama-2)

### Building In Other Containers

You can either add NanoLLM on top of your container by using it as a base image, or using NanoLLM as the base image in your Dockerfile.  When doing the former use the `--base` argument to `jetson-containers/build.sh` to build it off your container:

```
jetson-containers/build.sh --base my_container:latest --name my_container:llm nano_llm
```

Doing so will also install all the needed dependencies on top of your container (including CUDA, PyTorch, the LLM inference APIs, ect).  It should be based on the same version of Ubuntu as JetPack.  

And in the event that you want to add your own container on top of NanoLLM - thereby skipping its build process - then you can just use a FROM statement (like `FROM dustynv/nano_llm:r36.2.0`) at the top of your Dockerfile.  Or you can make your own [package](https://github.com/dusty-nv/jetson-containers/blob/master/docs/packages.md) with jetson-containers for it. 