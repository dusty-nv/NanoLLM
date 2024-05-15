#!/usr/bin/env python3
# Example for vision/language model inference on continous video streams
import time
import logging

from nano_llm import NanoLLM, ChatHistory, remove_special_tokens
from nano_llm.utils import ArgParser, load_prompts, wrap_text
from nano_llm.plugins import VideoSource, VideoOutput

from termcolor import cprint
from jetson_utils import cudaMemcpy, cudaToNumpy, cudaFont

# parse args and set some defaults
parser = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input', 'video_output'])
parser.add_argument("--max-images", type=int, default=8, help="the number of video frames to keep in the history")
args = parser.parse_args()

prompts = load_prompts(args.prompt)

if not prompts:
    prompts = ["What changes occurred in the video?"] # "Concisely state what is happening in the video."
    
if not args.model:
    args.model = "Efficient-Large-Model/VILA1.5-3b"

print(args)

# load vision/language model
model = NanoLLM.from_pretrained(
    args.model, 
    api=args.api,
    quantization=args.quantization, 
    max_context_len=args.max_context_len,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling, 
)

assert(model.has_vision)

# create the chat history
chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

# warm-up model
chat_history.append(role='user', text='What is 2+2?')
logging.info(f"Warmup response:  '{model.generate(chat_history.embed_chat()[0], streaming=False)}'".replace('\n','\\n'))
chat_history.reset()

# open the video stream
num_images = 0
last_image = None
last_text = ''

def on_video(image):
    global last_image
    last_image = cudaMemcpy(image)
    if last_text:
        font_text = remove_special_tokens(last_text)
        wrap_text(font, image, text=font_text, x=5, y=5, color=(120,215,21), background=font.Gray50)
    video_output(image)
    
video_source = VideoSource(**vars(args))
video_source.add(on_video, threaded=False)
video_source.start()

video_output = VideoOutput(**vars(args))
video_output.start()

font = cudaFont()

# apply the prompts to each frame
while True:
    if last_image is None:
        continue

    chat_history.append(role='user', text=f'Image {num_images + 1}:')
    chat_history.append(role='user', image=last_image)
    
    last_image = None
    num_images += 1

    for prompt in prompts:
        chat_history.append(role='user', msg=prompt)
        embedding, _ = chat_history.embed_chat()
        
        print('>>', prompt)
        
        reply = model.generate(
            embedding,
            kv_cache=chat_history.kv_cache,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        for token in reply:
            cprint(token, 'blue', end='\n\n' if reply.eos else '', flush=True)
            if len(reply.tokens) == 1:
                last_text = token
            else:
                last_text = last_text + token

        chat_history.append(role='bot', text=reply.text, tokens=reply.tokens)
        chat_history.kv_cache = reply.kv_cache
        
        chat_history.pop(2)
        
    if num_images >= args.max_images:
        chat_history.reset()
        num_images = 0
        
    if video_source.eos:
        video_output.stream.Close()
        break
