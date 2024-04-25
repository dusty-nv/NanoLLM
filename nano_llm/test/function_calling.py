#!/usr/bin/env python3
import os
import logging
import traceback
import pprint

from nano_llm import NanoLLM, ChatHistory, Plugin, BotFunctions, bot_function
from nano_llm.utils import ArgParser, load_prompts, print_table

from termcolor import cprint


# parse args and set some defaults
parser = ArgParser(extras=ArgParser.Defaults + ['prompt'])
parser.add_argument('--disable-plugins', action='store_true')
args = parser.parse_args()

prompts = load_prompts(args.prompt)



'''
def TIME(text):
    if text.endswith('`TIME()`'):
        return datetime.now().strftime("%-I:%M %p")
'''

user_profile=[]

@bot_function(docs='nosig')
def SAVE(text=None):
    """
    `SAVE("text")` - save text to the database that can be retrieved later, for example `SAVE("the user's name is John")`
    """
    global user_profile
    if text:
        user_profile.append(text)
    
    
    
print('num BotFunctions', len(BotFunctions()))
print('\n\n' + BotFunctions.generate_docs())
print(BotFunctions())




if not prompts:
    prompts = [
        'You have some functions available to you like `TIME()`, `DATE()`, `SAVE("text"), `CURRENT_WEATHER()`, `WEATHER_FORECAST(day=1)`, and `LOCATION()` which will add those to the chat - please call them when required.  `LOCATION()` will return the closest city and state.  `WEATHER_FORECAST()` has an optional `day` keyword argument that specifies the number of days ahead for the forecast (by default, tomorrow).  If the user tells you personal details about themselves such as their name, save it with the `SAVE("text")` function - for example, `SAVE("The user\'s name is John")` - you need to actually call this in order for it to be saved, and it will return nothing.  Always surround the code you write by backticks, for example `DATE()`, so that it can be parsed and executed.\nHi I\'m Dusty! how are you today?', 
        #"Hi, how are you today?",
        "Yes, please save my name so you remember it in the future!",
        "I have brown hair and green eyes.",
        "What month is it?",
        "What time is it?",
        #"Where are we?",
        "What city are we in?",
        "What is the temperature?",
        "How warm will it be in 2 days?",
    ]
    
if not args.model:
    args.model = "meta-llama/Meta-Llama-3-8B-Instruct"

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

# create the chat history
chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

while True: 
    # get the next prompt from the list, or from the user interactivey
    if isinstance(prompts, list):
        if len(prompts) > 0:
            user_prompt = prompts.pop(0)
            cprint(f'>> PROMPT: {user_prompt}', 'blue')
        else:
            break
    else:
        cprint('>> PROMPT: ', 'blue', end='', flush=True)
        user_prompt = sys.stdin.readline().strip()
    
    print('')
    
    # special commands:  load prompts from file
    # 'reset' or 'clear' resets the chat history
    if user_prompt.lower().endswith(('.txt', '.json')):
        user_prompt = ' '.join(load_prompts(user_prompt))
    elif user_prompt.lower() == 'reset' or user_prompt.lower() == 'clear':
        logging.info("resetting chat history")
        chat_history.reset()
        continue

    # add the latest user prompt to the chat history
    entry = chat_history.append(role='user', msg=user_prompt)

    # get the latest embeddings (or tokens) from the chat
    embedding, position = chat_history.embed_chat(
        max_tokens=model.config.max_length - args.max_new_tokens,
        wrap_tokens=args.wrap_tokens,
        return_tokens=not model.has_embed,
    )

    # generate bot reply
    reply = model.generate(
        embedding, 
        plugins=BotFunctions() if not args.disable_plugins else None, 
        kv_cache=chat_history.kv_cache,
        stop_tokens=chat_history.template.stop,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
    )
        
    # stream the output
    for token in reply:
        print(token, end='', flush=True)
        #cprint(f"NEW TOKEN '{token}'", "yellow")
        
    print('\n')
    #cprint(reply.text, 'green')
    
    #print_table(model.stats)
    #print('')
    
    chat_history.append(role='bot', text=reply.text) # save the output
    chat_history.kv_cache = reply.kv_cache           # save the KV cache 
