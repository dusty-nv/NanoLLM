#!/usr/bin/env python3
import os
import re
import json
import logging
import termcolor
import numpy as np

from .message import ChatMessage
from .stream import StreamingResponse
from .templates import ChatTemplate, ChatTemplates, StopTokens
from ..utils import AttributeDict, escape_html, code_tags
                        
class ChatHistory():
    """
    Multimodal chat history that can contain a mix of media including text/images.
    
    ChatHistory objects can be indexed like a list to access its messages,
    where each :class:`ChatMessage` can have a different type of content::
    
       chat_history[n]  # will return the n-th chat entry

    Each type of media has an associated embedding function (e.g. LLM's typically 
    do text token embedding internally, and images use CLIP + projection layers). 
    From these, it assembles the embedding for the entire chat as input to the LLM.
    
    It uses templating to add the required special tokens as defined by different
    model architectures.  In normal 2-turn chat, there are 'user' and 'bot' roles
    defined, but arbitrary roles can be added, each with their own template.
    
    The system prompt can also be configured through the chat template
    and by setting the :attr:`ChatHistory.system_prompt` property.
    """
    def __init__(self, model, chat_template=None, system_prompt=None, **kwargs):
        """
        Parameters:
           
           model (NanoLLM):  The model instance used for embeddings
           
           chat_template (str|dict):  Either a chat template dict, or the name of the 
                                      chat template to use like ``llama-2``, ``vicuna-v1``
                                      If None, will attempt to determine model type.
                                  
           system_prompt (str):  Set the default system prompt used at the beginning of chats.
                                 If ``None``, will use system prompt from the template by default.
           
           tools (bool|str):  If True, tool calling will be enabled for models that have the
                              ``tool_call`` and ``tool_response`` roles in their chat templates.
                              When enabled, the function descriptors will automatically be generated
                              from their pydoc strings, and appended to the system prompt.
                                   
           print_stats (bool):  If True, generation performance will be printed to the terminal after EOS.
                                This also gets enabled by default if ``--debug`` or ``--verbose`` is used.
        """
        self.model = model 
        self.messages = None
        
        #: The :class:`KVCache` from :meth:`NanoLLM.generate()` used to store the model state.
        self.kv_cache = None
        
        # look-up or load the chat template
        if not chat_template or chat_template == 'auto':
            self.template = ChatTemplate(model)
            if self.template is None:
                raise RuntimeError(f"Couldn't automatically determine model type from {model.config.name}, please set the --chat-template argument")
            logging.info(f"using chat template '{self.template.name}' for model {model.config.name}")
        elif isinstance(chat_template, str):
            if os.path.isfile(chat_template):
                with open(chat_template) as template_file:
                    self.template = AttributeDict(json.load(template_file))
            else:
                self.template = AttributeDict(ChatTemplates[chat_template])
        elif isinstance(chat_template, dict):
            self.template = AttributeDict(chat_template)
        else:
            raise TypeError(f"chat_template should be a str or dict (was {type(chat_template)})")

        # parse the stop tokens    
        if 'stop' in self.template:
            if not isinstance(self.template.stop, list):
                self.template.stop = [self.template.stop]
                
            for i, stop in enumerate(self.template.stop):
                if isinstance(stop, str):
                    self.template.stop[i] = self.model.tokenizer(stop, add_special_tokens=False, return_tensors='np').input_ids.squeeze().tolist()
        else:
            self.template.stop = [self.model.tokenizer.eos_token_id]
         
        #self.template.stop = [x for x in self.template.stop if x >= 0]  # filter out ignored stop tokens
        logging.info(f"model '{self.model.config.name}', chat template '{self.template.name}' stop tokens:  {self.model.tokenizer.batch_decode(self.template.stop)} -> {self.template.stop}")      

        # setup the default system prompt
        if system_prompt:
            self.template['system_prompt'] = system_prompt

        # try to determine the function-calling style
        if 'tool_spec' not in self.template:
            if 'tool_call' in self.template:
                self.template.tool_spec = 'openai'
            else:
                self.template.tool_spec = kwargs.get('tool_spec')

        self.print_stats = kwargs.get('print_stats', kwargs.get('debug', False))
        
        self.web_regex = [
            (re.compile(r'`(.*?)`'), r'<code>\1</code>'),  # code blocks
            (re.compile(r'\*(.*?)\*'), r'*<i>\1</i>*'),    # emotives inside asterisks
        ]
        
        from nano_llm import BotFunctions
        self.BotFunctions = BotFunctions
        
        self.reset()

    @property
    def num_tokens(self):
        """
        Return the number of tokens used by the chat so far.
        :meth:`embed_chat()` needs to have been called for this to be upated,
        because otherwise the input wouldn't have been tokenized yet.
        """
        position = 0
        for msg in self.messages:
            position += msg.num_tokens
        return position
        
    def __len__(self):
        """
        Returns the number of messages in the chat history
        """
        return len(self.messages)
        
    def __getitem__(self, key):
        """
        Return the n-th chat message with the subscript indexing operator
        """
        return self.messages[key]
        
    def __delitem__(self, key):
        """
        Remove one or more messages from the chat history::
        
           del chat_history[-2]   # remove the second-to-last entry
           del chat_history[-2:]  # pop the last 2 entries
           del chat_history[1:]   # remove all entries but the first
           
        This will also update the KV cache and alter the bot memory.
        """
        if isinstance(key, int):
            start = key
            stop = key + 1
        elif isinstance(key, ChatMessage):
            start = self.messages.index(key)
            stop = start + 1
        elif isinstance(key, slice):
            start = key.start
            stop = key.stop
        else:
            raise TypeError(f"The `del chat_history[*]` operator expects an int, ChatMessage, or slice (was '{type(key)}')")
        
        if start is None:
            start = 0
            
        if stop is None:
            stop = len(self.messages)
      
        self.remove(start, stop)
     
    def append(self, role='user', msg=None, **kwargs):
        """
        Add a chat entry consisting of a text message, image, ect.
        See the :class:`ChatMessage` class for description of arguments.
        This can also accept an existing :class:`ChatMessage` set to ``msg``.
        """
        if isinstance(msg, ChatMessage):
            self.messages.append(msg)
        elif isinstance(msg, StreamingResponse):
            self.messages.append(ChatMessage(role, text=msg.text, tokens=msg.tokens, history=self, **kwargs))
            self.kv_cache = msg.kv_cache
        else:
            self.messages.append(ChatMessage(role, msg=msg, history=self, **kwargs))
            
        self.reindex()
        return self.messages[-1]

    def pop(self, count):
        """
        Remove the last N messages from the chat and KV cache.
        """
        num_tokens = 0
        
        for n in range(0, count):
            num_tokens += self.messages[len(self.messages)-n-1].num_tokens
            
        if self.kv_cache:
            self.kv_cache.pop(num_tokens)
            
        del self.messages[-count:]
        self.reindex()

    def remove(self, start, stop=None):
        """
        Remove the chat entries from the start (inclusive) to stop (exclusive) indexes.
        If stop is not specified, then only the single entry at the start index will be removed::
        
          chat_history.remove(0)    # remove the first chat entry
          chat_history.remove(0,2)  # remove the first and second chat entries
          chat_history.remove(-1)   # remove the last chat entry
          chat_history.remove(-2,0) # remove the last two entries
          
        This will also update the KV cache and alter the bot's memory (potentially destructively)
        """
        num_messages = len(self.messages)
        
        if stop is None:
            stop = start + 1
             
        if start < 0:
            start += num_messages
            
        if stop <= 0:
            stop += num_messages

        if stop > num_messages:
            raise ValueError(f"remove index {stop} exceeded the number of messages ({num_messages})")
            
        if stop == num_messages:
            return self.pop(num_messages - start)
       
        if self.kv_cache:
            self.kv_cache.remove(self.messages[start].start_token, self.messages[stop].start_token)
            
        del self.messages[start:stop]       
        self.reindex()
       
    def reset(self, system_prompt=True, use_cache=True, wrap_tokens=None):
        """
        Reset the chat history, and optionally add the system prompt to the new chat.
        If ``use_cache=True``, then the system prompt tokens/embedding will be cached.
        If `wrap_tokens` is set, then the most recent N tokens from the chat will be kept.
        """
        if wrap_tokens:
            wrap_entry = self.find_wrap_entry(wrap_tokens)
            if wrap_entry:
                logging.warning(f"Wrapping chat to keep the most recent {len(self.messages)-wrap_entry} messages")
                self.messages = self.messages[wrap_entry:]
            else:
                logging.warning(f"Chat history overflow couldn't find previous chat entry to wrap to (clearing chat)")
                self.messages = []
        else:
            self.messages = []

        self.kv_cache = None
        self.image_embedding = None
        
        if isinstance(system_prompt, str):
            self.add_system_prompt(system_prompt=system_prompt, use_cache=use_cache)
        elif system_prompt:
            self.add_system_prompt(use_cache=use_cache)

    def turn(self, role='user'):
        """
        Returns true if it's the given role's turn in the chat, otherwise false.
        """
        n = len(self.messages)
        prev_role = self.messages[n-1].role if n > 0 else None
        
        if role == 'system':
            return (n == 0)
        elif role == 'user':
            if n == 0:
                return ('system' not in self.template)
            else:
                return (prev_role != 'tool_response')
        elif role == 'bot':
            return (prev_role == 'user' or prev_role == 'tool_response')
        else:
            logging.warning(f"unrecognized role in ChatHistory.turn() (role={role})")
            
        return True
        
    def to_list(self, messages=None, html=False):
        """
        Serialize the history to a list of dicts, where each dict is a chat entry
        with the non-critical keys removed (suitable for web transport, ect)
        """
        if messages is None:
            messages = self.messages
        
        if messages and isinstance(messages[0], ChatMessage):    
            messages = [{'role' : msg.role, msg.type : msg.content} for msg in messages]
        
        if html:
            messages = self.to_html(messages)
            
        return messages

    def add_system_prompt(self, system_prompt=None, use_cache=True):
        """
        Add the system prompt message to the chat, containing :attr:`ChatHistory.system_prompt`
        appended by the tool function descriptions if tools are enabled.  If the ``system`` role
        is not defined by the model's chat template, then this function does nothing.
        
        Arguments:
        
            use_cache (bool):  If true, then the system prompt tokens/embeddedings will be cached.
                               This is the default because the system prompt typically may not change.
        Returns:
        
            The :class:`ChatMessage` that was added to the chat with the ``system`` role.
        """
        if 'system' not in self.template:
            return None

        if system_prompt is not None:
            self.template.system_prompt = system_prompt
        
        return self.append(role='system', text=self.template.system_prompt, use_cache=use_cache)
            
    @property
    def system_prompt(self):
        """
        Get the system prompt, the typically hidden instruction at the beginning
        of the chat like "You are a curious and helpful AI assistant, ..."
        """
        return self.template.get('system_prompt', '')
        
    @system_prompt.setter
    def system_prompt(self, instruction):
        """
        Set the system prompt instruction string and reset the chat history.
        TODO make it so this doesn't reset the chat history, but uncaches it.
        """
        if instruction is None:
            return
            
        if self.template['system_prompt'] == instruction:
            return

        self.reset(system_prompt=instruction)

    def embed_chat(self, use_cache=True, max_tokens=None, wrap_tokens=None, **kwargs):
        """
        Assemble the embedding of either the latest or entire chat.
        
        If ``use_cache=True`` (the default), and only the new embeddings will be returned.
        If ``use_cache=False``, then the entire chat history will be returned.
        
        This function returns an ``(embedding, position)`` tuple, where the embedding array
        contains the new embeddings (or tokens) from the chat, and position is the current
        overall position in the history (up to the model's context window length)
        
        If the number of tokens in the chat history exceeds the length given in ``max_tokens`` argument
        (which is typically the model's context window, minus the max generation length),
        then the chat history will drop all but the latest ``wrap_tokens``, starting with a user prompt.
        If `max_tokens` is provided but `wrap_tokens` is not, then the overflow tokens will be truncated.
        """
        embeddings = []
        position = 0
      
        for n, msg in enumerate(self.messages):
            if use_cache:
                if msg.cached:
                    position += msg.num_tokens
                else:
                    embeddings.append(msg.embed())
                    use_cache = False  # all entries after this need to be included
            else:
                embeddings.append(msg.embed())
              
            if not use_cache and logging.getLogger().isEnabledFor(logging.DEBUG) and (len(self.messages) - n < 5):
                logging.debug(f"chat msg {n}  role={msg.role}  type={msg.type}  tokens={msg.num_tokens}  `{msg.template if msg.template else msg.content if isinstance(msg.content, str) else ''}`".replace('\n', '\\n'))

        entries = len(embeddings)
        embeddings = np.concatenate(embeddings, axis=1) #, position

        '''
        if max_tokens and position + embeddings.shape[1] > max_tokens:
            if wrap_tokens:
                self.reset(wrap_tokens=wrap_tokens)
                embeddings, position = self.embed_chat(use_cache=False, max_tokens=max_tokens, wrap_tokens=wrap_tokens, **kwargs)
                logging.warning(f"Chat overflow, max history lenth {max_tokens} tokens exceeded (keeping the most recent {embeddings.shape[1]} tokens)")
            else:
                logging.warning(f"Truncating chat history overflow to {max_tokens} tokens")
                return embeddings[:,:max_tokens,:], position
        '''
            
        logging.debug(f"chat embed  entries={entries}  shape={embeddings.shape}  position={position}")
        return embeddings, position  

    def reindex(self):
        """
        Update the linked lists in the messages that refer to each other.
        This gets called after messages are added, removed, or their order changed.
        You wouldn't typically need to call this yourself.
        """
        for i, msg in enumerate(self.messages):
            msg.index = i
            msg.history = self
            
            if i == 0:
                msg.prev = None
            elif i > 0:
                msg.prev = self.messages[i-1]
                msg.prev.next = msg
                
            if i >= len(self.messages) - 1:
                msg.next = None
           
    def find_wrap_entry(self, wrap_tokens):
        """
        Find the oldest entry from which the chat doesn't exceed the number of wrap_tokens,
        and that the entry should be a user query.  This is used to keep those more recent
        chat entries when the history overflows past the max context window of the model.
        """
        position = 0
        for n in range(len(self.messages)-1, -1, -1):
            msg = self.messages[n]
            position += msg.num_tokens
            if position >= wrap_tokens:
                for i in range(n+1, len(self.messages)):
                    if self.messages[i].role == 'user':
                        return i
     
    def to_html(self, messages=None):
        """
        Sanitize message contents to HTML representation, apply code formatting, ect.       
        """
        messages = self.to_list(messages, html=False)
        
        def web_text(text):
            for stop_token in StopTokens:
                text = text.replace(stop_token, '')
               
            text = text.strip()
            text = text.strip('\n')
             
            if text.find('<tool_call>') == 0:
                text = text.replace('\n', '')

            text = text.replace('<s>', '')
            text = escape_html(text)
            
            for regex, replace in self.web_regex:
                text = regex.sub(replace, text)

            return code_tags(text)
          
        def web_image(image):
            from nano_llm.web import WebServer
            
            if not isinstance(image, str):
                if not hasattr(image, 'filename'):
                    return None
                image = image.filename
                
            if WebServer.Instance:
                return os.path.join(self.server.mounts.get(os.path.dirname(image), ''), os.path.basename(image))
            else:
                return image
                
        for entry in messages:
            if 'text' in entry:
                entry['text'] = web_text(entry['text'])
            if 'image' in entry:
                entry['image'] = web_image(entry['image'])
                if not entry['image']:
                    del entry['image']
                
        return messages
     
    def run_tools(self, message, tools={}, append=True):
        """
        Invoke any function calls in the output text and return the results.
        """  
        if not tools:
            return None
              
        if isinstance(message, ChatMessage):
            text = message.content if message.is_type('text') else None
        elif isinstance(message, dict):
            text = message.get('text')
        elif isinstance(message, str):
            text = message
        else:
            raise ValueError("expected a message dict or string (was {type(message)})")
            
        if not text:
            return None

        tool_response = self.BotFunctions.run(text, template=self.template, functions=tools)

        if not tool_response:
            return None
            
        if append:
            self.append('tool_response', tool_response)
            
        return tool_response

