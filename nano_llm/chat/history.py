#!/usr/bin/env python3
import os
import json
import logging
import numpy as np

from .message import ChatMessage
from .templates import ChatTemplate, ChatTemplates
from ..utils import AttributeDict
                        
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
           
           model (NanoLLM) -- the model instance used for embeddings
           
           chat_template (str|dict) -- either a chat template dict, or the name of the 
                                       chat template to use like 'llama-2', 'vicuna-v1'
                                       If None, will attempt to determine model type.
                                  
           system_prompt (str) -- set the default system prompt
                                  if None, will use system prompt from the template.
                                  
           print_stats (bool) -- if True, generation performance will be printed to the terminal after EOS.
                                 This also gets enabled by default if --debug or --verbose is used.
        """
        self.model = model 
        self.messages = None
        
        #: The :class:`KVCache` from :meth:`NanoLLM.generate()` used to store the model state.
        self.kv_cache = None
        
        if not chat_template:
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
            self.template = AttributeDict(template)
        else:
            raise TypeError(f"chat_template should be a str or dict (was {type(chat_template)})")
            
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

        if system_prompt:
            self.template['system_prompt'] = system_prompt

        self.print_stats = kwargs.get('print_stats', kwargs.get('debug', False))
        
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
        else:
            self.messages.append(ChatMessage(role, msg=msg, **kwargs))
            
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
       
    def reset(self, add_system_prompt=True, use_cache=True, wrap_tokens=None):
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
        
        if add_system_prompt and 'system' in self.template:
            self.append(role='system', text=self.system_prompt, use_cache=use_cache)
     
    def to_list(self):
        """
        Serialize the history to a list of dicts, where each dict is a chat entry
        with the non-critical keys removed (suitable for web transport, ect)
        """
        return [{msg.type : msg.content} for msg in self.messages]

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
        if self.template['system_prompt'] == instruction:
            return
            
        self.template['system_prompt'] = instruction
        self.reset()

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
              
            if not use_cache and logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"chat msg {n}  role={msg.role}  type={msg.type}  tokens={msg.num_tokens}  `{msg.template if msg.template else msg.content if isinstance(msg.content, str) else ''}`".replace('\n', '\\n'))

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
            
        logging.debug(f"chat embed  shape={embeddings.shape}  position={position}")
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
            
