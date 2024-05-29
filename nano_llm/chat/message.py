#!/usr/bin/env python3
import logging
import datetime

import torch
import numpy as np

from ..utils import ImageExtensions, ImageTypes, print_table


class ChatMessage():
    """
    Create a chat entry consisting of a text message, image, ect as input.  
    
    Args:
    
      role (str): The chat's turn template to apply, typically 'user' or 'bot'.
                  The role should have a corresponding entry in the active ChatTemplate.
       
      text (str): String containing the message's content for text messages.
      
      image (str|image): Either a np.ndarray, torch.Tensor, cudaImage, PIL.Image,
                         or a path to an image file (.jpg, .png, .bmp, ect)

      kwargs: For messages with alternate content types, pass them in via kwargs
              and they will automatically be determined like so::
       
                 message = ChatMessage(role='user', audio='sounds.wav')
                 
              There are additional lower-level kwargs that can be set below.
              
      use_cache (bool): cache the tokens/embeddings for reused prompts (defaults to false)
      tokens (list[int] or np.ndarray): the message contents already having been tokenized
      embedding (np.ndarray): the message contents already having been embedded
      history (ChatHistory): the ChatHistory object this message belongs to
    """    
    def __init__(self, role='user', text=None, image=None, **kwargs):
    
        #: The content or media contained in the message
        self.content = None
        
        #: The type of the message ('text', 'image', 'audio', ect)
        self.type = None
        
        #: The user role or character ('user', 'assistant', 'system', ect)
        self.role = role
        
        #: The version of this message with the role template applied
        self.template = None
        
        #: The tokenized version of the message
        self.tokens = kwargs.get('tokens', None)
        
        #: The embedding of the message
        self.embedding = kwargs.get('embedding', None)
        
        #: The ChatHistory object this message belongs to
        self.history = kwargs.get('history', None)
        
        #: Set to true if the tokens/embeddings should be cached for reused prompts
        self.use_cache = kwargs.get('use_cache', False)
        
        #: Set to true if the message is already in the chat embedding
        self.cached = kwargs.get('cached', self.tokens or self.embedding)
        
        #: The index of this message in the chat history
        self.index = None
        
        #: The previous message in the chat history
        self.prev = None
        
        #: The next message in the chat history
        self.next = None
        
        # Determine the message type
        if text is not None:
            self.content = text
            self.type = 'text'
        elif image is not None:
            self.content = image
            self.type = 'image'
        else:
            for key, value in kwargs.items():
                content_type = self.content_type(value)
                if content_type:
                    self.type = content_type
                    self.content = value
                    break
                    
            if self.type is None:
                raise ValueError(f"couldn't find valid message content in {kwargs}, please specify its type")

        # Apply variable substitutions
        self.apply_substitutions(kwargs.get('substitutions'))
        
    @property
    def num_tokens(self):
        """
        Return the number of tokens used by this message.
        embed() needs to have been called for this to be valid.
        """
        if self.tokens is not None:
            if isinstance(self.tokens, (np.ndarray, torch.Tensor)):
                return self.tokens.shape[1]
            elif isinstance(self.tokens, list):
                return len(self.tokens)
            else:
                raise TypeError(f"ChatMessage had tokens with invalid type ({type(self.tokens)})")
        elif self.embedding is not None:
            return self.embedding.shape[1]
        else:
            return 0

    @property
    def start_token(self):
        """
        The token offset or position in the chat history at which this message begins.
        """
        offset = 0
        
        for i in range(0, self.index):
            offset += self.history[i].num_tokens
            
        return offset
        
    @staticmethod
    def content_type(content):
        """
        Try to automatically determine the message content type.
        """
        if isinstance(content, str):
            if content.endswith(ImageExtensions):
                return 'image'
            else:
                return "text" 
        elif isinstance(content, ImageTypes):
            return 'image'
        else:
            return None
    
    def is_type(self, type):
        """
        Return true if the message is of the given type (like 'text', 'image', ect)
        """
        return (self.type == type)
    
    def apply_substitutions(self, substitutions=None):
        """
        Apply variable substitutions to the message content, like "Today's date is ${DATE}".
        This is separate from the templating that occurs with the special tokens & separators.
        """
        if self.type != 'text' or self.cached or substitutions is False:
            return
            
        if isinstance(substitutions, dict):
            for key, value in substitutions.items():
                self.content = self.content.replace(key, value)
            return
            
        if "${DATE}" in self.content:
            self.content = self.content.replace("${DATE}", datetime.date.today().strftime("%Y-%m-%d"))
            
        if "${TIME}" in self.content:
            self.content = self.content.replace("${TIME}", datetime.datetime.now().strftime("%-I:%M %p"))
           
        if "${TOOLS}" in self.content:
            from nano_llm import BotFunctions
            self.content = self.content.replace("${TOOLS}", BotFunctions.generate_docs(style=self.history.tool_style))
          
        if "${LOCATION}" in self.content:
            from nano_llm.plugins.bot_functions.location import LOCATION
            self.content = self.content.replace("${LOCATION}", LOCATION())
               
    def embed(self, return_tensors='np', **kwargs):
        """
        Apply message templates, tokenization, and generate the embedding.
        """
        if self.embedding is not None:
            return self.embedding
            
        if self.tokens is not None and not self.history.model.has_embed:
            if isinstance(self.tokens, list):
                self.tokens = np.expand_dims(np.asarray(self.tokens, dtype=np.int32), axis=0)
            return self.tokens
          
        if self.history is None:
            raise RuntimeError("this message needs to be added to a ChatHistory before embed() is called")
             
        # lookup the role template to apply
        first_msg = 1 if 'system' in self.history.template else 0
        role = 'first' if 'first' in self.history.template and self.index == first_msg else self.role

        if role not in self.history.template:
            raise RuntimeError(f"chat template {self.history.template.get('name', '')} didn't have a role defined for '{entry.role}' (had keys: {self.history.template.keys()})")
         
        # extract template prefix/postfix
        template = self.history.template[role]
        split_template = template.split('${MESSAGE}')
        
        if len(split_template) == 1:  # there was no ${MESSAGE}
            split_template.append('')

        if self.prev and self.prev.role == self.role:
            split_template[0] = ''
            
        if self.next and self.next.role == self.role:
            split_template[1] = ''
         
        # embed based on media type
        if self.type == 'text':
            self._embed_text(self.history.model, split_template, return_tensors=return_tensors, **kwargs)
        elif self.type == 'image':
            self._embed_image(self.history.model, split_template, return_tensors=return_tensors, **kwargs)
            
        # mark as cached
        self.cached = True
        
        if self.embedding is not None:
            return self.embedding
            
        if self.tokens is not None:
            return self.tokens

    def _embed_text(self, model, template, return_tensors='np', **kwargs):
        """
        Generate the token embeddings for a text message.
        """
        self.template = template[0] + self.content + template[1]
        
        if self.tokens is not None:
            if model.has_embed:
                self.embedding = model.embed_tokens(self.tokens, return_tensors=return_tensors, **kwargs)
        else:     
            self.embedding, self.tokens = model.embed_text(
                self.template, use_cache=self.use_cache,
                return_tensors=return_tensors, return_tokens=True,
                **kwargs
            )
    
    def _embed_image(self, model, template, return_tensors='np', **kwargs):
        """
        Generate the encoded vision embeddings for an image.
        """
        if not model.has_vision:
            raise RuntimeError(f"attempted to embed an image in the chat, but '{model.config.name}' was not a multimodal vision model")

        # add the template prefix                
        embeddings = [] 

        if template[0]:
            embeddings.append(model.embed_text(template[0], use_cache=True, return_tensors=return_tensors))
            
        # encode the image
        image_outputs = model.embed_image(self.content, return_tensors=return_tensors, return_dict=True)
        self.history.image_embedding = image_outputs.image_embeds # save the unprojected embeddings for RAG
        embeddings.append(image_outputs.embedding)
        
        # add the template trailer
        template[1] = '\n' + template[1]
        
        if template[1]:
            embeddings.append(model.embed_text(template[1], use_cache=True, return_tensors=return_tensors))
                
        # concatenate all embeddings
        self.embedding = np.concatenate(embeddings, axis=1)
        
        if self.history.print_stats:
            print_table(model.vision.stats)
            
        logging.debug(f"chat embed image  shape={self.embedding.shape}  dtype={self.embedding.dtype}  template={template}")
