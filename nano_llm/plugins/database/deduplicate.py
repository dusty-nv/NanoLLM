#!/usr/bin/env python3
import math
import time
import logging
import numpy as np

from nano_llm import Plugin


class Deduplicate(Plugin):
    """
    Filter out similar text over a given time period to prevent repetitive messages.
    Use of numberbatch embeddings and cosine similarity from https://stackoverflow.com/a/53407328  
    """
    EmbeddingsIndex = {}
    
    def __init__(self, similarity_threshold: float = 0.25, timeout: float = 10.0, **kwargs):
        """
        Filter out similar text over a given time period to prevent repetitive messages. 
        
        Args:
          similarity_threshold (float):  How similar the text should be to considered matching (between 0 and 1)
          timeout (float):  The time in seconds after which the previous text is forgotten and new text used.
        """ 
        super().__init__(**{'inputs': 'text', 'outputs': 'text', **kwargs})

        self.last_time = 0
        self.last_text = None
        self.last_embed = None

        if not self.EmbeddingsIndex:
            numberbatch_path = "/data/nano_llm/numberbatch-en.txt"
            logging.info(f"{self.name} loading {numberbatch_path}")
            with open(numberbatch_path) as f:
                for line in f:
                    values = line.split(' ')
                    word = values[0]
                    embedding = np.asarray(values[1:], dtype='float32')
                    self.EmbeddingsIndex[word] = embedding
        
        self.add_parameter('similarity_threshold', default=similarity_threshold, end=True)
        self.add_parameter('timeout', default=timeout, end=True)
 
 
    def process(self, input, **kwargs):
        """
        Drop messages that are too similar to the previous.
        """
        current_time = time.perf_counter()
        
        if self.last_text and (current_time - self.last_time < float(self.timeout)):
            input_embed = self.embed_text(input)
            
            if self.last_embed is None:
                self.last_embed = self.embed_text(self.last_text)
                
            similarity = self.cosine_similarity(input_embed, self.last_embed)
            
            logging.debug(f"{self.name}  similarity={similarity:.5f}  threshold={self.similarity_threshold}  {[self.last_text, input]}")
                      
            self.send_stats({
                'summary': [f"{similarity:.3f}"]
            })
            
            if similarity < self.similarity_threshold:
                self.last_embed = input_embed
            else:
                return None

        self.last_text = input
        self.last_time = current_time
        
        return input
        
        
    def cosine_similarity(self, v1, v2):
        """
        Compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        """
        if isinstance(v1, str):
            v1 = self.embed_text(v1)
            
        if isinstance(v2, str):
            v2 = self.embed_text(v2)
               
        sumxx, sumxy, sumyy = 0, 0, 0
        
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
            
        return sumxy/math.sqrt(sumxx*sumyy)


    def embed_text(self, sentence):
        """
        Lookup and assemble the embedding of the given text.
        """
        sent_vector = 0
        
        for word in sentence.lower().split():
            if word not in self.EmbeddingsIndex:
                word_vector = np.array(np.random.uniform(-1.0, 1.0, 300))
                self.EmbeddingsIndex[word] = word_vector
            else:
                word_vector = self.EmbeddingsIndex[word]
            sent_vector = sent_vector + word_vector

        return sent_vector
