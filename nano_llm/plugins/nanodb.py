#!/usr/bin/env python3
import nanodb
import torch
import json

from nano_llm import Plugin
from nano_llm.web import WebServer
from nano_llm.utils import torch_image

from jetson_utils import cudaImage


class NanoDB(Plugin):
    """
    Plugin that loads a NanoDB database and searches it for incoming text/images.   
    """
    def __init__(self, path: str = "/data/nanodb/coco/2017", 
                 model: str = "openai/clip-vit-large-patch14-336", dtype: str = 'float16',
                 reserve: int = 1024, top_k: int = 8, crop: bool = False, **kwargs):
        """
        Multimodal vector database with CUDA and CLIP/SigLIP embeddings.
        
        Args:
          path (str):  The directory to either load or create the NanoDB database under.
          model (str):  The CLIP or SigLIP embedding model to use (on HuggingFace or local path)
          dtype (str):  Whether to compute and store the embeddings in 16-bit or 32-bit floating point.
          reserve (int):  The number of megabytes (MB) to reserve for the database vectors.
          top_k (int):  The number of search results and top K similar entries to return.
          crop (bool):  Enable or disable cropping of images (CLIP was trained with cropping, SigLIP was not)
        """
        super().__init__(inputs='text/image', outputs='search', **kwargs)
        
        self.db = nanodb.NanoDB(
            path=path, model=model, 
            dtype=dtype, metric='cosine', 
            reserve=reserve*(1<<20), crop=crop, **kwargs
        )
        
        self.scans = self.db.scans
        self.add_parameters(top_k=top_k)

            
    def process(self, input, add=False, metadata=None, top_k=None, sender=None, **kwargs):
        """
        Search the database for the closest matches to the input.
        
        Parameters:
        
          input (str|PIL.Image) -- either text or an image to search for
                                          
        Returns:
        
          Returns a list of K search results
        """
        if not top_k:
            top_k = self.top_k
        
        if add:
            self.db.add(input, metadata=metadata)
            return
            
        if len(self.db) == 0:
            return
            
        indexes, similarity = self.db.search(input, k=top_k)
        
        results = [dict(index=indexes[n], similarity=float(similarity[n]), metadata=self.db.metadata[indexes[n]])
                   for n in range(top_k) if indexes[n] >= 0]
        
        import pprint
        print('NANODB search results for', input)
        pprint.pprint(results, indent=2)
                   
        self.output(results, query=input)
        
        if not WebServer.Instance:
            return 
            
        html = []
        
        for result in results:
            path = result['metadata']['path']
            for root, mount in WebServer.Instance.mounts.items():
                if root in path:
                    html.append(dict(
                        image=path.replace(root, mount), 
                        similarity=f"{result['similarity']*100:.1f}%",
                        metadata=json.dumps(result['metadata'], indent=2).replace('"', '&quot;')
                    ))

        self.send_state({'search_results': html})
     
    @classmethod
    def type_hints(cls):
        return {
            'dtype': {
                'options': ['float16', 'float32'],
            },
        }
