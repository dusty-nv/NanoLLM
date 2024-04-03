#!/usr/bin/env python3
import os
import time
import json
import torch
import pprint
import logging
import threading

from datetime import datetime
from termcolor import cprint

from nano_llm import Agent, StopTokens

from nano_llm.web import WebServer
from nano_llm.plugins import VideoSource, VideoOutput, ChatQuery, PrintStream, ProcessProxy, EventFilter, NanoDB
from nano_llm.utils import ArgParser, print_table, wrap_text

from jetson_utils import cudaFont, cudaMemcpy, cudaToNumpy, cudaDeviceSynchronize, saveImage


class VideoQuery(Agent):
    """
    Closed-loop visual agent that repeatedly applies a set of prompts to a video stream, and is also able to match
    the incoming stream against a vector database, and then use the matching metadata for multimodal RAG.
    Also serves an interactive web UI for the user to change queries, set event filters, and tag images in the database.
    """
    def __init__(self, model="liuhaotian/llava-v1.5-13b", nanodb=None, vision_scaling='resize', **kwargs):
        """
        Args:
        
          model (NanoLLM|str): the NanoLLM multimodal model instance, or name/path of a multimodal model to load.
          nanodb (NanoDB|str): optional NanoDB plugin instance (or path to a NanoDB on disk) to match the incoming stream against.
          vision_scaling (str): ``'resize'`` to ignore aspect ratio when downscaling to the often-square resolution of the vision encoder,
                                or ``crop`` to center-crop the images first to maintain aspect ratio (while discarding pixels).
          kwargs:  forwarded to the plugin initializers for ChatQuery, VideoSource, and VideoOutput
        """                    
        super().__init__()

        if not vision_scaling:
            vision_scaling = 'resize'
            
        #: The model plugin (ChatQuery)
        self.llm = ProcessProxy('ChatQuery', model=model, drop_inputs=True, vision_scaling=vision_scaling, **kwargs) #ProcessProxy((lambda **kwargs: ChatQuery(model, drop_inputs=True, **kwargs)), **kwargs)
        self.llm.add(PrintStream(color='green', relay=True).add(self.on_text))
        self.llm.start()

        # test / warm-up query
        self.warmup = True
        self.text = ""
        self.eos = False
        
        self.llm("What is 2+2?")
        
        while self.warmup:
            time.sleep(0.25)
            
        # create video streams    
        self.video_source = VideoSource(**kwargs)  #: The video source plugin
        self.video_output = VideoOutput(**kwargs)  #: The video output plugin
        
        self.video_source.add(self.on_video, threaded=False)
        self.video_output.start()
        
        self.font = cudaFont()
        
        self.pause_video = False
        self.pause_image = None
        self.last_image = None
        self.tag_image = None

        self.pipeline = [self.video_source]
        
        # setup prompts
        self.prompt_history = kwargs.get('prompt')
        
        if not self.prompt_history:
            self.prompt_history = [
                'Describe the image concisely.', 
                'How many fingers is the person holding up?',
                'What does the text in the image say?',
                'There is a question asked in the image.  What is the answer?',
                'Is there a person in the image?  Answer yes or no.',
            ]
        
        self.prompt = self.prompt_history[0]
        
        self.last_prompt = None
        self.auto_refresh = True
        self.auto_refresh_db = True
        
        self.rag_threshold = 1.0
        self.rag_prompt = None
        self.rag_prompt_last = None
        
        self.keyboard_prompt = 0
        self.keyboard_thread = threading.Thread(target=self.poll_keyboard)
        self.keyboard_thread.start()

        #: The `NanoDB <https://github.com/dusty-nv/jetson-containers/tree/master/packages/vectordb/nanodb>`_ vector database 
        if nanodb:
            self.db = NanoDB(
                path=nanodb, 
                model=None, # disable DB's model because VLM's CLIP is used 
                reserve=kwargs.get('nanodb_reserve'), 
                k=18, drop_inputs=True,
            ).start().add(self.on_search)
            self.llm.add(self.on_image_embedding, channel=ChatQuery.OutputImageEmbedding)
        else:
            self.db = None
            
        # webserver
        mounts = {
            scan : f"/images/{n}" 
            for n, scan in enumerate(self.db.scans)
        } if self.db else {}
        
        mounts['/data/datasets/uploads'] = '/images/uploads'
        
        video_source = self.video_source.stream.GetOptions()['resource']
        video_output = self.video_output.stream.GetOptions()['resource']
        
        webrtc_args = {}
        
        if video_source['protocol'] == 'webrtc':
            webrtc_args.update(dict(webrtc_input_stream=video_source['path'].strip('/'), 
                                    webrtc_input_port=video_source['port'],
                                    send_webrtc=True))
        else:
            webrtc_args.update(dict(webrtc_input_stream='input', 
                                    webrtc_input_port=8554,
                                    send_webrtc=False))
        
        if video_output['protocol'] == 'webrtc':
            webrtc_args.update(dict(webrtc_output_stream=video_output['path'].strip('/'), 
                                    webrtc_output_port=video_output['port']))
        else:
            webrtc_args.update(dict(webrtc_output_stream='output', 
                                    webrtc_output_port=8554))

        web_title = kwargs.get('web_title')
        web_title = web_title if web_title else 'LIVE LLAVA'
        
        #: the webserver (by default on ``https://localhost:8050``)
        self.server = WebServer(
            msg_callback=self.on_websocket, 
            index='video_query.html', 
            title=web_title, 
            model=os.path.basename(model),
            mounts=mounts,
            nanodb=nanodb,
            **webrtc_args,
            **kwargs
        )
        
        #: event filters for parsing bot output and triggering actions when conditions are met.
        self.events = EventFilter(server=self.server)
   
    def on_video(self, image):
        """
        When a new frame is recieved from the video source, run the model on it with the set prompt,
        applying RAG using the metadata from the most-similar match from the vector database (if enabled).
        Then render the latest text from the model over the image, and send it to the output video stream.
        """
        if self.pause_video:
            if not self.pause_image:
                self.pause_image = cudaMemcpy(image)
            image = cudaMemcpy(self.pause_image)
        
        if self.auto_refresh or self.prompt != self.last_prompt or self.rag_prompt != self.rag_prompt_last:
            np_image = cudaToNumpy(image)
            cudaDeviceSynchronize()
            
            if self.rag_prompt:
                prompt = self.rag_prompt + '. ' + self.prompt
            else:
                prompt = self.prompt
                
            self.llm(['/reset', np_image, prompt])
            
            self.last_prompt = self.prompt
            self.rag_prompt_last = self.rag_prompt
            
            if self.db:
                self.last_image = cudaMemcpy(image)

        # draw text overlays
        text = self.text.replace('\n', '').replace('</s>', '').strip()
        y = 5
        
        if self.rag_prompt:
            y = wrap_text(self.font, image, text='RAG: ' + self.rag_prompt, x=5, y=y, color=(255,172,28), background=self.font.Gray40)
            
        y = wrap_text(self.font, image, text=self.prompt, x=5, y=y, color=(120,215,21), background=self.font.Gray40)

        if text:
            y = wrap_text(self.font, image, text=text, x=5, y=y, color=self.font.White, background=self.font.Gray40)
        
        self.video_output(image)
   
    def on_text(self, text):
        """
        When new output is recieved from the model, update the text to render,
        and check if it satisfied any of the event filters when the output is complete.
        """
        if self.eos:
            self.text = text  # new query response
            self.eos = False
        elif not self.warmup:  # don't view warmup response
            self.text = self.text + text

        if text.endswith(tuple(StopTokens + ['###'])):
            self.print_stats()
            
            if not self.warmup:
                self.events(self.text, prompt=self.prompt)
                
            self.warmup = False
            self.eos = True

    def on_image_embedding(self, embedding):
        """
        Recieve the image embedding from CLIP that was used when the model processed
        the last image, and search it against the database to find the most similar
        images and their metadata for RAG.  Also, if the user requested the last image
        be tagged, add the embedding to the vector database along with the metadata tags.
        """
        if self.tag_image and self.last_image:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"/data/datasets/uploads/{timestamp}.jpg"
            metadata = dict(path=filename, time=timestamp, tags=self.tag_image)
            self.tag_image = None 
            
            def save_image(filename, image, embedding, metadata):
                saveImage(filename, image)
                self.db(embedding, add=True, metadata=metadata)
                logging.info(f"added incoming image to database with tags '{self.tag_image}' ({filename})")
        
            threading.Thread(target=save_image, args=(filename, self.last_image, embedding, metadata)).start()
            
        if self.auto_refresh_db:
            self.db(embedding)
        
    def on_search(self, results):
        """
        Recieve the similar matches from the vector database and update RAG with them,
        along with the most recent results shown in the web UI.
        """
        html = []
        
        for result in results:
            path = result['metadata']['path']
            for root, mount in self.server.mounts.items():
                if root in path:
                    html.append(dict(
                        image=path.replace(root, mount), 
                        similarity=f"{result['similarity']*100:.1f}%",
                        metadata=json.dumps(result['metadata'], indent=2).replace('"', '&quot;')
                    ))
                    
        if html:
            self.server.send_message({'search_results': html})
         
        if len(results) >= 3:
            cprint(f"nanodb search results (top 3)\n{pprint.pformat(results[:3], indent=2)}", color='blue')
        
        # RAG
        self.rag_prompt = None
        
        if len(results) == 0:    
            return
            
        result = results[0]

        if result['similarity'] > self.rag_threshold and 'tags' in result['metadata']:
            self.rag_prompt = f"This image is of {result['metadata']['tags']}"
        
    def on_websocket(self, msg, msg_type=0, metadata='', **kwargs):
        """
        Websocket message handler from the client.
        """
        if msg_type == WebServer.MESSAGE_JSON:
            #print(f'\n\n###############\n# WEBSOCKET JSON MESSAGE\n#############\n{msg}')
            if 'prompt' in msg:
                self.prompt = msg['prompt']
                if self.prompt not in self.prompt_history:
                    self.prompt_history.append(self.prompt)
            elif 'pause_video' in msg:
                self.pause_video = msg['pause_video']
                self.pause_image = None
                logging.info(f"{'pausing' if self.pause_video else 'resuming'} processing of incoming video stream")
            elif 'auto_refresh' in msg:
                self.auto_refresh = msg['auto_refresh']
                logging.info(f"{'enabling' if self.auto_refresh else 'disabling'} auto-refresh of model output with prior query")
            elif 'auto_refresh_db' in msg:
                self.auto_refresh_db = msg['auto_refresh_db']
                logging.info(f"{'enabling' if self.auto_refresh_db else 'disabling'} auto-refresh of vector database search results")
            elif 'save_db' in msg:
                if self.db:
                    self.db.db.save()
            elif 'tag_image' in msg:
                self.tag_image = msg['tag_image']
            elif 'vision_scaling' in msg:
                self.llm(vision_scaling=msg['vision_scaling'])
            elif 'max_new_tokens' in msg:
                self.llm(max_new_tokens=int(msg['max_new_tokens']))
            elif 'rag_threshold' in msg:
                self.rag_threshold = float(msg['rag_threshold']) / 100.0
                logging.debug(f"set RAG threshold to {self.rag_threshold}")
                
    def poll_keyboard(self):
        while True:
            try:
                key = input().strip() #getch.getch()
                
                if key == 'd' or key == 'l':
                    self.keyboard_prompt = (self.keyboard_prompt + 1) % len(self.prompt_history)
                    self.prompt = self.prompt_history[self.keyboard_prompt]
                elif key == 'a' or key == 'j':
                    self.keyboard_prompt = self.keyboard_prompt - 1
                    if self.keyboard_prompt < 0:
                        self.keyboard_prompt = len(self.prompt_history) - 1
                    self.prompt = self.prompt_history[self.keyboard_prompt]
                    
                num = int(key)
                
                if num > 0 and num <= len(self.prompt_history):
                    self.keyboard_prompt = num - 1
                    self.prompt = self.prompt_history[self.keyboard_prompt]
                    
            except Exception as err:
                continue
     
    def print_stats(self):
        #print_table(self.llm.model.stats)
        curr_time = time.perf_counter()
            
        if not hasattr(self, 'start_time'):
            self.start_time = curr_time
        else:
            frame_time = curr_time - self.start_time
            self.start_time = curr_time
            refresh_str = f"{1.0 / frame_time:.2f} FPS ({frame_time*1000:.1f} ms)"
            self.server.send_message({'refresh_rate': refresh_str})
            logging.info(f"refresh rate:  {refresh_str}")

    def start(self):
        """
        Start the webserver & websocket listening in other threads.
        """
        super().start()
        self.server.start()
        return self
        
if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['video_input', 'video_output', 'web', 'nanodb'])
    args = parser.parse_args()
    agent = VideoQuery(**vars(args)).run() 
    