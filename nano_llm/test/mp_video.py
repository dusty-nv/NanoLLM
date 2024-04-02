#!/usr/bin/env python3
import logging

from nano_llm import Agent

from nano_llm.plugins import VideoSource, VideoOutput, ProcessProxy
from nano_llm.utils import ArgParser


class MultiprocessVideo(Agent):
    """
    Test of running a video stream across processes.
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.video_source = VideoSource(return_tensors='np', **kwargs)
        self.video_output = ProcessProxy((lambda **kwargs: VideoOutput(**kwargs)), **kwargs)
        
        self.video_source.add(self.on_video, threaded=False)
        self.video_source.add(self.video_output)
        
        self.pipeline = [self.video_source]
        
    def on_video(self, image):
        logging.debug(f"captured {image.shape} ({image.dtype}) frame from {self.video_source.resource}")

         
if __name__ == "__main__":
    from nano_llm.utils import ArgParser

    parser = ArgParser(extras=['video_input', 'video_output', 'log'])
    args = parser.parse_args()
    
    agent = MultiprocessVideo(**vars(args)).run()