#!/usr/bin/env python3
import logging

from nano_llm import Agent

from nano_llm.plugins import VideoSource, VideoOutput
from nano_llm.utils import ArgParser


class VideoStream(Agent):
    """
    Relay, view, or test a video stream.  Use the ``--video-input`` and ``--video-output`` arguments
    to set the video source and output protocols used from `jetson_utils <https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md>`_
    like V4L2, CSI, RTP/RTSP, WebRTC, or static video files.

    For example, this will capture a V4L2 camera and serve it via WebRTC with H.264 encoding:
    
    .. code-block:: text
    
        python3 -m nano_llm.agents.video_stream \ 
           --video-input /dev/video0 \ 
           --video-output webrtc://@:8554/output
    
    It's also used as a basic test of video streaming before using more complex agents that rely on it.
    """
    def __init__(self, video_input=None, video_output=None, **kwargs):
        """
        Args:
          video_input (Plugin|str): the VideoSource plugin instance, or URL of the video stream or camera device.
          video_output (Plugin|str): the VideoOutput plugin instance, or output stream URL / device ID.
        """
        super().__init__()

        self.video_source = VideoSource(video_input, **kwargs)
        self.video_output = VideoOutput(video_output, **kwargs)
        
        self.video_source.add(self.on_video, threaded=False)
        self.video_source.add(self.video_output)
        
        self.pipeline = [self.video_source]
        
    def on_video(self, image):
        logging.debug(f"captured {image.width}x{image.height} frame from {self.video_source.resource}")

         
if __name__ == "__main__":
    parser = ArgParser(extras=['video_input', 'video_output', 'log'])
    args = parser.parse_args()
    
    agent = VideoStream(**vars(args)).run() 