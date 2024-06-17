#!/usr/bin/env python3
import time
import logging

from nano_llm import Plugin
from nano_llm.utils import cuda_image

from jetson_utils import videoOutput


class VideoOutput(Plugin):
    """
    Saves images to a compressed video or directory of individual images, the display, or a network stream.
    https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
    """
    def __init__(self, video_output : str = "webrtc://@:8554/output", 
                 video_output_codec : str = None, video_output_bitrate : int = None, 
                 video_output_save : str = None, **kwargs):
        """
        Output video to a network stream (RTP/RTSP/WebRTC), video file, or display.
        
        Args:
          video_output (str): Stream URL, path to video file, directory of images.
          video_output_codec (str): Force a particular codec ('h264', 'h265', 'vp8', 'vp9', 'mjpeg', ect)
          video_output_bitrate (int): The desired bitrate in bits per second (default is 4 Mbps)
        """
        super().__init__(outputs=0, **kwargs)
        
        options = {}

        if video_output_codec:
            options['codec'] = video_output_codec
            
        if video_output_bitrate:
            options['bitrate'] = video_output_bitrate

        if video_output_save:
            options['save'] = video_output_save
            
        if video_output is None:
            video_output = ''
            
        args = None if 'display://' in video_output else ['--headless']
        
        self.stream = videoOutput(video_output, options=options, argv=args)
        self.resource = video_output
        self.time_last = time.perf_counter()
        self.framerate = 0
        
    def process(self, image, **kwargs):
        """
        Input should be a jetson_utils.cudaImage, np.ndarray, torch.Tensor, or have __cuda_array_interface__
        """
        image = cuda_image(image)
        shape = image.shape
        
        self.stream.Render(image, stream=image.stream)
        
        curr_time = time.perf_counter()
        self.framerate = self.framerate * 0.9 + (1.0 / (curr_time - self.time_last)) * 0.1
        self.time_last = curr_time
        self.send_stats(summary=[f"{shape[1]}x{shape[0]}", f"{self.framerate:.1f} FPS"])
            
