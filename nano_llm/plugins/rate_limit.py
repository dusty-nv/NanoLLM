#!/usr/bin/env python3
import time
import logging

from nano_llm import Plugin


class RateLimit(Plugin):
    """
    Rate limiter plugin with the ability to pause/resume from the queue.
    
      video_limiter = RateLimit(30)  # 30 FPS
      audio_limiter = RateLimit(48000, chunk=4800)  
      
    It can also chunk indexable outputs into smaller amounts of data at a time.
    """
    def __init__(self, rate : float = None, chunk : int = None, 
                       drop : bool = False, on_demand : bool = False, **kwargs):
        """
        Rate limiter plugin with the ability to pause/resume from the queue.
        
        Args:
          rate (float): The number of items per second that can be transmitted (or the playback factor for audio)
          chunk (int): For indexable inputs, the maximum number of items 
                       that can be in each output message.
          drop (bool): If true, only the most recent inputs will be transmitted, with older inputs being dropped.
                       Otherwise, the queue will continue to grow and be throttled to the given rate.
          on_demand (bool): If true, outputs will only be sent when the reciever's input queues
                            are empty and ready for more data.  This will effectively rate limit to the
                            downstream processing speed.
        """
        super().__init__(outputs='items', drop_inputs=drop, **kwargs)
        
        self.paused = -1
        self.tx_rate = 0
        self.last_time = time.perf_counter()

        self.add_parameter('rate', default=rate)
        self.add_parameter('chunk', default=chunk)
        self.add_parameter('drop_inputs', name='Drop', default=drop, kwarg='drop')
        self.add_parameter('on_demand', default=on_demand)
        
    def process(self, input, sample_rate=None, **kwargs):
        """
        First, wait for any pauses that were requested in the output.
        If chunking enabled, chunk the input down until it's gone.
        Then wait as necessary to maintain the requested output rate.
        """
        while True:
            if self.interrupted:
                #logging.debug(f"RateLimit interrupted (input={len(input)})")
                return
            
            pause_duration = self.pause_duration()
            
            if pause_duration > 0:
                #logging.debug(f"RateLimit pausing for {pause_duration} seconds (input={len(input)})")
                time.sleep(pause_duration)
                continue
            
            if self.rate < 16 and sample_rate is not None:
                rate = self.rate * sample_rate
            else:
                rate = self.rate
                       
            if self.chunk is not None and self.chunk > 0:
                #logging.debug(f"RateLimit chunk {len(input)}  {self.chunk}  {time.perf_counter()}")
                if len(input) > self.chunk:
                    self.output(input[:self.chunk], sample_rate=sample_rate, **kwargs)
                    self.update_stats()
                    input = input[self.chunk:]
                    time.sleep(self.chunk/rate*0.95)
                    new=False
                    continue
                else:
                    self.output(input, sample_rate=sample_rate, **kwargs)
                    self.update_stats()
                    time.sleep(len(input)/rate*0.95)
                    return
            else:
                self.output(input, sample_rate=sample_rate, **kwargs)
                self.update_stats()
                if self.rate > 0:
                    time.sleep(1.0/self.rate)
                return
     
    def update_stats(self):
        """
        Calculate and send the throughput statistics when new outputs are transmitted.
        """
        curr_time = time.perf_counter()
        elapsed_time = curr_time - self.last_time
        self.tx_rate = (self.tx_rate * 0.5) + ((1.0 / elapsed_time) * 0.5)
        self.last_time = curr_time
        self.send_stats(
            summary=[f"{self.tx_rate:.1f} tx/sec"],
        )
               
    def pause(self, duration=None, until=None):
        """
        Pause audio playback for `duration` number of seconds, or until the end time.
        
        If `duration` is 0, it will be paused indefinitely until unpaused.
        If `duration` is negative, it will be unpaused.
        
        If already paused, the pause will be extended if it exceeds the current duration.
        """
        current_time = time.perf_counter()
        
        if duration is None and until is None:
            raise ValueError("either 'duration' or 'until' need to be specified")
            
        if duration is not None:
            if duration <= 0:
                self.paused = duration  # disable/infinite pausing
            else:
                until = current_time + duration
         
        if until is not None:
            if until > self.paused and self.paused != 0:
                self.paused = until
                logging.debug(f"RateLimit - pausing output for {until-current_time} seconds")

    def unpause(self):
        """
        Unpause audio playback
        """
        self.pause(-1.0)
        
    def is_paused(self):
        """
        Returns true if playback is currently paused.
        """
        return self.pause_duration() > 0
      
    def pause_duration(self):
        """
        Returns the time to go if still paused (or zero if unpaused)
        """
        if self.paused < 0:
            return 0

        if self.paused == 0:
            return float('inf')
            
        current_time = time.perf_counter()
        
        if current_time >= self.paused:
            self.paused = -1
            return 0
            
        return self.paused - current_time
