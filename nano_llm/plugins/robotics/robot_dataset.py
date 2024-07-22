#!/usr/bin/env python3
import time
import logging

from nano_llm import Plugin, DatasetTypes, load_dataset


class RobotDataset(Plugin):
    """
    Set :func:``nano_llm.datasets.load_datasets`` for the various episodic dataset formats this plugin can load.
    """
    def __init__(self, path: str=None, dataset_type: str=None, max_episodes: int=None, max_steps: int=None,
                       framerate: int=10, **kwargs):
        """
        Load various trajectory dataset formats like RLDS/TFDS, Open X-Embodiment, and Robomimic/MimicGen.
        """
        self.dataset = load_dataset(path, dataset_type=dataset_type, max_episodes=max_episodes, max_steps=max_steps)
        
        outputs = ['actions']
        
        for step in self.dataset:
            for i, k in enumerate(step.images):
                outputs.append(f'img_{i}')
            break
      
        super().__init__(outputs=outputs, **kwargs)
        
        self.add_parameters(max_episodes=max_episodes, max_steps=max_steps, framerate=framerate)

    @classmethod
    def type_hints(cls):
        return {
            'dataset_type': dict(options=list(DatasetTypes), display_name='Type'),
        }
          
    @property
    def max_episodes(self):
        return self.dataset.max_episodes
        
    @max_episodes.setter
    def max_episodes(self, value):
        self.dataset.max_episodes = value
     
    @property
    def max_steps(self):
        return self.dataset.max_steps
        
    @max_steps.setter
    def max_steps(self, value):
        self.dataset.max_steps = value

    def run(self):
        time_last = time.perf_counter()
        time_avg = 0
        
        while not self.stop_flag:
            for step in self.dataset:
                if self.stop_flag:
                    return

                timestamp = time.perf_counter()
                time_diff = timestamp - time_last
                time_rate = 1.0 / self.framerate
                time_wait = time_rate - time_diff
                
                if time_wait > 0:
                    time.sleep(time_wait)
                    timestamp = time.perf_counter()  
                
                time_avg = (time_avg * 0.5) + ((timestamp - time_last) * 0.5)
                time_last = timestamp

                action_space = getattr(self.dataset, 'action_space', None)
                
                for i, image in enumerate(step.images):
                    self.output(image, channel=i+1, timestamp=timestamp, action_space=action_space)
  
                self.output(step.action, channel=0, timestamp=timestamp, action_space=action_space)
                self.send_stats(summary=[f"{1.0/time_avg:.1f} FPS"])

