#!/usr/bin/env python3
import os
import cv2

import tensorflow_datasets as tfds
import numpy as np


DATASET = os.environ.get('_DATASET')
DATASET_TYPE = os.environ.get('_DATASET_TYPE')
        
MAX_EPISODES = int(os.environ.get('_MAX_EPISODES', 0))
MAX_STEPS = int(os.environ.get('_MAX_STEPS', 0))
   
OUTPUT = str(os.environ.get('_OUTPUT'))
WIDTH = int(os.environ.get('_WIDTH', 0))
HEIGHT = int(os.environ.get('_HEIGHT', 0))                
             
SAMPLE_STEPS = int(os.environ.get('_SAMPLE_STEPS', 0))
SAMPLE_ACTIONS = int(os.environ.get('_SAMPLE_ACTIONS', 0))

DOF = int(os.environ.get('_DOF', 7))

         
class RLDSDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """
    DatasetBuilder for RLDS/TFDS format
    """
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(HEIGHT, WIDTH, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(HEIGHT, WIDTH, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        
                        'state': tfds.features.Tensor(
                            shape=(DOF,),
                            dtype=np.float32,
                            doc='Robot state',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(DOF,),
                        dtype=np.float32,
                        doc='Robot action',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self):
        """Generator of examples for each split."""
        from nano_llm import load_dataset
        
        self.dataset = load_dataset(
            DATASET, 
            dataset_type=DATASET_TYPE, 
            max_episodes=MAX_EPISODES,
            max_steps=MAX_STEPS,
        )
        
        self.output = OUTPUT

        self.width = WIDTH
        self.height = HEIGHT
        
        step = next(iter(self.dataset))
        
        if not self.width or not self.height:
            size = step.images[0].shape
            self.width = size[-2]
            self.height = size[-3]
          
        self.dof = len(step.action)
        
        self.sample_steps = SAMPLE_STEPS
        self.sample_actions = SAMPLE_ACTIONS
        
        episode = []        
        episodes = 0
        
        for i, step in enumerate(self.dataset):
            obs = {}
            
            if 'state' in step:
                obs['state'] = step.state
            else:
                obs['state'] = [0] * DOF
                    
            if len(step.images) > 0:
                obs['image'] = self.resize_image(step.images[0])
            
            if len(step.images) > 1:
                obs['wrist_image'] = self.resize_image(step.images[1])
                   
            episode.append({
                'observation': obs,
                'action': step.action.astype(np.float32),
                'is_first': step.is_first,
                'is_last': step.is_last,
                'language_instruction': step.instruction
            })

            if step.is_last:
                if self.sample_actions:
                    for n in range(len(episode)):
                        for m in range(n+1, min(n+sample_actions, len(episode))):
                            episode[n]['action'][:-1] += episode[m]['action'][:-1]
                 
                if self.sample_steps:
                    episode = [episode[n] for n in range(0, len(episode), 2)]
                    episode[-1]['is_last'] = True
                    
                #if i % 25:
                #    print(f"processed step {i}")
                               
                yield f"episode_{episodes}", {
                    'steps': episode,
                    'episode_metadata': {
                        'file_path': 'unknown',
                     }
                }
                
                episode = []
                episodes += 1
                
    def resize_image(self, image):
        img_width = image.shape[-2]
        img_height = image.shape[-3]
        
        if self.width == img_width and self.height == img_height:
            return image

        if self.width < img_width and self.height < img_height:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        
        return cv2.resize(image, (self.height, self.width), interpolation=interpolation)
        
