#!/usr/bin/env python3
import os
import json
import pickle
import logging
import numpy as np

from glob import glob
from pprint import pprint, pformat
from pathlib import Path
from datetime import datetime
from nano_llm.utils import AttributeDict

from .rlds import RLDSDataset


class BridgeDataset(RLDSDataset):
    """
    BridgeData V2 robot dataset (https://github.com/rail-berkeley/bridge_data_v2)
    This is for the original version in RLDS/TFDS format with 224x224 images from:
    
      https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset
      
    Although Bridge is included in OXE, models such as OpenVLA and Octo use this original.
    """
    def __init__(self, cache_dir="/data/datasets/bridge", **kwargs):
        super().__init__(
            "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds", 
            name='bridge_orig', cache_dir=cache_dir, **kwargs
        )
    
    def filter_episode(self, data):
        """
        Relabels actions to use reached proprioceptive state; discards first & last timestep
        https://github.com/openvla/openvla/blob/317dbd9e41edaa83ddf2b33f5412b5e7ddadcd4e/prismatic/vla/datasets/rlds/utils/data_utils.py#L166
        """
        episode = AttributeDict(steps=[])
        
        for step in data['steps']:
            step = super().filter_step(step)
            
            if not step:
                logging.warning(f"RLDSDataset | {self.config.name} had missing step data (skipping episode)")
                return
                
            episode.steps.append(step)
        
        for i, step in enumerate(episode.steps):
            if i >= len(episode.steps) - 1:
                break
            step.action[:6] = episode.steps[i+1].state[:6] - step.state[:6]
            
        episode.steps = episode.steps[1:-1]
        
        return episode
            
    def filter_step(self, step):
        """
        The steps were already converted from TFDS format above, so just return them.
        """
        return step
       
        
class BridgeDatasetRaw:
    """
    BridgeData V2 robot dataset (https://github.com/rail-berkeley/bridge_data_v2)
    This is for the raw dataset with 640x480 images, extracted from one of:
    
      https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip (411 GB)
      https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_6_18.zip (30 GB)
      
    https://github.com/kvablack/dlimp/blob/main/rlds_converters/bridge_dataset/bridge_dataset_dataset_builder.py
    """
    def __init__(self, path, max_episodes=None, max_steps=None, rescan=False, **kwargs):
        """
        Scan the path for directories containing episodes of images, trajectories, and action data.
        """
        self.root = path
        self.config = AttributeDict(name='bridge_raw')
        
        self.num_steps = 0
        self.max_steps = max_steps
        
        index_path = os.path.join(path, 'index.pkl')
        
        if not rescan:
            self.episodes, self.stats = self.load_index(index_path)

        if rescan or not self.episodes:
            self.episodes = self.scan(path, max_episodes=max_episodes, **kwargs)
            self.stats = self.compute_stats(self.episodes)
            logging.info(f"BridgeDataset | saving {index_path}")
            with open(index_path, 'wb') as file:
                pickle.dump((self.episodes, self.stats), file, protocol=pickle.HIGHEST_PROTOCOL)

        if max_episodes and len(self.episodes) > max_episodes:
            self.episodes = self.episodes[:max_episodes]
            
        for episode in self.episodes:
            self.num_steps += len(episode.steps)

        if self.max_steps:
            self.num_steps = min(self.num_steps, self.max_steps)
        
        logging.success(f"BridgeDataset | loaded index of {len(self.episodes)} episodes, {self.num_steps} images from {self.root}")       

    def __len__(self):
        """
        Returns the number of timesteps or frames in the dataset, over all the episodes.
        """
        return self.num_steps
        
    def __iter__(self):
        """
        Returns an iterator over all steps (or up to max_steps if it was set) with the episodes running back-to-back.  
        `step.is_first` will be set on new episodes, and `set.is_last` will be set at the end of an episode.
        """
        steps = 0
        for episode in self.episodes:
            for step in episode.steps:
                #step.metadata = episode.metadata
                yield(step)
                steps += 1
                if self.max_steps and steps >= self.max_steps:
                    return

    def scan(self, path, max_episodes=None, **kwargs):
        """
        Search for episodes and build the metadata index.
        """
        paths = sorted(glob(os.path.join(self.root, *("*" * 4))))
        episodes = []
        num_images = 0
        
        for path in paths:
            if not os.path.isdir(path):
                logging.warning(f"BridgeDataset | {path} was not found or not a directory  (skipping)")
                continue
                
            for dated_folder in os.listdir(path):
                if "lmdb" in dated_folder:
                    continue

                search_path = os.path.join(
                    path, dated_folder, "raw", "traj_group*", "traj*"
                )
                
                all_traj = glob(search_path)
                
                if not all_traj:
                    logging.debug(f"BridgeDataset | no trajectories found in {search_path}  (skipping)")
                    continue

                '''
                config_path = os.path.join(path, dated_folder, "config.json")
                config = {}
                
                if os.path.exists(config_path):
                    with open(config_path, "rb") as f:
                        config = json.load(f)
                '''
                
                for traj_path in all_traj:
                    episode = self.load_episode(traj_path)
                    
                    if not episode:
                        continue

                    episodes += [episode]
                    num_images += len(episode.steps)

                    if len(episodes) % 250 == 0:
                        logging.info(f"BridgeDataset | scanned {len(episodes)} episodes, {num_images} images from {self.root}")
                        
                    if max_episodes and len(episodes) >= max_episodes:
                        return episodes 
            
        if not episodes:
            raise RuntimeError(f"BridgeDataset | did not find any episodes under {self.root}")

        return episodes

    def compute_stats(self, episodes):
        logging.debug(f"BridgeDataset | gathering dataset for stats computation ({len(self.episodes)} episodes)")
        
        actions = []
        
        for episode in episodes:
            for step in episode.steps:
                actions.append(step.action)

        actions = np.asarray(actions)
        
        logging.debug(f"BridgeDataset | computing dataset statistics ({len(self.episodes)} episodes, {len(actions)} steps)")

        return AttributeDict(
            mean = actions.mean(0).tolist(),
            std = actions.std(0).tolist(),
            max = actions.max(0).tolist(),
            min = actions.min(0).tolist(),
            q01 = np.quantile(actions, 0.01, axis=0).tolist(),
            q99 = np.quantile(actions, 0.99, axis=0).tolist()
        )
        
    def load_episode(self, path):
        date_time = datetime.strptime(path.split("/")[-4], "%Y-%m-%d_%H-%M-%S")

        if date_time < datetime(2021, 7, 23): # episodes before then needed latency corrections
            logging.debug(f"BridgeDataset | skipping episode from {date_time}  {path}")
            return
            
        image_dirs = sorted(glob(os.path.join(path, 'images*')))
        images = []
        
        for image_dir in image_dirs:
            image_files = sorted(
                glob(os.path.join(image_dir, 'im_*.jpg')),
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )

            if image_files:   
                images.append(image_files)

        if not images:
            logging.debug(f"BridgeDataset | episode {path} did not contain any images (skipping)")
            return
         
        if not all(len(images[i]) == len(images[0]) for i in range(len(images))):
            logging.debug(f"BridgeDataset | episode {path} had {len(images)} sets of images with different number of frames {[len(x) for x in images]}")
            return
                                 
        state = self.load_state(path)
        actions = self.load_actions(path)
        metadata = self.load_metadata(path)
        instruction = self.load_lang(path)

        if not instruction:
            logging.debug(f"BridgeDataset | episode {path} was missing 'instruction' annotation (skipping)")
            return
        
        if not metadata:
            logging.debug(f"BridgeDataset | episode {path} was missing metadata (skipping)")
            return
            
        steps = len(images[0])
        lengths = dict(images=steps, state=len(state), action=len(actions)+1)

        if not all(x == steps for x in lengths.values()):
            logging.debug(f"BridgeDataset | episode {path} did not have matching number of observations {lengths}")
            return

        # https://github.com/openvla/openvla/blob/317dbd9e41edaa83ddf2b33f5412b5e7ddadcd4e/prismatic/vla/datasets/rlds/utils/data_utils.py#L166
        arm_actions = state[1:, :6] - state[:-1, :6]
        gripper_actions = binarize_gripper_actions(actions[:, -1:])
        actions = np.concatenate((arm_actions, gripper_actions), axis=1)
        
        # drop the first and last frames from the episode
        episode = AttributeDict(path=path, metadata=metadata, steps=[])
            
        for i in range(1, steps-1):
            episode.steps.append(AttributeDict(                   
                state = state[i],
                action = actions[i] if i < len(actions) else None,
                images = [x[i] for x in images],
                instruction = instruction,
                is_first = False,
                is_last = False,
            ))
        
        episode.steps[0].is_first = True
        episode.steps[-1].is_last = True
        
        return episode
                        
    def load_index(self, path):
        if not os.path.isfile(path):
            logging.debug(f"BridgeDataset | missing {path} - scanning index")
            return [], {}
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as error:
            logging.error(error)
   
    def load_state(self, path):
        fp = os.path.join(path, "obs_dict.pkl")
        if not os.path.isfile(fp):
            logging.debug(f"BridgeDataset | missing {fp}")
            return []
        with open(fp, "rb") as f:
            x = pickle.load(f)
        #pprint(x, indent=2)
        #print('loaded state', fp, list(x.keys()))
        return np.asarray(x["full_state"])

    def load_actions(self, path):
        fp = os.path.join(path, "policy_out.pkl")
        if not os.path.isfile(fp):
            logging.debug(f"BridgeDataset | missing {fp}")
            return []
        with open(fp, "rb") as f:
            act_list = pickle.load(f)
        #pprint(act_list, indent=2)
        if isinstance(act_list[0], dict):
            act_list = [x["actions"] for x in act_list]
        return np.asarray(act_list)

    def load_lang(self, path):
        fp = os.path.join(path, "lang.txt")
        if not os.path.isfile(fp):
            #logging.debug(f"BridgeDataset | missing {fp}")
            return []
        with open(fp, "r") as f:
            text = f.readline().strip()
        return text

    def load_metadata(self, path):
        fp = os.path.join(Path(path).parents[2], "collection_metadata.json")
        if not os.path.isfile(fp):
            logging.debug(f"BridgeDataset | missing {fp}")
            return {}
        with open(fp, "r") as f:
            return json.load(f)
 
    def dump(self, max_steps=1):
        print(f"{self.config.name} dataset stats:\n\n{pformat(self.stats, indent=2)}\n")
        steps = 0
        
        for e, episode in enumerate(self.episodes):
            print(f"Episode {e} of {len(self.episodes)}  {episode.path}\n\n{pformat(episode.metadata, indent=2)}\n")
            for s, step in enumerate(episode.steps):
                print(f"Step {s} of {len(episode.steps)}  (episode {e} of {len(self.episodes)})\n\n{pformat(step, indent=2)}\n")
                steps += 1
                if max_steps and steps >= max_steps:
                    return


def binarize_gripper_actions(actions):
    """
    For the gripper, near 0.0 is closed and 1.0 open. Convert intermediate values based on the state reached.
    This uses a numpy array of all the gripper action states in the episode, and returns the binary version.
    
    https://github.com/octo-models/octo/blob/5eaa5c6960398925ae6f52ed072d843d2f3ecb0b/octo/data/utils/data_utils.py#L296
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    in_between_mask = np.logical_not(np.logical_or(open_mask, closed_mask))

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
     
    return new_actions

