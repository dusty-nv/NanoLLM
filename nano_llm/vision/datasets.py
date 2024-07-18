#!/usr/bin/env python3
import os
import json
import time
import glob
import array
import pickle
import logging
import subprocess

import torch
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tdfs

from glob import glob
from pprint import pprint, pformat
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from nano_llm.utils import AttributeDict


__all__ = ['load_dataset', 'DatasetTypes', 'RLDSDataset', 'OpenXDataset', 'BridgeDataset', 'BridgeDatasetRaw']


class TFDSDataset:
    """
    Load a TFDS dataset with support for downloading from HTTP/HTTPS or Google Cloud Storage, 
    and streaming from local cache on disk.  This is inherited by `RLDSDataset` and others to
    implement specific dataset formats and structures with added filtering & pre-processing.
    
      https://github.com/tensorflow/datasets
      https://www.tensorflow.org/datasets/api_docs/python/tfds
      
    TFDS datasets can get quite large (several hundred GB), so check your free disk space first.
    """
    def __init__(self, path, split='train', cache_dir='/data/datasets', **kwargs):
        """
        If path is a URL to an http/https server or Google Cloud Storage Bucket (gs://)
        then the TDFS dataset will first be downloaded and saved to cache_dir.
        """
        tensorflow_disable_device('GPU') # disable GPU memory pool
        
        # make sure the cache dir exists
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # either check local dir from the dataset, or download it
        if os.path.isdir(path):
            self.path = self.find_dataset(path)
        else:
            self.path = self.download(path, cache_dir=cache_dir, **kwargs)

        # load the config
        config_path = os.path.join(self.path, 'dataset_info.json')
        
        if not os.path.isfile(config_path):
            raise IOError(f"TFDSDataset | could not find {config_path}")
         
        with open(config_path) as file:
            self.config = AttributeDict({
                **json.load(file), 
                **dict(name=os.path.basename(path.strip('/')), split=split),
                **kwargs
            })

        # open the dataset (it gets loaded iteratively) 
        self.dataset = tdfs.builder_from_directory(self.path).as_dataset(split=split) #tdfs.load(dataset_name, split=split, data_dir=cache_dir)
        logging.success(f"TFDSDataset | loaded {self.config.name} from {path} (records={len(self.dataset)})")

    @staticmethod
    def download(url, rescan=False, redownload=False, cache_dir='/data/datasets', **kwargs):
        """
        Recursively mirror the remote directory over HTTP/HTTPS or GCS (gs://)
        to the local cache and return the direct path in which the TFDS resides.
        """
        if redownload:
            rescan = True
            
        if not url.endswith('/'):
            url = url + '/'
                
        url_path = urlparse(url).path
        data_dir = os.path.join(cache_dir, os.path.basename(url_path.strip('/')))
        
        if os.path.isdir(data_dir) and not rescan:
            logging.debug(f"TFDSDataset | cached dataset from {url} already found downloaded under {data_dir}  (rescan={rescan}, redownload={redownload})")
            return TFDSDataset.find_dataset(data_dir)
            
        if url.startswith('gs'):
            download_cmd = f"gsutil -m cp -r {'' if redownload else '-n'} {url} {cache_dir}"
        elif url.startswith('http'):
            cut_dirs = len(url_path.split('/')) - 3  # https://www.baeldung.com/linux/wget-mirroring
            download_cmd = f"wget -r --no-parent --no-host-directories --cut-dirs={cut_dirs} --reject='index.html*' --directory-prefix={cache_dir} {'' if redownload else '--no-clobber'} {kwargs.get('wget_flags', '')} {url}"
        else:
            raise ValueError(f'invalid path or URL:  {url}')
            
        logging.info(f"TFDSDataset | downloading from {url} into {cache_dir}\n{download_cmd}")
        subprocess.run(download_cmd, executable='/bin/bash', shell=True, check=True)

        return TFDSDataset.find_dataset(data_dir)

    @staticmethod
    def find_config(path, config='dataset_info.json'):
        """
        Find the local path containing the TFDS dataset config.
        """
        config_files = glob(path.rstrip('/') + f'/**/{config}', recursive=True)
        
        if not config_files:
             raise IOError(f"TFDSDataset | could not find {config} under {path}") 
          
        return config_files[0]
        
    @staticmethod
    def find_dataset(path):
        """
        Find the local path under which the TFDS records reside.
        """
        return os.path.dirname(TFDSDataset.find_config(path))

          
class RLDSDataset(TFDSDataset):
    """
    Load a TDFS dataset in RLDS format - https://github.com/google-research/rlds
    """
    def __init__(self, path, split='train', max_episodes=None, max_steps=None, cache_dir='/data/datasets', **kwargs):
        """
        If path is a URL to an http/https server or Google Cloud Storage Bucket (gs://)
        then the TDFS dataset will first be downloaded and saved to cache_dir.
        """
        if max_episodes:
            split = f"{split}[:{max_episodes}]"
        
        super().__init__(path, split=split, cache_dir=cache_dir, **kwargs)

        step_raw = next(iter(next(iter(self.dataset))['steps']))
        step_img = next(iter(self))

        layout = AttributeDict(
            cameras = len(step_img.images),
            image_size = step_img.images[0].shape,
            step = list(step_raw.keys()),
            action = step_raw['action'],
            observation = AttributeDict()
        )
        
        if isinstance(layout.action, tf.Tensor):
            layout.action = tf.shape(layout.action).numpy().tolist()
         
        for key, value in step_raw['observation'].items():
            if isinstance(value, tf.Tensor):
                if value.dtype == tf.string:
                    value = str
                else:
                    value = value.numpy()
                    value = (value.shape, value.dtype)
            elif hasattr(value, '__len__'):
                value = (type(value), len(value))
            else:
                value = type(value)
                
            layout.observation[key] = value
            
        self.config.update(layout)
        
        self.max_steps = max_steps
        self.num_steps = 0

        for episode in iter(self.dataset):
            self.num_steps += len(episode['steps'])
            
        if self.max_steps:
            self.num_steps = min(self.num_steps, self.max_steps)
        
        logging.success(f"RLDSDataset | loaded {self.config.name} - episode format:\n{pformat(layout, indent=2)}")
    
    def __len__(self):
        """
        Returns the number of timesteps or frames in the dataset, taken over all episodes.
        """
        return self.num_steps
                   
    def __iter__(self):
        """
        Returns an iterator over all steps (or up to max_steps if it was set) with the episodes running back-to-back.  
        `step.is_first` will be set on new episodes, and `set.is_last` will be set at the end of an episode.
        """
        steps = 0
        for episode in iter(self.dataset):
            episode = self.filter_episode(episode)
            if not episode:
                continue
            for step in episode['steps']:
                step = self.filter_step(step)
                if not step:
                    continue
                yield(step)
                steps += 1
                if self.max_steps and steps >= self.max_steps:
                    return
                
    @property
    def episodes(self):
        """
        Returns an iterator over all the episodes, nested over the steps in each episode::
        
            for episode in dataset.episodes():
                for step in episode:
                    ...
        """
        def generator(episode):
            for step in episode['steps']:
                step = self.filter_step(step)
                if step:
                    yield(step)
                
        for episode in iter(self.dataset):
            episode = self.filter_episode(episode)
            if episode:
                yield(generator(episode))
          
    def dump(self, max_steps=1):
        """
        Print out the specified number of steps for inspecting the dataset format.
        """
        for i, step in enumerate(self):
            pprint(step, indent=2)
            if max_steps and i > max_steps:
                break

    def filter_episode(self, episode):
        """
        Override this function to implement custom filtering or transformations on each episode.
        """
        return episode
        
    def filter_step(self, step):
        """
        Apply filtering and data transformations to each step (override this for custom processing)
        """
        data = AttributeDict(
            action=step.get('action'),
            images=[],
            instruction=None,
            is_first=bool(step.get('is_first')),
            is_last=bool(step.get('is_last')),
        )
        
        observation = step['observation']
        image_keys = ['image', 'agentview_rgb']

        for image_key in image_keys:
            for observation_key in observation:
                if image_key in observation_key:
                    data.images.append(observation[observation_key].numpy())

        instruction = observation.get('natural_language_instruction', step.get('language_instruction'))
        
        if instruction is not None:
            data.instruction = instruction.numpy().decode('UTF-8')
         
        if 'state' in observation:
            data.state = observation['state'].numpy()
            
        for key, value in data.items():
            value = self.filter_key(key, value)
            
            if value is None:
                logging.warning(f"RLDSDataset | episode step has missing or empty key: {key}  (skipping)")           
                return None
            else:
                data[key] = value
                        
        return data

    def filter_key(self, key, value):
        """
        Apply filtering to each data entry in the step dict (return None to exclude)
        """
        if value is None:
            return None
        elif isinstance(value, (array.array, np.ndarray)):
            if len(value) == 0:
                return None
        elif isinstance(value, tf.Tensor):
            if len(value) == 0:
                return None
            return value.numpy()
        elif not key.startswith('is_') and not value:
            return None
            
        return value   
     
def tensorflow_disable_device(device):
    """
    TensorFlow gets used for loading TFDS records/tensors, but we don't want it hogging all the GPU memory,
    which it preallocates by default - so just disable it from using GPUs since it's not needed anyways.
    
        https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    """
    devices = tf.config.list_physical_devices(device)
    
    if not devices:
        return

    logging.warning(f"disabling tensorflow device {device} ({len(devices)})")
    tf.config.set_visible_devices([], device)
    logical_devices = tf.config.list_logical_devices(device)
    logging.info(f"tensorflow  Physical {device}: {len(devices)}  Logical {device}: {len(logical_devices)}")


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
    
                         
class OpenXDataset(RLDSDataset):
    """
    Open X-Embodiment (OXE) dataset collection (https://github.com/google-deepmind/open_x_embodiment)
    """
    Names = ['fractal20220817_data', 'kuka', 'bridge', 'taco_play', 'jaco_play', 'berkeley_cable_routing', 'roboturk', 'nyu_door_opening_surprising_effectiveness', 'viola', 'berkeley_autolab_ur5', 'toto', 'language_table', 'columbia_cairlab_pusht_real', 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds', 'nyu_rot_dataset_converted_externally_to_rlds', 'stanford_hydra_dataset_converted_externally_to_rlds', 'austin_buds_dataset_converted_externally_to_rlds', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'maniskill_dataset_converted_externally_to_rlds', 'furniture_bench_dataset_converted_externally_to_rlds', 'cmu_franka_exploration_dataset_converted_externally_to_rlds', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'ucsd_pick_and_place_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'usc_cloth_sim_converted_externally_to_rlds', 'utokyo_pr2_opening_fridge_converted_externally_to_rlds', 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', 'utokyo_saytap_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds', 'utokyo_xarm_bimanual_converted_externally_to_rlds', 'robo_net', 'berkeley_mvp_converted_externally_to_rlds', 'berkeley_rpt_converted_externally_to_rlds', 'kaist_nonprehensile_converted_externally_to_rlds', 'stanford_mask_vit_converted_externally_to_rlds', 'tokyo_u_lsmo_converted_externally_to_rlds', 'dlr_sara_pour_converted_externally_to_rlds', 'dlr_sara_grid_clamp_converted_externally_to_rlds', 'dlr_edan_shared_control_converted_externally_to_rlds', 'asu_table_top_converted_externally_to_rlds', 'stanford_robocook_converted_externally_to_rlds', 'eth_agent_affordances', 'imperialcollege_sawyer_wrist_cam', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'uiuc_d3field', 'utaustin_mutex', 'berkeley_fanuc_manipulation', 'cmu_food_manipulation', 'cmu_play_fusion', 'cmu_stretch', 'berkeley_gnm_recon', 'berkeley_gnm_cory_hall', 'berkeley_gnm_sac_son']

    def __init__(self, name, cache_dir="/data/datasets/open_x_embodiment", **kwargs):
        super().__init__(f"gs://gresearch/robotics/{name}", name=name, cache_dir=cache_dir, **kwargs)

    @staticmethod
    def gcs_path(dataset_name):
        # https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb
        if dataset_name == 'robo_net':
            version = '1.0.0'
        elif dataset_name == 'language_table':
            version = '0.0.1'
        else:
            version = '0.1.0'
        return f'gs://gresearch/robotics/{dataset_name}/{version}'      


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
            action = AttributeDict(
                mean = actions.mean(0).tolist(),
                std = actions.std(0).tolist(),
                max = actions.max(0).tolist(),
                min = actions.min(0).tolist(),
                q01 = np.quantile(actions, 0.01, axis=0).tolist(),
                q99 = np.quantile(actions, 0.99, axis=0).tolist()
            )
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
            

class DroidDataset:
    """
    DROID robot manipulator dataset from https://github.com/droid-dataset/droid
    This is for the raw version with HD video, scanning the Google Cloud bucket 
    for the desired number of episodes instead of downloading all 8.7TB at once.
    """
    BucketRLDS = "gs://gresearch/robotics/droid"
    Bucket100 = "gs://gresearch/robotics/droid_100"
    BucketRaw = "gs://gresearch/robotics/droid_raw/1.0.1"
        
    def __init__(self, bucket=BucketRaw, max_episodes=None, cache_dir="/data/datasets/droid", **kwargs):
        """
        Download the dataset from GCS up to the set number of episodes.
        """
        if not bucket:
            bucket = DroidDataset.BucketRaw
            
        self.bucket = bucket
        self.cache_dir = cache_dir
        self.episodes = {}
        
        os.makedirs(cache_dir, exist_ok=True)

        # recursively list all the metadata files
        metadata_list_path = os.path.join(cache_dir, 'metadata_list.txt')
        
        if not os.path.isfile(metadata_list_path):
            self.run(f"gsutil -m ls {bucket}/**.json | tee {metadata_list_path}")
        
        with open(metadata_list_path) as file:
            metadata_list = [x.strip() for x in file.readlines()[:-3]]  # the last 3 lines are other json files
            
        def find_metadata(name):
            for metadata in metadata_list:
                if name in metadata:
                    return metadata

        # load annotations (language instructions)
        with open(self.download("aggregated-annotations-030724.json")) as file:
            annotations = json.load(file)
            
        # build episode index
        for episode_key in annotations:
            if len(annotations[episode_key]) > 1:
                continue
                
            time_begin = time.perf_counter()
            
            metadata_remote_path = find_metadata(episode_key)
            metadata_local_path = self.download(metadata_remote_path)
            
            if not metadata_local_path:
                logging.warning(f"DroidDataset | could not find metadata for episode {episode_key}  (skipping)")
                continue
                
            with open(metadata_local_path) as file:
                episode = AttributeDict(
                    **json.load(file),
                    path = metadata_local_path,
                    instruction = list(annotations[episode_key].values())[0],
                )

            def download_files():
                for key in episode:
                    if '_path' in key and 'svo' not in key:
                        episode[key] = self.download(os.path.join(
                            os.path.dirname(metadata_remote_path), 
                            '/'.join(episode[key].split('/')[3:])
                        ))
                        
                        if not episode[key]:
                            return False
                            
                return True  
        
            if not download_files():
                continue
                
            self.episodes[episode_key] = episode
            
            if time.perf_counter() - time_begin > 1.0:
                logging.info(f"downloaded {len(self.episodes)} episodes")
            
            if max_episodes and len(self.episodes) >= max_episodes:
                break
            
        logging.success(f"downloaded DROID dataset ({len(self.episodes)} episodes)")
       
    def download(self, filename, use_cache=True):
        if not filename:
            return None
            
        filename = filename.replace(self.bucket, '').strip('/')
        local_path = os.path.join(self.cache_dir, filename).replace(':', '-')
        
        if use_cache and os.path.isfile(local_path):
            return local_path
            
        try:
            self.run(f"gsutil -m cp {'-n' if use_cache else ''} {os.path.join(self.bucket,filename)} {local_path}")
        except Exception as error:
            logging.error(f"error downloading {filename}")
            return None
            
        return local_path
          
    def run(self, cmd, executable='/bin/bash', shell=True, check=True, **kwargs):
        logging.debug(f"DroidDataset | {cmd}")
        return subprocess.run(cmd, executable=executable, shell=shell, check=check, **kwargs)

    
DatasetTypes = {
    'openx': OpenXDataset,
    'droid': DroidDataset,
    'bridge_orig': BridgeDataset,
    'bridge_raw': BridgeDatasetRaw,
}


def load_dataset(dataset: str=None, dataset_type: str=None, **kwargs):
    """
    Dataset factory function that supports different dataset formats and sources.
    """
    if not dataset_type:
        if dataset in DatasetTypes:
            dataset_type = dataset
            dataset = None
        else:    
            dataset_type = 'openx'
    
    if dataset:    
        dataset = DatasetTypes[dataset_type](dataset, **kwargs)
    else:
        dataset = DatasetTypes[dataset_type](**kwargs)
        
    dataset.type = dataset_type
    return dataset
        
                
if __name__ == "__main__":

    from nano_llm.utils import ArgParser
    
    # parse args and set some defaults
    parser = ArgParser(extras=['log'])
    
    parser.add_argument("--dataset", type=str, default=None, required=True, help=f"path or name of the dataset to load")
    parser.add_argument("--dataset-type", type=str, default=None, choices=list(DatasetTypes.keys()), help=f"type of the dataset to load")
    parser.add_argument("--max-episodes", type=int, default=None, help="the maximum number of episodes from the dataset to process")
    parser.add_argument("--max-steps", type=int, default=None, help="the maximum number of frames to process across all episodes")
    parser.add_argument("--rescan", action='store_true', help="rescan the dataset files for changes or rebuild the index")
    
    args = parser.parse_args()

    dataset = load_dataset(**vars(args))
    
    dataset.dump(max_steps=args.max_steps)
     
    logging.debug(f"{dataset.__class__.__name__} | done inspecting {dataset.config.name}")
    
