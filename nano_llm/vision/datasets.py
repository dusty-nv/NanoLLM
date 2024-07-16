#!/usr/bin/env python3
import os
import json
import time
import glob
import pickle
import pprint
import logging
import subprocess

import torch
import numpy as np

from nano_llm.utils import AttributeDict


__all__ = ['load_dataset', 'DatasetTypes', 'RLDSDataset', 'OpenXDataset', 'BridgeDataset', 'BridgeDatasetRaw']


class RLDSDataset:
    """
    Load a TDFS dataset in RLDS format - https://github.com/google-research/rlds
    """
    def __init__(self, path, split='train', max_episodes=None, max_steps=None, cache_dir='/data/datasets', **kwargs):
        """
        If path is a URL to an http/https server or Google Cloud Storage Bucket (gs://)
        then the TDFS dataset will first be downloaded and saved to cache_dir.
        """
        import tensorflow_datasets as tdfs

        # tensorflow gets used for loading tfrecords/tensors, but we don't want it hogging all the GPU memory,
        # which it preallocates by default - so just disable it from using GPUs since it's not needed anyways.
        tensorflow_disable_device('GPU')
        
        # make sure the cache dir exists
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # download the dataset from GCS or HTTP/HTTPS
        if not os.path.isdir(path):
            self.download(path, cache_dir=cache_dir, **kwargs)

        if max_episodes:
            split = f"{split}[:{max_episodes}]"
         
        # open the dataset (it gets loaded iteratively) 
        self.name = kwargs.get(name, os.path.basename(url))
        self.dataset = tdfs.load(name, split=split, data_dir=cache_dir)

        self.num_episodes = 0
        self.num_steps = 0
        self.max_steps = max_steps
        
        for episode in iter(self.dataset):
            self.num_steps += len(episode['steps'])
            self.num_episodes += 1

        if self.max_steps:
            self.num_steps = min(self.num_steps, self.max_steps)

        self.image_dim = next(iter(self)).images[0].shape

        logging.success(f"RLDSDataset | loaded {self.name} from {path}  (episodes={self.num_episodes}, steps={self.num_steps}, image={self.image_dim})")

    def __len__(self):
        return self.num_steps
          
    def __iter__(self):
        steps = 0
        for episode in iter(self.dataset):
            for step in episode['steps']:
                step_dict = self.to_dict(step)
                if not step_dict:
                    continue
                yield(step_dict)
                steps += 1
                if self.max_steps and steps >= self.max_steps:
                    return
                
    @property
    def episodes(self):
        # returns generator over episodes
        def generator(episode):
            for step in episode['steps']:
                step_dict = self.to_dict(step)
                if step_dict:
                    yield(step_dict)
                
        for episode in iter(self.dataset):
            yield(generator(episode))
            
    @staticmethod
    def download(url, cache_dir='/data/datasets', **kwargs):
        if url.startswith('gs'):
            download_cmd = f'gsutil -m cp -r -n {url} {cache_dir}'
        elif url.startswith('http'):
            download_cmd = f'wget -r --no-clobber --no-host-directories --reject="index.html*" --directory-prefix={cache_dir} {kwargs.get("wget_flags", "")} {url}'
        else:
            raise ValueError(f'invalid path or URL:  {url}')
            
        logging.info(f'RLDSDataset | running command to download TFDS dataset from {url} to {cache_dir}\n{download_cmd}')
        subprocess.run(download_cmd, executable='/bin/bash', shell=True, check=True)
                              
    def dump(self, max_steps=1):
        for i, step in enumerate(self):
            pprint.pprint(step, indent=2)
            if max_steps and i > max_steps:
                break

    def to_dict(self, step, strict=True):
        data = AttributeDict(
            action=step.get('action'),
            images=[],
            instruction=None,
            is_first=step.get('is_first'),
            is_last=step.get('is_last'),
        )
        
        observation = step['observation']
        image_keys = ['image', 'agentview_rgb', 'image_wrist']
        
        for image_key in image_keys:
            if image_key in observation:
                data.images.append(observation[image_key].numpy())

        instruction = observation.get('natural_language_instruction', step.get('language_instruction'))
        
        if instruction is not None:
            data.instruction = instruction.numpy().decode('UTF-8')
         
        if strict:
            for key, value in data.items():
                if value is None or (not key.startswith('is') and not value):
                    logging.warning(f"RLDSDataset | episode step has missing or empty key: {key}  (skipping)")           
                    return None
                    
        return data

     
def tensorflow_disable_device(device):
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    import tensorflow as tf
    
    devices = tf.config.list_physical_devices(device)
    
    if not devices:
        return

    logging.warning(f"disabling tensorflow device {device} ({len(devices)})")
    tf.config.set_visible_devices([], device)
    logical_devices = tf.config.list_logical_devices(device)
    logging.info(f"tensorflow  Physical {device}: {len(devices)}  Logical {device}: {len(logical_devices)}")

                  
class OpenXDataset(RLDSDataset):
    """
    Open X-Embodiment (OXE) dataset collection (https://github.com/google-deepmind/open_x_embodiment)
    """
    Names = ['fractal20220817_data', 'kuka', 'bridge', 'taco_play', 'jaco_play', 'berkeley_cable_routing', 'roboturk', 'nyu_door_opening_surprising_effectiveness', 'viola', 'berkeley_autolab_ur5', 'toto', 'language_table', 'columbia_cairlab_pusht_real', 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds', 'nyu_rot_dataset_converted_externally_to_rlds', 'stanford_hydra_dataset_converted_externally_to_rlds', 'austin_buds_dataset_converted_externally_to_rlds', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'maniskill_dataset_converted_externally_to_rlds', 'furniture_bench_dataset_converted_externally_to_rlds', 'cmu_franka_exploration_dataset_converted_externally_to_rlds', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'ucsd_pick_and_place_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'usc_cloth_sim_converted_externally_to_rlds', 'utokyo_pr2_opening_fridge_converted_externally_to_rlds', 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', 'utokyo_saytap_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds', 'utokyo_xarm_bimanual_converted_externally_to_rlds', 'robo_net', 'berkeley_mvp_converted_externally_to_rlds', 'berkeley_rpt_converted_externally_to_rlds', 'kaist_nonprehensile_converted_externally_to_rlds', 'stanford_mask_vit_converted_externally_to_rlds', 'tokyo_u_lsmo_converted_externally_to_rlds', 'dlr_sara_pour_converted_externally_to_rlds', 'dlr_sara_grid_clamp_converted_externally_to_rlds', 'dlr_edan_shared_control_converted_externally_to_rlds', 'asu_table_top_converted_externally_to_rlds', 'stanford_robocook_converted_externally_to_rlds', 'eth_agent_affordances', 'imperialcollege_sawyer_wrist_cam', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'uiuc_d3field', 'utaustin_mutex', 'berkeley_fanuc_manipulation', 'cmu_food_manipulation', 'cmu_play_fusion', 'cmu_stretch', 'berkeley_gnm_recon', 'berkeley_gnm_cory_hall', 'berkeley_gnm_sac_son']

    def __init__(self, name, cache_dir="/data/datasets/open_x_embodiment", **kwargs):
        super().__init__(
            f"gs://gresearch/robotics/{name}", name=name, 
            cache_dir=cache_dir, **kwargs
        )

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
    def __init__(self, cache_dir="/data/datasets/bridge/tfds", **kwargs):
        super().__init__(
            "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset", 
            wget_flags="--cut-dirs=4", cache_dir=cache_dir, **kwargs
        )
        self.name = 'bridge_orig'
        
        
class BridgeDatasetRaw:
    """
    BridgeData V2 robot dataset (https://github.com/rail-berkeley/bridge_data_v2)
    This is for the raw dataset with 640x480 images, extracted from one of:
    
      https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip (411 GB)
      https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_6_18.zip (30 GB)
      
    https://github.com/kvablack/dlimp/blob/main/rlds_converters/bridge_dataset/bridge_dataset_dataset_builder.py
    """
    def __init__(self, path, max_episodes=None, max_steps=None, **kwargs):
        """
        Scan the path for directories containing episodes of images, trajectories, and action data.
        """
        self.root = path
        self.name = 'bridge_raw'
        
        self.num_images = 0
        self.max_steps = max_steps
        
        metadata_path = os.path.join(path, 'metadata.pkl')
        self.metadata = self.load_metadata(metadata_path)

        if not self.metadata:
            self.metadata = self.scan(path, max_episodes=max_episodes)
            logging.info(f"BridgeDataset | saving {metadata_path}")
            with open(metadata_path, 'wb') as file:
                pickle.dump(self.metadata, file, protocol=pickle.HIGHEST_PROTOCOL)

        if max_episodes and len(self.metadata) > max_episodes:
            self.metadata = self.metadata[:max_episodes]
            
        for episode in self.metadata:
            self.num_images += len(episode.steps)

        logging.success(f"BridgeDataset | loaded index of {len(self.metadata)} episodes, {self.num_images} images from {self.root}")       

    def __iter__(self):
        """
        Returns an iterator over all steps (or up to max_steps if it was set) with the episodes running back-to-back.  
        `step.is_first` will be set on new episodes, and `set.is_last` will be set at the end of an episode.
        """
        steps = 0
        for episode in self.metadata:
            for step in episode.steps:
                yield(step)
                steps += 1
                if self.max_steps and steps >= self.max_steps:
                    return

    def scan(self, path, max_episodes=None):
        """
        Search for episodes and build the metadata index.
        """
        paths = glob.glob(os.path.join(self.root, *("*" * 4)))
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
                
                all_traj = glob.glob(search_path)
                
                if not all_traj:
                    logging.warning(f"BridgeDataset | no trajectories found in {search_path}  (skipping)")
                    continue

                '''
                config_path = os.path.join(path, dated_folder, "config.json")
                config = {}
                
                if os.path.exists(config_path):
                    with open(config_path, "rb") as f:
                        config = json.load(f)
                '''
                
                for ep_path in all_traj:
                    image_dirs = glob.glob(os.path.join(ep_path, 'images*'))
                    images = []
                    
                    for image_dir in image_dirs:
                        image_files = sorted(
                            glob.glob(os.path.join(image_dir, 'im_*.jpg')),
                            key=lambda x: int(x.split("_")[-1].split(".")[0]),
                        )
 
                        if image_files:   
                            images.append(image_files)

                    if not images:
                        logging.warning(f"BridgeDataset | episode {ep_path} did not contain any images (skipping)")
                        continue
                     
                    if not all(len(images[i]) == len(images[0]) for i in range(len(images))):
                        logging.warning(f"BridgeDataset | episode {ep_path} had {len(images)} sets of images with different number of frames {[len(x) for x in images]}")
                        continue
                                             
                    state = self.load_state(ep_path)
                    actions = self.load_actions(ep_path)
                    instruction = self.load_lang(ep_path)

                    if not instruction:
                        logging.warning(f"BridgeDataset | episode {ep_path} was missing 'instruction' annotation (skipping)")
                        continue
                    
                    steps = len(images[0])
                    lengths = dict(images=steps, state=len(state), action=len(actions)+1)

                    if not all(x == steps for x in lengths.values()):
                        logging.warning(f"BridgeDataset | episode {ep_path} did not have matching number of observations {lengths}")
                        continue

                    episode = AttributeDict(path=ep_path, steps=[])
                    
                    for i in range(steps):
                        episode.steps.append(AttributeDict(                   
                            state = state[i],
                            action = actions[i] if i < len(actions) else None,
                            images = [x[i] for x in images],
                            instruction = instruction,
                            is_first = bool(i == 0),
                            is_last = bool(i == (steps-1))
                        ))
                    
                    episodes += [episode]
                    num_images += steps

                    if len(episodes) % 250 == 0:
                        logging.debug(f"BridgeDataset | scanned {len(episodes)} episodes, {num_images} images from {self.root}")
                        
                    if max_episodes and len(episodes) >= max_episodes:
                        return episodes    
            
        if not episodes:
            raise RuntimeError(f"BridgeDataset | did not find any episodes under {self.root}")

        return episodes
     
    def dump(self, max_steps=1):
        for i, step in enumerate(self):
            if max_steps and i >= max_steps:
                break
            pprint.pprint(step, indent=2)
        
    def load_metadata(self, path):
        if not os.path.isfile(path):
            logging.warning(f"BridgeDataset | missing {path} - scanning metadata")
            return []
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as error:
            logging.error(error)
   
    def load_state(self, path):
        fp = os.path.join(path, "obs_dict.pkl")
        if not os.path.isfile(fp):
            logging.warning(f"BridgeDataset | missing {fp}")
            return []
        with open(fp, "rb") as f:
            x = pickle.load(f)
        #print('loading state', fp, list(x.keys()))
        return x["full_state"]

    def load_actions(self, path):
        fp = os.path.join(path, "policy_out.pkl")
        if not os.path.isfile(fp):
            logging.warning(f"BridgeDataset | missing {fp}")
            return []
        with open(fp, "rb") as f:
            act_list = pickle.load(f)
        pprint.pprint(act_list, indent=2)
        if isinstance(act_list[0], dict):
            act_list = [x["actions"] for x in act_list]
        return act_list

    def load_lang(self, path):
        fp = os.path.join(path, "lang.txt")
        if not os.path.isfile(fp):
            logging.warning(f"BridgeDataset | missing {fp}")
            return []
        with open(fp, "r") as f:
            text = f.readline().strip()
        return text


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

    args = parser.parse_args()

    dataset = load_dataset(**vars(args))
    
    dataset.dump()
     
