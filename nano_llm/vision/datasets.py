#!/usr/bin/env python3
import os
import json
import time
import pprint
import logging
import subprocess

import torch
import numpy as np

from nano_llm.utils import AttributeDict


__all__ = ['load_dataset', 'DatasetTypes', 'OpenXDataset']


class OpenXDataset:
    """
    Open X-Embodiment dataset collection (https://github.com/google-deepmind/open_x_embodiment)
    """
    Names = ['fractal20220817_data', 'kuka', 'bridge', 'taco_play', 'jaco_play', 'berkeley_cable_routing', 'roboturk', 'nyu_door_opening_surprising_effectiveness', 'viola', 'berkeley_autolab_ur5', 'toto', 'language_table', 'columbia_cairlab_pusht_real', 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds', 'nyu_rot_dataset_converted_externally_to_rlds', 'stanford_hydra_dataset_converted_externally_to_rlds', 'austin_buds_dataset_converted_externally_to_rlds', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'maniskill_dataset_converted_externally_to_rlds', 'furniture_bench_dataset_converted_externally_to_rlds', 'cmu_franka_exploration_dataset_converted_externally_to_rlds', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'ucsd_pick_and_place_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'usc_cloth_sim_converted_externally_to_rlds', 'utokyo_pr2_opening_fridge_converted_externally_to_rlds', 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', 'utokyo_saytap_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds', 'utokyo_xarm_bimanual_converted_externally_to_rlds', 'robo_net', 'berkeley_mvp_converted_externally_to_rlds', 'berkeley_rpt_converted_externally_to_rlds', 'kaist_nonprehensile_converted_externally_to_rlds', 'stanford_mask_vit_converted_externally_to_rlds', 'tokyo_u_lsmo_converted_externally_to_rlds', 'dlr_sara_pour_converted_externally_to_rlds', 'dlr_sara_grid_clamp_converted_externally_to_rlds', 'dlr_edan_shared_control_converted_externally_to_rlds', 'asu_table_top_converted_externally_to_rlds', 'stanford_robocook_converted_externally_to_rlds', 'eth_agent_affordances', 'imperialcollege_sawyer_wrist_cam', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'uiuc_d3field', 'utaustin_mutex', 'berkeley_fanuc_manipulation', 'cmu_food_manipulation', 'cmu_play_fusion', 'cmu_stretch', 'berkeley_gnm_recon', 'berkeley_gnm_cory_hall', 'berkeley_gnm_sac_son']
    Cache = "/data/datasets/open_x_embodiment"
    
    def __init__(self, name, split='train', max_episodes=None, max_steps=None, cache_dir=Cache):
        import tensorflow_datasets as tdfs

        if not name:
            raise ValueError(f"select the dataset name with one of these values: {OpenXDataset.Names}")

        os.makedirs(cache_dir, exist_ok=True)
        
        # tensorflow gets used for loading tfrecords/tensors, but we don't want it hogging all the GPU memory,
        # which it preallocates by default - so just disable it from using GPUs since it's not needed anyways.
        tensorflow_disable_device('GPU')

        # https://github.com/google-deepmind/open_x_embodiment?tab=readme-ov-file#dataset-not-found
        download_cmd = f"gsutil -m cp -r -n gs://gresearch/robotics/{name} {cache_dir}"
        logging.info(f"running command to download OpenXEmbodiment dataset '{name}' to {cache_dir}\n{download_cmd}")
        subprocess.run(download_cmd, executable='/bin/bash', shell=True, check=True)
        
        if max_episodes:
            split = f"{split}[:{max_episodes}]"
            
        self.name = name
        self.dataset = tdfs.load(name, split=split, data_dir=cache_dir)

        self.num_steps = 0
        self.max_steps = max_steps

        self.num_episodes = 0
        self.episode_index = []
        
        for episode in iter(self.dataset):
            self.num_steps += len(episode['steps'])
            self.num_episodes += 1
            self.episode_index.append(self.num_steps)

        if self.max_steps:
            self.num_steps = min(self.num_steps, self.max_steps)

        self.image_key = None
        self.image_dim = self.step(0).get('image')
        
        if self.image_dim is not None:
            self.image_dim = self.image_dim.shape
 
        logging.success(f"loaded OpenXEmbodiment dataset '{name}' (episodes={self.num_episodes}, steps={self.num_steps}, image={self.image_dim})")
    
    def __len__(self):
        return self.num_steps
          
    def __iter__(self):
        i = 0
        for episode in iter(self.dataset):
            for step in episode['steps']:
                yield(self.to_dict(step))
                i += 1
                if self.max_steps and i >= self.max_steps:
                    return

    def __getitem__(self, index):
        return self.step(index)
    
    @property
    def episodes(self):
        # returns generator over episodes
        def generator(episode):
            for step in episode['steps']:
                yield(self.to_dict(step))
                
        for episode in iter(self.dataset):
            yield(generator(episode))
            
    def episode(self, index):
        # returns generator for a specific episode
        for e, episode in enumerate(iter(self.dataset)):
            if e == index:
                for step in episode['steps']:
                    yield(self.to_dict(step))
     
    def step(self, index):
        # this is slow because the tf.Dataset can't be indexed, only iterated
        i = 0
        
        for e, episode in enumerate(iter(self.dataset)):
            if index < self.episode_index[e]:
                for step in episode['steps']:
                    if i == index:
                        return self.to_dict(step)
                    i += 1
            else:
                i = self.episode_index[e]
         
        '''
        for i, index in enumerate(self.episode_index):
            if step < index:
                episode = next(iter(self.dataset.range(i,i+1)))
                episode_step = step - self.episode_index[i-1] if i > 0 else step
                step = next(iter(episode.range(episode_step, episode_step+1)))
                return self.to_dict(step)
        '''
                                             
    def dump(self, path):
        episode = next(iter(self.dataset))
        step = next(iter(episode['steps']))
        pprint.pprint(step, indent=2)
        for step in self:
            pprint.pprint(step, indent=2)
            break
     
    @staticmethod
    def to_dict(step):
        obs = step['observation']
        data = AttributeDict()
        
        image_keys = ['image', 'agentview_rgb']
        
        for image_key in image_keys:
            if image_key in obs:
                self.image_key = image_key
                data.image = obs[image_key].numpy()
                break

        if 'image_wrist' in obs:
            data.image_wrist = obs['image_wrist'].numpy()
        
        instruction = obs.get('natural_language_instruction', step.get('language_instruction'))
        
        if instruction is not None:
            data.instruction = instruction.numpy().decode('UTF-8')
             
        return data
                      
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


'''
class BridgeDataset:
    """
    BridgeData V2 robot dataset from https://github.com/rail-berkeley/bridge_data_v2
    This is for the raw dataset with 640x480 images, extracted from one of:
    
      https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip (411 GB)
      https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_6_18.zip (30 GB)
    """
    def __init__(self, path, **kwargs):
        """
        Scan the path for directories containing episodes and 
        """
        import h5py

        data = {}
        
        with h5py.File(path, "r") as file:
            print(file.attrs)
            print(list(file.keys()))
            for key, value in file.items():
                print(key, value.shape, value.dtype)
                array = np.array(file[key])
                print(key, array.shape, array.dtype)
                #data[key] = np.array(F[key])

        logging.success(f"loaded JacoPlay dataset from {path}")
'''

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
}

def load_dataset(dataset: str=None, dataset_type: str=None, **kwargs):
    if not dataset_type:
        if dataset in DatasetTypes:
            dataset_type = dataset
            dataset = None
        else:    
            dataset_type = 'openx'
        
    return DatasetTypes[dataset_type](dataset, **kwargs)

                
if __name__ == "__main__":

    from nano_llm.utils import ArgParser
    
    # parse args and set some defaults
    parser = ArgParser(extras=['log'])
    
    parser.add_argument("--dataset", type=str, default=None, required=True, help=f"path or name of the dataset to load")
    parser.add_argument("--dataset-type", type=str, default=None, choices=list(DatasetTypes.keys()), help=f"type of the dataset to load")
    parser.add_argument("--max-episodes", type=int, default=None, help="the maximum number of episodes from the dataset to process")
    parser.add_argument("--max-steps", type=int, default=None, help="the maximum number of frames to process across all episodes")

    args = parser.parse_args()

    print(args)

    dataset = load_dataset(**vars(args))
    dataset.dump(None)
     
