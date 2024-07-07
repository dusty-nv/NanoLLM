#!/usr/bin/env python3
#
# Example for vision/language action (VLA) model inference on video streams,
# with benchmarking and accuracy evaluation on Open-X Embodiment datasets.
#
# You can run it on video files or devices like this:
#
#    huggingface-cli download --local-dir /data/models/openvla-7b openvla/openvla-7b
#
#    python3 -m nano_llm.vision.vla \
#      --model openvla/openvla-7b \
#      --eval-model /data/models/openvla-7b \
#      --dataset /data/o
#      --max-images 8 \
#      --max-new-tokens 48 \
#      --video-input /data/my_video.mp4 \
#      --video-output /data/my_output.mp4 \
#      --prompt 'What changes occurred in the video?'
#
# The model should have been trained to understand video sequences (like VILA-1.5)
#
import os
import sys
import time
import pprint
import logging
import subprocess
import transformers

import numpy as np

from nano_llm import NanoLLM, ChatHistory, remove_special_tokens
from nano_llm.utils import AttributeDict, convert_tensor



class VLAModel:
    def __init__(self, model="openvla/openvla-7b", actions={}, max_images=1, **kwargs):
        if isinstance(model, str):
            self.model = NanoLLM.from_pretrained(model, **kwargs)
        elif isinstance(model, NanoLLM):
            self.model = model
        else:
            raise TypeError(f"expected model as str or NanoLLM (was {type(model)})")
            
        self.chat = ChatHistory(model, **kwargs)
        self.prompt = "What action should the robot take to ${INSTRUCTION}?"  
        self.instruction = kwargs.get('instruction', 'stop')
        
        self.num_images = 0
        self.max_images = max_images

        assert(model.has_vision)
        
        # setup action configs
        actions['unnormalized'] = dict(normalized=False)
        
        for key, scene in actions.items():
            scene['name'] = key
            action = scene['action']
            for stat_key, stat in action.items():
                if isinstance(stat, list):
                    action[stat_key] = np.array(stat)

        self.action_configs = actions
        self.actions = 'unnormalized'
        
        # map the tokenizer vocab range to discrete action bins
        self.bins = np.linspace(-1, 1, self.model.config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = self.config.vocab_size - self.config.pad_to_multiple_of
        
        # LLM warmup
        self.chat.append(role='user', text='What is 2+2?')
        logging.info(f"Warmup response:  '{self.model.generate(self.chat.embed_chat()[0], streaming=False)}'".replace('\n','\\n'))
        self.chat.reset()

    @property
    def actions(self):
        return self._actions
        
    @actions.setter
    def actions(self, key):
        if isinstance(key, str):
            self._actions = self.action_configs[key]
        elif isinstance(key, dict):
            self._actions = key
        else:
            raise TypeError(f"actions can be set to str or dict (was {type(key)})")
            
    def embed(self, image, instruction='', prompt=None, **kwargs):
        if not instruction:
            instruction = self.instruction
         
        if not prompt:
            prompt = self.prompt
        
        prompt = prompt.replace('${INSTRUCTION}', instruction)
        
        logging.debug(f"{self.model.config.name} prompt:  `{prompt}`")

        self.chat.reset()
        
        self.chat.append('user', image=image)
        self.chat.append('user', text=prompt)
        
        return self.chat.embed_chat()[0]
        
    def predict_actions(self, image, actions={}, **kwargs):
        if not actions:
            actions = self.actions

        num_actions = len(actions['q01'])

        reply = model.generate(
            self.embed(image, **kwargs),
            kv_cache=self.chat.kv_cache,
            detokenize=False,
            min_new_tokens=num_actions,
            max_new_tokens=num_actions+1,
            **kwargs,
        ).wait()
        
        return decode_actions(reply.tokens[:num_actions], actions, **kwargs)
        
    def decode_actions(self, tokens, actions={}, return_tensors='np', **kwargs):
        if not actions:
            actions = self.actions
   
        # map from vocab bins back into action space (-1,1)
        pred_actions = self.vocab_size - convert_tensor(tokens, return_tensors='np')
        pred_actions = np.clip(pred_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        pred_actions = self.bin_centers[pred_actions]
        
        # denormalize the actions using robot coefficients
        if not actions.get('normalized', True):
            action_high, action_low = actions['q99'], actions['q01']
            pred_actions = np.where(
                actions['mask'],
                0.5 * (pred_actions + 1) * (action_high - action_low) + action_low,
                pred_actions,
            )
        
        return convert_tensors(pred_actions, return_tensors=return_tensors, **kwargs)
     
    def __call__(self, image, **kwargs):
        return self.predict_actions(image, **kwargs)
           

class OpenXDataset:
    Names = ['fractal20220817_data', 'kuka', 'bridge', 'taco_play', 'jaco_play', 'berkeley_cable_routing', 'roboturk', 'nyu_door_opening_surprising_effectiveness', 'viola', 'berkeley_autolab_ur5', 'toto', 'language_table', 'columbia_cairlab_pusht_real', 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds', 'nyu_rot_dataset_converted_externally_to_rlds', 'stanford_hydra_dataset_converted_externally_to_rlds', 'austin_buds_dataset_converted_externally_to_rlds', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'maniskill_dataset_converted_externally_to_rlds', 'furniture_bench_dataset_converted_externally_to_rlds', 'cmu_franka_exploration_dataset_converted_externally_to_rlds', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'ucsd_pick_and_place_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'usc_cloth_sim_converted_externally_to_rlds', 'utokyo_pr2_opening_fridge_converted_externally_to_rlds', 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', 'utokyo_saytap_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds', 'utokyo_xarm_bimanual_converted_externally_to_rlds', 'robo_net', 'berkeley_mvp_converted_externally_to_rlds', 'berkeley_rpt_converted_externally_to_rlds', 'kaist_nonprehensile_converted_externally_to_rlds', 'stanford_mask_vit_converted_externally_to_rlds', 'tokyo_u_lsmo_converted_externally_to_rlds', 'dlr_sara_pour_converted_externally_to_rlds', 'dlr_sara_grid_clamp_converted_externally_to_rlds', 'dlr_edan_shared_control_converted_externally_to_rlds', 'asu_table_top_converted_externally_to_rlds', 'stanford_robocook_converted_externally_to_rlds', 'eth_agent_affordances', 'imperialcollege_sawyer_wrist_cam', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'uiuc_d3field', 'utaustin_mutex', 'berkeley_fanuc_manipulation', 'cmu_food_manipulation', 'cmu_play_fusion', 'cmu_stretch', 'berkeley_gnm_recon', 'berkeley_gnm_cory_hall', 'berkeley_gnm_sac_son']
    Cache = "/data/datasets/open_x_embodiment"
    
    def __init__(self, name, split='train', cache_dir=Cache):
        import tensorflow_datasets as tdfs

        if not name:
            raise ValueError(f"select the dataset name with one of these values: {OpenXDataset.Names}")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # tensorflow gets used for loading tfrecords/tensors, but we don't want it hogging all the GPU memory,
        # which it preallocates by default - so just disable it from using GPUs since it's not needed anyways.
        self.disable_tf_device('GPU')

        # https://github.com/google-deepmind/open_x_embodiment?tab=readme-ov-file#dataset-not-found
        download_cmd = f"gsutil -m cp -r -n gs://gresearch/robotics/{name} {cache_dir}"
        logging.info(f"running command to download OpenXEmbodiment dataset '{name}' to {cache_dir}\n{download_cmd}")
        subprocess.run(download_cmd, executable='/bin/bash', shell=True, check=True)
        
        self.dataset = tdfs.load(name, split=split, data_dir=cache_dir)
        '''
        self.dataset = tdfs.load(
            name, 
            split='train',
            data_dir="gs://gresearch/robotics",
            download_and_prepare_kwargs=dict(download_dir=cache_dir)
        )
        '''
     
    def __iter__(self):
        for episode in iter(self.dataset):
            for step in episode['steps']:
                obs = step['observation']
                
                data = AttributeDict(
                    instruction = obs['natural_language_instruction'].numpy().decode('UTF-8'),
                    image = obs['image'].numpy()
                )
                
                if 'image_wrist' in obs:
                    data['image_wrist'] = obs['image_wrist'].numpy()
                
                yield(data)

    def dump(self, path):
        episode = next(iter(self.dataset))
        step = next(iter(episode['steps']))
        pprint.pprint(step, indent=2)
        for step in self:
            pprint.pprint(step, indent=2)
            break
            
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
     
    @staticmethod
    def disable_tf_device(device):
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        import tensorflow as tf
        
        devices = tf.config.list_physical_devices(device)
        
        if not devices:
            return

        logging.warning(f"disabling tensorflow device {device} ({len(devices)})")
        tf.config.set_visible_devices([], device)
        logical_devices = tf.config.list_logical_devices(device)
        logging.info(f"tensorflow  Physical {device}: {len(devices)}  Logical {device}: {len(logical_devices)}")

    
                
if __name__ == "__main__":

    from nano_llm.utils import ArgParser, load_prompts, wrap_text
    from nano_llm.plugins import VideoSource, VideoOutput

    from termcolor import cprint
    from jetson_utils import cudaMemcpy, cudaToNumpy, cudaFont

    # parse args and set some defaults
    parser = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input', 'video_output'])
    
    parser.add_argument("--eval-model", type=str, default=None, help="path to the original HuggingFace model to enable error comparison")
    parser.add_argument("--dataset", type=str, default=None, choices=OpenXDataset.Names, help="the OpenX dataset to run inference evaluation on")
    parser.add_argument("--dump", type=str, default=None, help="dump the OpenX dataset to a directory of individual image files")
    parser.add_argument("--max-images", type=int, default=1, help="the number of video frames to keep in the history")
    
    args = parser.parse_args()

    if not args.model:
        args.model = "openvla/openvla-7b"
        
    '''
    prompts = load_prompts(args.prompt)

    if not prompts:
        prompts = ["pick up the block"]
    '''   
    
    print(args)

    if args.dump:
        dataset = OpenXDataset(args.dataset)
        dataset.dump(args.dump)
        sys.exit(0)
        
    # load vision/language model
    model = NanoLLM.from_pretrained(args.model, **vars(args))

    assert(model.vla)
    
    # open the video stream
    num_images = 0
    last_image = None
    last_text = ''

    def on_video(image):
        global last_image
        last_image = cudaMemcpy(image)
        if last_text:
            font_text = remove_special_tokens(last_text)
            wrap_text(font, image, text=font_text, x=5, y=5, color=(120,215,21), background=font.Gray50)
        video_output(image)
        
    video_source = VideoSource(**vars(args), cuda_stream=0)
    video_source.add(on_video, threaded=False)
    video_source.start()

    video_output = VideoOutput(**vars(args))
    video_output.start()

    font = cudaFont()

    # apply the prompts to each frame
    while True:
        if last_image is None:
            continue

        chat_history.append('user', text=f'Image {num_images + 1}:')
        chat_history.append('user', image=last_image)
        
        last_image = None
        num_images += 1

        for prompt in prompts:
            chat_history.append('user', prompt)
            embedding, _ = chat_history.embed_chat()
            
            print('>>', prompt)
            
            reply = model.generate(
                embedding,
                kv_cache=chat_history.kv_cache,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            
            for token in reply:
                cprint(token, 'blue', end='\n\n' if reply.eos else '', flush=True)
                if len(reply.tokens) == 1:
                    last_text = token
                else:
                    last_text = last_text + token

            chat_history.append('bot', reply)
            chat_history.pop(2)
            
        if num_images >= args.max_images:
            chat_history.reset()
            num_images = 0
            
        if video_source.eos:
            video_output.stream.Close()
            break
