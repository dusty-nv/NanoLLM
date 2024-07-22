#!/usr/bin/env python3
import os
import json
import logging
import subprocess

from glob import glob
from urllib.parse import urlparse
from nano_llm.utils import AttributeDict


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
        import tensorflow_datasets as tdfs
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


def tensorflow_disable_device(device):
    """
    TensorFlow gets used for loading TFDS records/tensors, but we don't want it hogging all the GPU memory,
    which it preallocates by default - so just disable it from using GPUs since it's not needed anyways.
    
        https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    """
    import tensorflow as tf
    devices = tf.config.list_physical_devices(device)
    
    if not devices:
        return

    logging.warning(f"disabling tensorflow device {device} ({len(devices)})")
    tf.config.set_visible_devices([], device)
    logical_devices = tf.config.list_logical_devices(device)
    logging.info(f"tensorflow  Physical {device}: {len(devices)}  Logical {device}: {len(logical_devices)}")

