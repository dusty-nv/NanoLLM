#!/usr/bin/env python3
from .tfds import TFDSDataset
from .rlds import RLDSDataset

from .oxe import OXEDataset
from .droid import DroidDataset
from .bridge import BridgeDataset, BridgeDatasetRaw
from .robomimic import RobomimicDataset

DatasetTypes = {
    'tfds': TFDSDataset,
    'rlds': RLDSDataset,
    'oxe': OXEDataset,
    'droid': DroidDataset,
    'bridge_orig': BridgeDataset,
    'bridge_raw': BridgeDatasetRaw,
    'robomimic': RobomimicDataset,
    'mimicgen': RobomimicDataset,
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
            dataset_type = 'oxe'
    
    if dataset:    
        dataset = DatasetTypes[dataset_type](dataset, **kwargs)
    else:
        dataset = DatasetTypes[dataset_type](**kwargs)
        
    dataset.type = dataset_type
    return dataset
