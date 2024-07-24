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
        data = DatasetTypes[dataset_type](dataset, **kwargs)
    else:
        data = DatasetTypes[dataset_type](**kwargs)
        
    data.path = dataset
    data.type = dataset_type
    
    return data
    
    
def convert_dataset(dataset: str=None, dataset_type: str=None, 
                    output: str=None, output_type: str=None, 
                    width: int=None, height: int=None,
                    max_episodes: int=None, max_steps: int=None,
                    sample_steps: int=None, sample_actions: int=None, 
                    **kwargs):
    """
    Convert a dataset from one type to another (currently only exporting to RLDS is supported)
    """
    if any([bool(not x) for x in [dataset, dataset_type, output, output_type]]):
        raise ValueError("must supply valid arguments to convert_dataset()")
    
    if not hasattr(DatasetTypes[output_type], 'export'):
        raise ValueError(f"{output_type} is not a type of dataset that can be exported")

    return DatasetTypes[output_type].export(
        dataset=dataset,
        dataset_type=dataset_type,
        output=output,
        width=width, height=height, 
        max_episodes=max_episodes,
        max_steps=max_steps, 
        sample_steps=sample_steps,
        sample_actions=sample_actions, 
        **kwargs
    )
