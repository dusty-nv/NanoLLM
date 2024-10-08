#!/usr/bin/env python3
import os
import logging

from .tfds import TFDSDataset
from .rlds import RLDSDataset

from .oxe import OXEDataset
from .dump import DumpDataset
from .droid import DroidDataset
from .bridge import BridgeDataset, BridgeDatasetRaw
from .robomimic import RobomimicDataset

from ..utils import download_model


DatasetTypes = {
    'tfds': TFDSDataset,
    'rlds': RLDSDataset,
    'oxe': OXEDataset,
    'droid': DroidDataset,
    'bridge_orig': BridgeDataset,
    'bridge_raw': BridgeDatasetRaw,
    'robomimic': RobomimicDataset,
    'mimicgen': RobomimicDataset,
    'dump': DumpDataset,
}


def load_dataset(dataset: str=None, dataset_type: str=None, download=True, cache="/data/datasets/huggingface", **kwargs):
    """
    Dataset factory function that supports different dataset formats and sources.
    """
    if not dataset_type or dataset_type == 'None':
        if dataset in DatasetTypes:
            dataset_type = dataset
            dataset = None
        else:    
            dataset_type = 'oxe'

    if (
        download and dataset and 
        '/' in dataset and 
        not os.path.isdir(dataset)
    ):
        try:
            kwargs['name'] = dataset.split('/')[-1]
            dataset = download_model(
                dataset, type='dataset',
                cache_dir=os.environ.get('HF_DATASETS', cache),
            )   
        except Exception as error:
            logging.warning(f"could not download dataset {dataset} from HuggingFace Hub ({error})")
            
    if dataset:    
        data = DatasetTypes[dataset_type](dataset, **kwargs)
    else:
        data = DatasetTypes[dataset_type](**kwargs)
        
    data.path = dataset
    data.type = dataset_type
    
    return data
    
    
def convert_dataset(dataset: str=None, dataset_type: str=None, output: str=None, output_type: str=None, **kwargs):
    """
    Convert a dataset from one type to another (currently only exporting to RLDS is supported)
    @see :func:`RLDSDataset.export` and :func:`DumpDataset.export` for kwargs, like width/height.
    """
    if not dataset or not output_type: #any([bool(not x) for x in [dataset, dataset_type, output, output_type]]):
        raise ValueError("must supply dataset and output_type to convert_dataset()")
    
    if not hasattr(DatasetTypes[output_type], 'export'):
        raise ValueError(f"{output_type} is not a type of dataset that can be exported")

    return DatasetTypes[output_type].export(
        dataset=dataset,
        dataset_type=dataset_type,
        output=output,
        **kwargs
    )
