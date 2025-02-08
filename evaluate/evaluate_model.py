import json
import numpy as np
import os

from glob import glob
from tqdm import tqdm

from emsegment.Segment import segment_dataset
from emsegment.FindSegments import find_segments



def fun(output_dir,
        model_dir,
        eval_config, 
        num_workers,
        GPU_pool,
        checkpoints_start,
        checkpoints_end=1000000,
        seg_config='seg_config.json',
        model_name=None):
    
    # Find model info
    model_dir = os.path.abspath(model_dir)
    project_name, model_name = model_dir.split('/')[-2:]

    with open(os.path.join(model_dir, 'training_config.json'), 'r') as f:
        training_config = json.load(f)

    # Create destination directory
    project_dir = os.path.join(output_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # Find relevant model checkpoints
    model_checkpoints = []
    for checkpoint_path in glob(os.path.join(model_dir, 'checkpoints/model_checkpoint*')):
        checkpoint_it = int(checkpoint_path.split('_')[-1])

        if checkpoint_it <= checkpoints_start and checkpoint_it <= checkpoints_end:
            # Only keep checkpoints that are contained in the provided range
            model_checkpoints.update({checkpoint_it: checkpoint_path})
    
    # Find eval info including test volume ROI
    with open(eval_config, 'r') as f:
        eval_config = json.load(f)
    roi_start = eval_config['roi_start_nm']
    roi_size = eval_config['roi_size']
    catmaid_pid = eval_config['catmaid_project_id']
    species_name = eval_config['project']

    # Run segmentation of the test volume for each checkpoint
    padding = np.array(training_config['output_shape']) - np.array(training_config['input_shape'])
    model_configs = {i: {
                'model_path': checkpoint,
                'num_fmaps': training_config['num_fmaps'],
                'output_shape': training_config['output_shape'],
                'padding': padding
                     } for i, checkpoint in model_checkpoints.items()}

    for checkpoint_it, model_config in tqdm(model_configs, desc='Segmenting data'):
        project_prefix = '_'.join(['eval', project_name, model_name, checkpoint_it, species_name])

        # Segment
        seg_config = segment_dataset(
                                project_dir=project_dir,
                                project_prefix=project_prefix,
                                model_config=model_config,
                                input_path=eval_config['raw_path'],
                                raw_dataset=eval_config['raw_dataset'],
                                GPU_pool=GPU_pool,
                                num_workers=num_workers,
                                roi_start=roi_start,
                                roi_size=roi_size,
                                seg_config=seg_config,
                                return_config=True
                                )
        
        # Find connected components
        find_segments(
                  db_name=seg_config['db_name'],
                  fragments_path=seg_config['fragments_path'],
                  edges_collection=seg_config['edges_collection'],
                  thresholds_minmax=[0,1],
                  thresholds_step=0.2,
                  num_workers=num_workers)
        
    # Run evaluation
    # Return eval results

    project_dir = os.path.join(output_dir, project_name)

    