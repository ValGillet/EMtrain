import json
import numpy as np
import os

from glob import glob
from tqdm import tqdm

from emtrain.evaluate.workers.evaluate_model_worker import evaluate_model

from subprocess import check_call


def evaluate_models(output_dir,
                   model_dir,
                   eval_config, 
                   num_workers,
                   GPU_pool,
                   checkpoints_start=0,
                   checkpoints_end=1000000,
                   seg_config_path='seg_config.json',
                   catmaid_secrets=None):
    
    catmaid_secrets = os.environ['CATMAID_CREDENTIALS'] if catmaid_secrets is None else catmaid_secrets
    
    thresholds_minmax = [0,1]
    thresholds_step = 0.05
    
    # Find model info
    model_dir = os.path.abspath(model_dir)
    project_name, model_name = model_dir.split('/')[-2:]

    with open(os.path.join(model_dir, 'training_config.json'), 'r') as f:
        training_config = json.load(f)

    # Create destination directory
    project_dir = os.path.join(output_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # Find relevant model checkpoints
    model_checkpoints = {}
    for checkpoint_path in glob(os.path.join(model_dir, 'checkpoints/model_checkpoint*')):
        checkpoint_it = int(checkpoint_path.split('_')[-1])

        if checkpoint_it >= checkpoints_start and checkpoint_it <= checkpoints_end:
            # Only keep checkpoints that are contained in the provided range
            model_checkpoints.update({checkpoint_it: checkpoint_path})
    
    # Find eval info including test volume ROI
    with open(eval_config, 'r') as f:
        eval_config = json.load(f)
    species_name        = eval_config['project']
    volume_suffix       = eval_config['volume_suffix']
    raw_path            = eval_config['raw_path']


    # Run segmentation of the test volume for each checkpoint
    padding = np.array(training_config['model']['input_shape']) - np.array(training_config['model']['output_shape'])
    model_configs = {i: {
                'model_path': checkpoint,
                'num_fmaps': training_config['model']['num_fmaps'],
                'output_shape': training_config['model']['output_shape'],
                'padding': padding.tolist()
                     } for i, checkpoint in dict(sorted(model_checkpoints.items())).items()}

    for checkpoint_it, model_config in tqdm(model_configs.items(), desc='Evaluating models'):
        project_prefix = '_'.join(['eval', model_name, str(checkpoint_it), species_name])
        db_name = os.path.basename(raw_path).rstrip('.zarr')
        db_name = project_prefix + '_' + db_name
        db_name = db_name + '_' + volume_suffix

        config = {
                'project_dir': project_dir,
                'project_prefix': project_prefix,
                'model_config': model_config,
                'eval_config': eval_config,
                'catmaid_secrets': catmaid_secrets,
                'seg_config_path': seg_config_path,
                'num_workers': num_workers,
                'thresholds_minmax': thresholds_minmax,
                'thresholds_step': thresholds_step,
                'GPU_pool': GPU_pool
                 }

        worker_script = '/mnt/hdd1/SRC/EMpipelines/EMtrain/emtrain/evaluate/workers/evaluate_model_worker.py'
        config_dir = os.path.join(os.path.dirname(worker_script), 'worker_config')
        config_path = os.path.join(config_dir, db_name + '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent='')

        command = ['python', 
                   worker_script, 
                   config_path]
        check_call(' '.join(command), shell=True)


 


if __name__ == '__main__':

    output_dir = '/mnt/hdd1/SRC/EMpipelines/EMtrain/output'
    
    eval_config = '/mnt/hdd1/SRC/EMpipelines/EMtrain/emtrain/evaluate/volumes_config/NO_test_0_reduced.json'
    seg_config = '/mnt/hdd1/SRC/EMpipelines/EMtrain/emtrain/evaluate/seg_config.json'
    num_workers = 5
    GPU_pool = [2,3]
    checkpoints_start = 300000
    checkpoints_end = 1000000

    model_dirs = sorted(glob('/mnt/hdd2/DATA/segmentation_training/eciton_only/*'))
    for model_dir in model_dirs:
        print(model_dir)
        evaluate_models(output_dir,
                        model_dir,
                        eval_config, 
                        num_workers,
                        GPU_pool,
                        checkpoints_start=checkpoints_start,
                        checkpoints_end=checkpoints_end,
                        seg_config_path=seg_config)