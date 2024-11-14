import os
import json
import logging
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)

def get_augment_parameters(output_path,
                           return_config=False):
    
    logging.info('Computing augment parameters...')

    # gp.DeformAugment
    elastic_spacing = [10] + [np.random.randint(40, 101)]*2
    elastic_jitter = [0] + [np.random.randint(0, 11)]*2
    prob_elastic = np.random.uniform(0, 1) 

    # gp.IntensityAugment
    intensity_scmin = np.random.uniform(0.5, 1)
    intensity_scmax = np.random.uniform(1, 1.5)
    intensity_shmin = np.random.uniform(-0.05, -0.35)
    intensity_shmax = np.random.uniform(0.05, 0.35)

    # gp.NoiseAugment
    prob_noise = np.random.uniform(0.01, 0.05)

    # gp.DefectAugment
    prob_missing = np.random.uniform(0.01, 0.1)
    prob_low_contrast = np.random.uniform(0.01, 0.1)
    prob_deform = np.random.uniform(0.01, 0.1)
    
    augment_config =  {
                    'elastic_spacing': elastic_spacing,
                    'elastic_jitter': elastic_jitter,
                    'prob_elastic': prob_elastic,
                    'intensity_scmin': intensity_scmin,
                    'intensity_scmax': intensity_scmax,
                    'intensity_shmin': intensity_shmin,
                    'intensity_shmax': intensity_shmax,
                    'prob_noise': prob_noise,
                    'prob_missing': prob_missing,
                    'prob_low_contrast': prob_low_contrast,
                    'prob_deform': prob_deform
                      }

    if return_config:
        return augment_config
    
    logging.info(f'Saving parameters at {output_path}')
    with open(output_path, 'w') as f:
        json.dump(augment_config, f, indent='')

if __name__ == '__main__':
    
    output_path = sys.argv[1]
    get_augment_parameters(output_path)