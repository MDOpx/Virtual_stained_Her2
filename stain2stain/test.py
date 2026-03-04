"""Inference script for CPT (Contrastive Paired Translation) image-to-image model.

Loads a saved model from --checkpoints_dir and saves fake_B images to --results_dir.
Uses --model cpt and --dataset_mode aligned. See options/base_options.py and
options/test_options.py for options.
"""
import os
import sys
import torch

# Run from repo root: python stain2stain/test.py ...
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import util.util as util


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.dataroot = os.path.abspath(os.path.expanduser(opt.dataroot))
    if not os.path.isdir(opt.dataroot):
        raise FileNotFoundError('dataroot is not a directory: %s' % opt.dataroot)
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)
    model = create_model(opt)
    os.makedirs(opt.results_dir, exist_ok=True)
    print('Saving fake_B to:', opt.results_dir)

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        # fake_B만 results_dir에 저장
        if 'fake_B' in visuals:
            name = os.path.splitext(os.path.basename(img_path[0]))[0]
            im = util.tensor2im(visuals['fake_B'])
            save_path = os.path.join(opt.results_dir, '%s.png' % name)
            util.save_image(im, save_path)
    print('Done. Results saved to:', opt.results_dir)
