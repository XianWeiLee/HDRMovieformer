from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse


parser = argparse.ArgumentParser(
    description='HDR Enhancement using HDRformer')


# HDRMovie 1K
parser.add_argument('--input_dir', default='../example_data/HDRMovie1K/SDR', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--output_dir', default='', type=str, help='Directory for output')
parser.add_argument('--opt', type=str, default='HDR.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='net_g_latest.pth', type=str, help='Path to weights')

# HDRMovie 7K
# parser.add_argument('--input_dir', default='../example_data/HDRMovie7K/SDR', type=str, help='Directory of validation images')
# parser.add_argument('--result_dir', default='./results_zy/', type=str, help='Directory for results')
# parser.add_argument('--output_dir', default='', type=str, help='Directory for output')
# parser.add_argument('--opt', type=str, default='HDR_zy.yml', help='Path to option YAML file.')
# parser.add_argument('--weights', default='net_g_latest_zy.pth', type=str, help='Path to weights')

parser.add_argument('--dataset', default='HDR', type=str, help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')

args = parser.parse_args()

# gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)



def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''Converts a torch Tensor into an image Numpy array'''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    elif out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g

checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')

output_dir = args.output_dir

os.makedirs(result_dir, exist_ok=True)
if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)


input_dir = args.input_dir
print(f'input path {input_dir}')
print(f'result path {result_dir}')

input_paths = natsorted(glob(os.path.join(input_dir, '*.png')))
# input_paths = natsorted(glob(os.path.join(input_dir, '*.tif'))) # HDRMovie7K

with torch.inference_mode():
    for inp_path in tqdm(input_paths, total=len(input_paths)):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = cv2.imread(inp_path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / (2 ** 8 - 1)
        # img = img.astype(np.float32) / (2 ** 16 - 1) # HDRMovie7K
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_ = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
    

        # Padding in case images are not multiples of 4
        b, c, h, w = input_.shape
        H, W = ((h + factor) // factor) * \
            factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        restored = model_restoration(input_)
        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]

        restored = tensor2img(restored, np.uint16)

        # cv2.imwrite((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.tif')), restored) # HDRMovie7K
        cv2.imwrite((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png')), restored)


print('done!')
