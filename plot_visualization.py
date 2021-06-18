"""
This code can visualize the perturbation of the poisoned data. 
Please notice that the order of the clean and poisoned data MUST BE SAME.
"""
import os
import argparse
import numpy as onp
from utils import *

# Plotting
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

parser = argparse.ArgumentParser(description="Visualization of the poisoned data and their \
                                 normalized perturbations. Please provide both clean and poisoned data.")
parser.add_argument("--dataset", required=True, type=str, help="`mnist`, `cifar10`, and `imagenet` are \
                    available. To use different dataset, please specify the input size in the code directly")
parser.add_argument("--x_train_path", required=True, type=str, help="path for the clean data")
parser.add_argument("--x_train_ntga_path", required=True, type=str, help="path for the poisoned data")
parser.add_argument("--num", default=5, type=int, help="number of visualized data. \
                    The valid value is 1-5")
parser.add_argument("--save_path", default="", type=str, help="path to save figures")

args = parser.parse_args()

save = True
if args.dataset == "mnist":
    image_size = 28
    shape = (args.num*3, image_size, image_size)
    scale = 2
elif args.dataset == 'cifar10':
    image_size = 32
    shape = (args.num*3, image_size, image_size, 3)
    scale = 2
elif args.dataset == "imagenet":
    image_size = 224
    shape = (args.num*3, image_size, image_size, 3)
    scale = 2

def main():
    # Prepare dataset
    print("Loading dataset...")
    x_train = onp.load(args.x_train_path)
    x_train_ntga = onp.load(args.x_train_ntga_path)
    x_train, x_train_ntga = shaffle(x_train, x_train_ntga)
    
    # Plot visualization
    print("Plotting visualization...")
    if save and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    if args.num > 5:
        args.num = 5
        
    _x_train = onp.zeros((args.num*3, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    
    # Clean data
    _x_train[:args.num*1] = x_train[:args.num]
    
    # Poisoned data
    _x_train[args.num*1:args.num*2] = x_train_ntga[:args.num]
    
    # Normalized perturbation
    diff = x_train[:args.num] - x_train_ntga[:args.num]
    diff = diff*0.5 + 0.5
    diff = (diff-diff.min()) / (diff.max()-diff.min())
    _x_train[args.num*2:] = diff
    
    plot_visualization(_x_train, shape, num_row=3, num_col=args.num, scale=scale, 
                       row_title=["Clean", "NTGA", "Normalized Perturbation"], save=save, 
                       fname="{:s}figure_{:s}_visualization".format(args.save_path, args.dataset))
    print("================== DONE ==================")
    
if __name__ == "__main__":
    main()
