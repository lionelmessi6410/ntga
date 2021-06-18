"""
This code is based on Disentangling Trainability and Generalization in Deep Neural Networks (L. Xiao et al. 2020)
Paper link: https://arxiv.org/pdf/1912.13053.pdf
Colab link: https://colab.research.google.com/github/google/neural-tangents/blob/master/
notebooks/Disentangling_Trainability_and_Generalization.ipynb#scrollTo=aH1Zet-tuKbw
"""
import os
import argparse
import jax.numpy as np
from jax.api import grad, jit, vmap
from jax import lax, random
from jax.config import config
config.update('jax_enable_x64', True)

from functools import partial
from neural_tangents import stax
from utils import *
from utils_jax import *

# Plotting
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import *
sns.set_style(style='white')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

parser = argparse.ArgumentParser(description="Plot training and test dynamics of Gaussian Process.")
parser.add_argument("--dataset", required=True, type=str, help="clean dataset. `mnist`, `cifar10`, \
                    and `imagenet` are available. To use different dataset, please modify the path \
                    in the code directly")
parser.add_argument("--dtype", required=True, type=str, help="`Clean` or `NTGA`, used for figure's title")
parser.add_argument("--x_train_path", default=None, type=str, help="path for training data. Leave it empty \
                    to evaluate the performance on clean data(mnist or cifar10)")
parser.add_argument("--y_train_path", default=None, type=str, help="path for training labels. Leave it empty \
                    to evaluate the performance on clean data(mnist or cifar10)")
parser.add_argument("--x_val_path", default=None, type=str, help="path for validation data. Please specify \
                    the path for the poisoned dataset")
parser.add_argument("--y_val_path", default=None, type=str, help="path for validation label. Please specify \
                    the path for the poisoned dataset")
parser.add_argument("--train_size", default=512, type=int, help="size of training data")
parser.add_argument("--save_path", default="", type=str, help="path to save figures")
parser.add_argument("--cuda_visible_devices", default="0", type=str, help="specify which GPU to run \
                    an application on")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
seed = 0

@jit
@partial(vmap, in_axes=(0, None, None, None, None, None, None, None))
def experiment(W_var, b_var, layers, ts, x_train, x_test, y_train, y_test):
    """
    Run the experiment for different weight variances simultaneously.
    """
    W_std = np.sqrt(W_var)
    b_std = np.sqrt(b_var)

    dlayers = np.concatenate((layers[1:] - layers[:-1], np.array([0,])))

    input_to_kernel = partial(stax.Dense(1024)[2], get=('cov1', 'nngp', 'cov2', 'ntk'))
    kernel_fn = wrap(stax.serial(stax.Dense(1024, W_std, b_std), stax.Erf())[2])

    def body_fn(kernels, dlayer):
        kdd, ktd = kernels

        # Make predictions for the current set of kernels at all the different times.
        lambda_max = np.linalg.eigh(kdd.ntk)[0][-1]
        eta = y_train.size * 2. / lambda_max
        predict_fn = nt.predict.gradient_descent_mse(kdd.ntk, y_train, eta, diag_reg=1e-4)
#         predict_fn = nt.predict.gradient_descent_mse(kdd.ntk, y_train, diag_reg=1e-4)
        predict_fn = partial(predict_fn, fx_test_0=0., k_test_train=ktd.ntk)
        train, test = vmap(predict_fn)(ts)

        # Compute the next kernel after iterating the map for dlayers.
        kdd = lax.fori_loop(0, dlayer, lambda _, k: kernel_fn(k), kdd)
        ktd = lax.fori_loop(0, dlayer, lambda _, k: kernel_fn(k), ktd)

        return (kdd, ktd), (accuracy_vmap(train, y_train), accuracy_vmap(test, y_test), test)

    kdd = input_to_kernel(x_train, x_train)
    ktd = input_to_kernel(x_test, x_train)
    
    return lax.scan(body_fn, (kdd, ktd), dlayers)[1]

def contour_plot(train_acc, test_acc, xi_1s, xi_stars, W_var, W_ordered, W_chaotic, 
                 dts, layers, dtype, save=True):
    """
    Depth scale of trainability/generalization in the ordered phase.
    :param train_acc: np.ndarray. Training accuracies for different weight variances and time step.
    :param test_acc: np.ndarray. Test accuracies for different weight variances and time step.
    :param xi_1s: np.ndarray. Depth scales of trainability/generalization in the ordered phase.
    :param xi_stars: np.ndarray. Depth scales for generaliztion of NTK in the chaotic phase.
            For more details, please refer to Disentangling Trainability and Generalization in 
            Deep Neural Networks (L. Xiao et al. 2020).
            Paper link: https://arxiv.org/pdf/1912.13053.pdf
    :param W_var: np.ndarray. Weight variances.
    :param W_ordered: np.ndarray. Weight variances belong to ordered phase.
    :param W_chaotic: np.ndarray. Weight variances belong to chaotic phase.
    :param dts: np.ndarray. Different time step t used to evaluate NTK training and test dynamics.
    :param layers: np.ndarray. Number of layers.
    :param dtype: string. Clean and/or poisoned data name, e.g. "Clean" or "NTGA".
    """
    train_levels = np.linspace(0, 1, 11)
    if args.dataset == "mnist":
        test_levels = np.linspace(0, 1, 11)
    elif args.dataset == "cifar10":
        test_levels = np.linspace(0, .36, 11)
    elif args.dataset == "imagenet":
        test_levels = np.linspace(0.5, 1, 11)
    depth_scaling = 8.
    ndts = len(dts)
    
    for i, dt in enumerate(dts):
        plt.subplot(2, ndts, i + 1)
        im = plt.contourf(W_var, layers, train_acc[i], train_levels)
        plt.plot(W_ordered, depth_scaling * xi_1s, 'w--', linewidth=3)
        plt.plot(W_chaotic, depth_scaling * xi_stars, 'w--', linewidth=3)
        plt.title('Train ; t={}'.format(dt), fontsize=14)
        if i == 0:
            plt.ylabel('$l$', fontsize=12)
        plt.ylim([0, layers[-1]])
        if i == ndts - 1:
            cax = make_axes_locatable(plt.gca()).append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

        plt.subplot(2, ndts, ndts + i + 1)
        im = plt.contourf(W_var, layers, test_acc[i], test_levels)
        plt.title('Test ; t={}'.format(dt), fontsize=14)
        plt.plot(W_ordered, depth_scaling * xi_1s, 'w--', linewidth=3)
        plt.plot(W_chaotic, depth_scaling * xi_stars, 'w--', linewidth=3)
        plt.xlabel('$\\sigma_w^2$', fontsize=12)
        if i == 0:
            plt.ylabel('$l$', fontsize=12)
        plt.ylim([0, layers[-1]])
        if i == ndts - 1:
            cax = make_axes_locatable(plt.gca()).append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

    plt.suptitle(dtype, fontsize=24, y=1.03)
    finalize_plot((ndts / 2, 1))
    if save:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        plt.savefig(fname='{:s}figure_{:s}_gp_learning_dynamics_{:s}.pdf'.format(args.save_path, args.dataset,
                                                                                 dtype.lower()), 
                    format="pdf", bbox_inches='tight')
    plt.show()
    
def main():
    # Prepare dataset
    print("Loading dataset...")
    if args.x_train_path and args.y_train_path:
        x_train_all = np.load(args.x_train_path)
        x_train_all = x_train_all.reshape(x_train_all.shape[0], -1)
        y_train_all = np.load(args.y_train_path)
        x_train_all, y_train_all = shaffle(x_train_all, y_train_all)
        x_val = np.load(args.x_val_path)
        x_val = x_val.reshape(x_val.shape[0], -1)
        y_val = np.load(args.y_val_path)
    else:
        x_train_all, y_train_all, _, _ = tuple(np.asarray(x) for x in get_dataset(args.dataset, None, None, flatten=True))
        x_train_all, y_train_all = shaffle(x_train_all, y_train_all)
        x_val = x_train_all[-10000:]
        y_val = y_train_all[-10000:]
        
    x_train = x_train_all[:args.train_size]
    y_train = y_train_all[:args.train_size]
    
    # Compute C-map and Q-map for chi_c (slope of critical point)
    # Use chi_c to find the theoretical trainable area
    c_map = lambda W_var, b_var: qc_map(W_var, b_var)[1]
    q_map = lambda W_var, b_var: qc_map(W_var, b_var)[0]
    q_star = lambda W_var, b_var: fixed_point(q_map(W_var, b_var), 1., 1e-7)
    c_star = lambda W_var, b_var: fixed_point(c_map(W_var, b_var), 0.5, 1e-7)
    chi = lambda c, W_var, b_var: grad(c_map(W_var, b_var))(c)
    chi_1 = partial(chi, 1.)
    chi_c = lambda W_var, b_var: grad(c_map(W_var, b_var))(c_star(W_var, b_var))
    
    # Run the experiment for different weight variances simultaneously.
    print("Runing the experiment for different weight variances...")
    # Experiment parameters.
    W_var = np.linspace(0.5, 3.0, 40)
    W_critical = 1.76
    b_var = 0.18
    dts = np.array([8**k for k in range(5)])
    dts = np.append(dts, np.inf)
    layers = np.array([i for i in range(10)] + [10 + 5*i for i in range(20)])

    # Train all of the infinite networks.
    train_acc, test_acc, test_pred = experiment(W_var, b_var, layers, dts, x_train, x_val, y_train, y_val)

    # Rearrange the axes so they go [time, depth, weight_variance].
    train_acc = np.transpose(train_acc, (2, 1, 0))
    test_acc = np.transpose(test_acc, (2, 1, 0))

    # Compute the depth scales.
    W_ordered = np.array([W for W in W_var if W < W_critical])
    W_chaotic = np.array([W for W in W_var if W > W_critical]) 
    xi_1s = xi_1(W_ordered, b_var, chi_1) 
    xi_stars = xi_star(W_chaotic, b_var, chi_1, chi_c) 
    
    # Plot contour-plot
    print("Plotting contour-plot...")
    contour_plot(train_acc, test_acc, xi_1s, xi_stars, W_var, W_ordered, W_chaotic, dts, layers, args.dtype)
    print("================== DONE ==================")
    
if __name__ == "__main__":
    main()
