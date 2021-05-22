# +
import os
import argparse
import jax.numpy as np
from jax.api import grad, jit, vmap
from jax import random
from jax.config import config
config.update('jax_enable_x64', True)
from neural_tangents import stax

from models.dnn_infinite import DenseGroup
from models.cnn_infinite import ConvGroup
from attacks.projected_gradient_descent import projected_gradient_descent
from utils import *
from utils_jax import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generate NTGA attack!")
parser.add_argument("--model_type", default="fnn", type=str, help="surrogate model. Choose either `fnn` or `cnn`")
parser.add_argument("--dataset", required=True, type=str, help="dataset. `mnist`, `cifar10`, and `imagenet`\
                    are available. For ImageNet or other dataset, please modify the path in the code directly.")
parser.add_argument("--val_size", default=10000, type=int, help="size of validation data")
parser.add_argument("--t", default=64, type=int, help="time step used to compute poisoned data")
parser.add_argument("--eps", required=True, type=float, help="epsilon. Strength of NTGA")
parser.add_argument("--nb_iter", default=10, type=int, help="number of iteration used to generate poisoned data")
parser.add_argument("--block_size", default=512, type=int, help="block size of B-NTGA")
parser.add_argument("--batch_size", default=30, type=int, help="batch size")
parser.add_argument("--save_path", default="", type=str, help="path to save figures")
parser.add_argument("--cuda_visible_devices", default="0", type=str, help="specify which GPU to run \
                    an application on")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices

if args.model_type == "fnn":
    flatten = True
else:
    flatten = False

# Epsilon, attack iteration, and step size
if args.dataset == "mnist":
    num_classes = 10
    train_size = 60000 - args.val_size
elif args.dataset == "cifar10":
    num_classes = 10
    train_size = 50000 - args.val_size
elif args.dataset == "imagenet":
    num_classes = 2
    train_size = 2220
    print("For ImageNet, please specify the file path manually.")
else:
    raise ValueError("To load custom dataset, please modify the code directly.")
eps_iter = (args.eps/args.nb_iter)*1.1

seed = 0
    
def surrogate_fn(model_type, W_std, b_std, num_classes):
    """
    :param model_type: string. `fnn` or `cnn`.
    :param W_std: float. Standard deviation of weights at initialization.
    :param b_std: float. Standard deviation of biases at initialization.
    :param num_classes: int. Number of classes in the classification task.
    :return: triple of callable functions (init_fn, apply_fn, kernel_fn).
            In Neural Tangents, a network is defined by a triple of functions (init_fn, apply_fn, kernel_fn). 
            init_fn: a function which initializes the trainable parameters.
            apply_fn: a function which computes the outputs of the network.
            kernel_fn: a kernel function of the infinite network (GP) of the given architecture 
                    which computes the kernel matrix
    """
    if model_type == "fnn":
        init_fn, apply_fn, kernel_fn = stax.serial(DenseGroup(5, 512, W_std, b_std))
    elif model_type == "cnn":
        if args.dataset == 'imagenet':
            init_fn, apply_fn, kernel_fn = stax.serial(ConvGroup(2, 64, (3, 3), W_std, b_std),
                                                       stax.Flatten(),
                                                       stax.Dense(384, W_std, b_std),
                                                       stax.Dense(192, W_std, b_std),
                                                       stax.Dense(num_classes, W_std, b_std))
        else:
            init_fn, apply_fn, kernel_fn = stax.serial(ConvGroup(2, 64, (2, 2), W_std, b_std),
                                                       stax.Flatten(),
                                                       stax.Dense(384, W_std, b_std),
                                                       stax.Dense(192, W_std, b_std),
                                                       stax.Dense(num_classes, W_std, b_std))
    else:
        raise ValueError
    return init_fn, apply_fn, kernel_fn
    
def model_fn(kernel_fn, x_train=None, x_test=None, fx_train_0=0., fx_test_0=0., t=None, y_train=None, diag_reg=1e-4):
    """
    :param kernel_fn: a callable that takes an input tensor and returns the kernel matrix.
    :param x_train: input tensor (training data).
    :param x_test: input tensor (test data; used for evaluation).
    :param y_train: Tensor with one-hot true labels of training data.
    :param fx_train_0 = output of the network at `t == 0` on the training set. `fx_train_0=None`
            means to not compute predictions on the training set. fx_train_0=0. for infinite width.
    :param fx_test_0 = output of the network at `t == 0` on the test set. `fx_test_0=None`
            means to not compute predictions on the test set. fx_test_0=0. for infinite width.
            For more details, please refer to equations (10) and (11) in Wide Neural Networks of 
            Any Depth Evolve as Linear Models Under Gradient Descent (J. Lee and L. Xiao et al. 2019). 
            Paper link: https://arxiv.org/pdf/1902.06720.pdf.
    :param t: a scalar of array of scalars of any shape. `t=None` is treated as infinity and returns 
            the same result as `t=np.inf`, but is computed using identity or linear solve for train 
            and test predictions respectively instead of eigendecomposition, saving time and precision.
            Equivalent of training steps (but can be fractional).
    :param diag_reg: (optional) a scalar representing the strength of the diagonal regularization for `k_train_train`, 
            i.e. computing `k_train_train + diag_reg * I` during Cholesky factorization or eigendecomposition.
    :return: a np.ndarray for the model logits.
    """
    # Kernel
    ntk_train_train = kernel_fn(x_train, x_train, 'ntk')
    ntk_test_train = kernel_fn(x_test, x_train, 'ntk')
    
    # Prediction
    predict_fn = nt.predict.gradient_descent_mse(ntk_train_train, y_train, diag_reg=diag_reg)
    return predict_fn(t, fx_train_0, fx_test_0, ntk_test_train)

def adv_loss(x_train, x_test, y_train, y_test, kernel_fn, loss='mse', t=None, targeted=False, diag_reg=1e-4):
    """
    :param x_train: input tensor (training data).
    :param x_test: input tensor (test data; used for evaluation).
    :param y_train: Tensor with one-hot true labels of training data.
    :param y_test: Tensor with one-hot true labels of test data. If targeted is true, then provide the
            target one-hot label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting poisoned data. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None. This argument does not have
            to be a binary one-hot label (e.g., [0, 1, 0, 0]), it can be floating points values
            that sum up to 1 (e.g., [0.05, 0.85, 0.05, 0.05]).
    :param kernel_fn: a callable that takes an input tensor and returns the kernel matrix.
    :param loss: loss function.
    :param t: a scalar of array of scalars of any shape. `t=None` is treated as infinity and returns 
            the same result as `t=np.inf`, but is computed using identity or linear solve for train 
            and test predictions respectively instead of eigendecomposition, saving time and precision.
            Equivalent of training steps (but can be fractional).
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
    :param diag_reg: (optional) a scalar representing the strength of the diagonal regularization for `k_train_train`, 
            i.e. computing `k_train_train + diag_reg * I` during Cholesky factorization or eigendecomposition.
    :return: a float for loss.
    """
    # Kernel
    ntk_train_train = kernel_fn(x_train, x_train, 'ntk')
    ntk_test_train = kernel_fn(x_test, x_train, 'ntk')
    
    # Prediction
    predict_fn = nt.predict.gradient_descent_mse(ntk_train_train, y_train, diag_reg=diag_reg)
    fx = predict_fn(t, 0., 0., ntk_test_train)[1]
    
    # Loss
    if loss == 'cross-entropy':
        loss = cross_entropy_loss(fx, y_test)
    elif loss == 'mse':
        loss = mse_loss(fx, y_test)
        
    if targeted:
        loss = -loss        
    return loss
    
def main():
    # Prepare dataset
    # For ImageNet, please specify the file path manually
    print("Loading dataset...")
    x_train_all, y_train_all, x_test, y_test = tuple(np.array(x) for x in get_dataset(args.dataset, None, None, flatten=flatten))
    x_train_all, y_train_all = shaffle(x_train_all, y_train_all, seed)
    x_train, x_val = x_train_all[:train_size], x_train_all[train_size:train_size+args.val_size]
    y_train, y_val = y_train_all[:train_size], y_train_all[train_size:train_size+args.val_size]
    
    # Build model
    print("Building model...")
    key = random.PRNGKey(0)
    b_std, W_std = np.sqrt(0.18), np.sqrt(1.76) # Standard deviation of initial biases and weights
    init_fn, apply_fn, kernel_fn = surrogate_fn(args.model_type, W_std, b_std, num_classes)
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    
    # grads_fn: a callable that takes an input tensor and a loss function, 
    # and returns the gradient w.r.t. an input tensor.
    grads_fn = jit(grad(adv_loss, argnums=0), static_argnums=(4, 5, 7))
    
    # Generate Neural Tangent Generalization Attacks (NTGA)
    print("Generating NTGA....")
    epoch = int(x_train.shape[0]/args.block_size)
    x_train_adv = []
    y_train_adv = []
    for idx in tqdm(range(epoch)):
        _x_train = x_train[idx*args.block_size:(idx+1)*args.block_size]
        _y_train = y_train[idx*args.block_size:(idx+1)*args.block_size]
        _x_train_adv = projected_gradient_descent(model_fn=model_fn, kernel_fn=kernel_fn, grads_fn=grads_fn, 
                                                  x_train=_x_train, y_train=_y_train, x_test=x_val, y_test=y_val, 
                                                  t=args.t, loss='cross-entropy', eps=args.eps, eps_iter=eps_iter, 
                                                  nb_iter=args.nb_iter, clip_min=0, clip_max=1, batch_size=args.batch_size)

        x_train_adv.append(_x_train_adv)
        y_train_adv.append(_y_train)

        # Performance of clean and poisoned data
        _, y_pred = model_fn(kernel_fn=kernel_fn, x_train=_x_train, x_test=x_test, y_train=_y_train)
        print("Clean Acc: {:.2f}".format(accuracy(y_pred, y_test)))
        _, y_pred = model_fn(kernel_fn=kernel_fn, x_train=x_train_adv[-1], x_test=x_test, y_train=y_train_adv[-1])
        print("NTGA Robustness: {:.2f}".format(accuracy(y_pred, y_test)))
    
    # Save poisoned data
    x_train_adv = np.concatenate(x_train_adv)
    y_train_adv = np.concatenate(y_train_adv)
    
    if args.dataset == "mnist":
        x_train_adv = x_train_adv.reshape(-1, 28, 28, 1)
    elif args.dataset == "cifar10":
        x_train_adv = x_train_adv.reshape(-1, 32, 32, 3)
    elif args.dataset == "imagenet":
        x_train_adv = x_train_adv.reshape(-1, 224, 224, 3)
    else:
        raise ValueError("Please specify the image size manually.")
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    np.save('{:s}x_train_{:s}_ntga_{:s}.npy'.format(args.save_path, args.dataset, args.model_type), x_train_adv)
    np.save('{:s}y_train_{:s}_ntga_{:s}.npy'.format(args.save_path, args.dataset, args.model_type), y_train_adv)
    print("================== Successfully generate NTGA! ==================")

if __name__ == "__main__":
    main()
