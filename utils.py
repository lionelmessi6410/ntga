import os
import numpy as onp
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


def format_plot(title='', x='', y='', grid=True):  
    ax = plt.gca()

    plt.grid(grid)
    if title:
        plt.title(title, fontsize=26)
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)

def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
    plt.tight_layout()

def plot_images(image, shape=None, num_row=2, num_col=5, scale=1, row_title=None, fname=None, save=False):
    # plot images
    if shape is not None:
        image = onp.reshape(image, shape)
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*scale*num_col,2*scale*num_row))
    for i in range(num_row*num_col):
        if num_row == 1:
            ax = axes[i%num_col]
        else:
            ax = axes[i//num_col, i%num_col]
        if len(image.shape) == 3:
            ax.imshow(image[i], cmap='gray', vmin=0, vmax=1)
        elif len(image.shape) == 4:
            ax.imshow(image[i], vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(fname=fname+'.pdf', format="pdf", bbox_inches='tight')
    plt.show()


def plot_visualization(image, shape=None, num_row=3, num_col=5, scale=1, row_title=None, fname=None, save=False):
    # plot images
    if shape is not None:
        image = onp.reshape(image, shape)
        
    fig, big_axes = plt.subplots(num_row, 1, figsize=(1.5*scale*num_col, 2*scale*num_row), sharey=True)
    
    # Subplot rows
    for i, big_ax in enumerate(big_axes):
        big_ax.set_title(row_title[i], fontsize=16*scale, y=0.84)
        
        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.axis('off')
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False
    
    # Subplot columns
    for i in range(num_row*num_col):
        ax = fig.add_subplot(num_row, num_col, i+1)
        if len(image.shape) == 3:
            ax.imshow(image[i], cmap='gray', vmin=0, vmax=1)
        elif len(image.shape) == 4:
            ax.imshow(image[i], vmin=0, vmax=1)
        ax.axis('off')
#     fig.set_facecolor('w')
    plt.tight_layout()
    if save:
        plt.savefig(fname=fname+'.pdf', format="pdf", bbox_inches='tight')
    plt.show()


def normalize(x_train, y_train, x_test, y_test):
    x_train = x_train / onp.sqrt(onp.reshape(onp.einsum('ij,ij->i', x_train, x_train), (64, 1))) * onp.sqrt(x_train.shape[-1])
    y_train = y_train - onp.mean(y_train, axis=0, keepdims=True)
    x_test = x_test / onp.sqrt(onp.reshape(onp.einsum('ij,ij->i', x_test, x_test), (32, 1))) * onp.sqrt(x_test.shape[-1])
    return x_train, y_train, x_test, y_test

# Data Loading
def _partial_flatten_and_normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    x = onp.reshape(x, (x.shape[0], -1))
    return (x - onp.mean(x)) / onp.std(x)

def _flatten(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return onp.reshape(x, (x.shape[0], -1))/255

def _normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return x / 255


def _one_hot(x, k, dtype=onp.float32):
    """Create a one-hot encoding of x of size k."""
    return onp.array(x[:, None] == onp.arange(k), dtype)


def get_dataset(name, n_train=None, n_test=None, permute_train=False, flatten=False, normalize=False):
    """Download, parse and process a dataset to unit scale and one-hot labels."""
    ds_builder = tfds.builder(name)
    ds_train, ds_test = tfds.as_numpy(
        tfds.load(
            name + ':3.*.*',
            split=['train' + ('[:%d]' % n_train if n_train is not None else ''),
                   'test' + ('[:%d]' % n_test if n_test is not None else '')],
            batch_size=-1,
            as_dataset_kwargs={'shuffle_files': False}))

    train_images, train_labels, test_images, test_labels = (ds_train['image'],
                                                            ds_train['label'],
                                                            ds_test['image'],
                                                            ds_test['label'])
    num_classes = ds_builder.info.features['label'].num_classes
    
    if flatten and normalize:
        train_images = _partial_flatten_and_normalize(train_images)
        test_images = _partial_flatten_and_normalize(test_images)
    elif flatten:
        train_images = _flatten(train_images)
        test_images = _flatten(test_images)
    else:
        train_images = _normalize(train_images)
        test_images = _normalize(test_images)
        
    train_labels = _one_hot(train_labels, num_classes)
    test_labels = _one_hot(test_labels, num_classes)

    if permute_train:
        perm = onp.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


def shaffle(images, labels, seed=None):
    perm = onp.random.RandomState(seed).permutation(images.shape[0])
    images = images[perm]
    labels = labels[perm]
    return images, labels


def accuracy(y_pred, y_test):
    """
    This function calculates the accuracy of mean prediction of Gaussian Process
    :param y_pred: np.ndarray. Prediction of Gaussian Process.
    :param y_test: np.ndarray. Ground truth label.
    :return: a float for accuracy.
    """
    return onp.mean(onp.argmax(y_pred, axis=-1) == onp.argmax(y_test, axis=-1))
