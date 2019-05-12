#!/usr/bin/env python3
#
# Post-training view of loss, accuracy metrics

# Jinja2 tutorial:
# https://dev.to/goyder/automatic-reporting-in-python---part-1-from-planning-to-hello-world-32n1
# https://dev.to/goyder/automatic-reporting-in-python---part-2-from-hello-world-to-real-insights-8p3
# https://dev.to/goyder/automatic-reporting-in-python---part-3-packaging-it-up-1185


import os
from contextlib import redirect_stderr

import matplotlib
# png-generation only, no interactive GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Mute Tensorflow chatter messages ('1' means filter out INFO messages.)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Mute Keras chatter messages
with redirect_stderr(open(os.devnull, "w")):
    import keras

# custom version of keras.utils.vis_utils to get dpi argument and transparency
import datadiary.keras_vis_utils


def plot_vs_epoch(ax, epochs, train=None, val=None, do_legend=True):
    if train is not None:
        ax.plot(epochs, train, '.-', label='training')
    if val is not None:
        ax.plot(epochs, val, '.-', label='validation')
    if do_legend:
        ax.legend()


def plot_loss(ax, epochs, data_dict, overall_max_loss):
    # ax belongs to fig
    ax.set_title("Loss During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plot_vs_epoch(
            ax,
            epochs,
            data_dict.get('loss', []),
            data_dict.get('val_loss', [])
            )
    ax.grid()
    ax.set_ylim(0, overall_max_loss)


def plot_acc(ax, epochs, data_dict):
    # ax belongs to fig
    ax.set_title("Accuracy During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    plot_vs_epoch(
            ax,
            epochs,
            data_dict.get('acc_perc', []),
            data_dict.get('val_acc_perc', [])
            )
    ax.grid()
    ax.set_ylim(0, 100)


def plot_loss_acc(fig, epochs, train_data, global_data):
    plt.subplots_adjust(
            left=0.08,
            bottom=None,
            right=0.92,
            top=None,
            wspace=None,
            hspace=None,
            )
    ax1 = fig.add_subplot(121)
    plot_loss(ax1, epochs, train_data, global_data['max_loss'])
    ax2 = fig.add_subplot(122)
    plot_acc(ax2, epochs, train_data)
    return(ax1, ax2)


def gen_data_plots(output_file, train_data, global_data):
    best_epoch = train_data['best_epoch']

    # actually plot
    fig = plt.figure(figsize=(10, 5))
    (ax1, ax2) = plot_loss_acc(fig, train_data['epochs'], train_data, global_data)

    # use annotate instead of arrow because so much easier to get good results
    ax1_yscale = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    ax1.annotate('best=%.1f'%train_data['best_val_loss'],
            (best_epoch, train_data['best_val_loss']),
            (best_epoch, train_data['best_val_loss'] + .1*ax1_yscale),
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment='center'
            )
    ax2_yscale = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    ax2.annotate('best=%.1f'%train_data['best_val_acc_perc'],
            (best_epoch, train_data['best_val_acc_perc']),
            (best_epoch, train_data['best_val_acc_perc'] - .1*ax2_yscale),
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment='center'
            )

    # save and display to computer
    fig.savefig(
            str(output_file),
            dpi=200,
            bbox_inches='tight'
            )
    # Theoretically we are using the "object-oriented" matplotlib approach,
    #   but somehow all figures are kept around until we close them!
    # So close figure each time after we save the image.
    plt.close()


def plot_model(data_subdir, output_diagram_file):
    """Plotting model diagram and extracting model info
    Everything that needs a Keras model in this function

    Args:
        data_subdir (pathlib.Path): specific dir containint saved_models dir
        diary_subdir (pathlib.Path): output diary subdir (for model diagram png)
    """
    # make model structure png
    best_weights_file = data_subdir / 'saved_models' / 'weights.best.hdf5'
    if not best_weights_file.is_file():
        best_weights_file = list((data_subdir / 'saved_models').glob('*.hdf5'))[0]
    my_model = keras.models.load_model(str(best_weights_file))

    # Use dpi=192 for 2x size.
    # Size image in html down by 1/2x to get same size with 2x dpi.
    datadiary.keras_vis_utils.plot_model(
            my_model,
            to_file=str(output_diagram_file),
            show_shapes=True,
            dpi=192,
            transparent_bg=True
            )

    del my_model
    # NEED TO DO THIS or else memory leak and slowdown happen
    #   (keras 2.2.4, tensorflow 1.13.1)
    #   https://github.com/keras-team/keras/issues/2102
    keras.backend.clear_session()
