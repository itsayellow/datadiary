#!/usr/bin/env python3
#
# Post-training view of loss, accuracy metrics


import argparse
import json
import pathlib
import sys

from contextlib import redirect_stderr
import jinja2
import os
with redirect_stderr(open(os.devnull, "w")):
    from keras.models import load_model
    from keras.utils import plot_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def process_command_line(argv):
    """Process command line invocation arguments and switches.

    Args:
        argv: list of arguments, or `None` from ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: named attributes of arguments and switches
    """
    #script_name = argv[0]
    argv = argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(
            description="Plot training metrics.")

    # specifying nargs= puts outputs of parser in list (even if nargs=1)

    # required arguments
    parser.add_argument('datadir',
            help="Directory containing train_history.json."
            )
    parser.add_argument('diarydir',
            help="Directory for output diary entries."
            )

    # switches/options:
    #parser.add_argument(
    #    '-s', '--max_size', action='store',
    #    help='String specifying maximum size of images.  ' \
    #            'Larger images will be resized. (e.g. "1024x768")')
    #parser.add_argument(
    #    '-o', '--omit_hidden', action='store_true',
    #    help='Do not copy picasa hidden images to destination directory.')

    args = parser.parse_args(argv)

    return args


def plot_vs_epoch(ax, epochs, train=None, val=None, do_legend=True):
    if train is not None:
        ax.plot(epochs, train, '.-', label='training')
    if val is not None:
        ax.plot(epochs, val, '.-', label='validation')
    if do_legend:
        ax.legend()


def plot_loss(ax, epochs, data_dict):
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


def pyplot_quick_dirty(train_data):
    epochs = range(1, len(train_data['loss'])+1)
    plt.figure(num=1, figsize=(10,5))

    plt.subplot(121)
    plt.plot(epochs, train_data['loss'], label='loss')
    plt.plot(epochs, train_data['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(epochs, 100*np.array(train_data['acc']), label='acc')
    plt.plot(epochs, 100*np.array(train_data['val_acc']), label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()


def plot_loss_acc(fig, epochs, train_data):
    plt.subplots_adjust(
            left=0.08,
            bottom=None,
            right=0.92,
            top=None,
            wspace=None,
            hspace=None,
            )
    ax1 = fig.add_subplot(121)
    plot_loss(ax1, epochs, train_data)
    ax2 = fig.add_subplot(122)
    plot_acc(ax2, epochs, train_data)
    return(ax1, ax2)


def gen_data_plots(data_dir, diary_dir, train_data):
    loss_scale = max(train_data['val_loss']) - min(train_data['val_loss'])
    acc_perc_scale = max(train_data['val_acc_perc']) - min(train_data['val_acc_perc'])
    best_epoch = train_data['best_epoch']

    # actually plot
    fig = plt.figure(num=1, figsize=(10,5))
    (ax1, ax2) = plot_loss_acc(fig, train_data['epochs'], train_data)

    # use annotate instead of arrow because so much easier to get good results
    ax1.annotate('best=%.1f'%train_data['best_val_loss'],
            (best_epoch, train_data['best_val_loss']),
            (best_epoch, train_data['best_val_loss'] + .2*loss_scale),
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment='center'
            )
    ax2.annotate('best=%.1f'%train_data['best_val_acc_perc'],
            (best_epoch, train_data['best_val_acc_perc']),
            (best_epoch, train_data['best_val_acc_perc'] - .2*acc_perc_scale),
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment='center'
            )

    # save and display to computer
    fig.savefig(str(diary_dir / "training_metrics.png"), bbox_inches="tight")


def proces_data_dir(data_subdir, diary_dir, model_name):
    if data_subdir.name.startswith("data_"):
        job_id = "Local Job"
    else:
        job_id = job_dir.name
    diary_subdir = diary_dir / model_name
    diary_subdir.mkdir(parents=True, exist_ok=True)

    # extract data
    train_data_path = data_subdir / 'train_history.json'
    with train_data_path.open("r") as train_data_fh:
        train_data = json.load(train_data_fh)
    train_data['val_acc_perc'] = 100*np.array(train_data['val_acc'])
    train_data['acc_perc'] = 100*np.array(train_data['acc'])

    # find best val_loss
    train_data['epochs'] = range(1, len(train_data['acc'])+1)
    best_i = np.argmin(np.array(train_data['val_loss']))
    train_data['best_epoch'] = best_i + 1
    train_data['best_val_loss'] = train_data['val_loss'][best_i]
    train_data['best_val_acc_perc'] = train_data['val_acc_perc'][best_i]

    # make plot png
    gen_data_plots(data_subdir, diary_subdir, train_data)

    # make model structure png
    my_model = load_model(str(data_subdir / 'saved_models' / 'weights.best.hdf5'))
    plot_model(my_model, to_file=str(diary_subdir / 'model.png'), show_shapes=True)

    # create html report
    env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath="templates")
            )
    diary_entry_template = env.get_template("diary_entry.html")

    job = {}
    job['model_name'] = model_name
    job['job_id'] = job_id
    job['model_diagram_img'] = 'model.png'
    job['model_metrics_img'] = 'training_metrics.png'
    job['best_val_acc_perc'] = '{0:.1f}'.format(train_data['best_val_acc_perc'])
    job['best_val_acc_epoch'] = train_data['best_epoch'] 

    with (diary_subdir / 'report.html').open("w") as report_fh:
        report_fh.write(diary_entry_template.render(job=job))


def main(argv=None):
    args = process_command_line(argv)

    job_dir = pathlib.Path(args.datadir)
    if job_dir.name.startswith("data_"):
        data_subdir = job_dir
    else:
        data_subdir = job_dir.glob('data_*')[0]

    model_name = data_subdir.name.lstrip("data_")
    diary_dir = pathlib.Path(args.diarydir)
    proces_data_dir(data_subdir, diary_dir, model_name)

    return 0


if __name__ == "__main__":
    try:
        status = main(sys.argv)
    except KeyboardInterrupt:
        # Make a very clean exit (no debug info) if user breaks with Ctrl-C
        print("Stopped by Keyboard Interrupt", file=sys.stderr)
        # exit error code for Ctrl-C
        status = 130

    sys.exit(status)
