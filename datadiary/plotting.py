#!/usr/bin/env python3
#
# Post-training view of loss, accuracy metrics

# Jinja2 tutorial:
# https://dev.to/goyder/automatic-reporting-in-python---part-1-from-planning-to-hello-world-32n1
# https://dev.to/goyder/automatic-reporting-in-python---part-2-from-hello-world-to-real-insights-8p3
# https://dev.to/goyder/automatic-reporting-in-python---part-3-packaging-it-up-1185


import os
import os.path

import matplotlib
# png-generation only, no interactive GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def gen_data_plots(diary_dir, train_data, global_data):
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
            str(diary_dir / "training_metrics.png"),
            dpi=200,
            bbox_inches='tight'
            )
    # Theoretically we are using the "object-oriented" matplotlib approach,
    #   but somehow all figures are kept around until we close them!
    # So close figure each time after we save the image.
    plt.close()
