#!/usr/bin/env python3
#
# Post-training view of loss, accuracy metrics

# Jinja2 tutorial:
# https://dev.to/goyder/automatic-reporting-in-python---part-1-from-planning-to-hello-world-32n1
# https://dev.to/goyder/automatic-reporting-in-python---part-2-from-hello-world-to-real-insights-8p3
# https://dev.to/goyder/automatic-reporting-in-python---part-3-packaging-it-up-1185


import argparse
import json
import pathlib
import sys

from contextlib import redirect_stderr
import jinja2
import os
# Mute Tensorflow chatter messages ('1' means filter out INFO messages.)
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
# Mute Keras chatter messages
with redirect_stderr(open(os.devnull, "w")):
    from keras.models import load_model
    from keras.utils import plot_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL


TEMPLATE_SEARCH_PATH = [pathlib.Path(__file__).parent / 'templates']
JINJA_ENV = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=TEMPLATE_SEARCH_PATH)
        )

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
            help="Directory containing all experiment data subdirectories."
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


def gen_data_plots(data_dir, diary_dir, train_data, global_data):
    best_epoch = train_data['best_epoch']

    # actually plot
    fig = plt.figure(figsize=(10,5))
    (ax1, ax2) = plot_loss_acc(fig, train_data['epochs'], train_data, global_data)

    # use annotate instead of arrow because so much easier to get good results
    ax1_scale = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    ax1.annotate('best=%.1f'%train_data['best_val_loss'],
            (best_epoch, train_data['best_val_loss']),
            (best_epoch, train_data['best_val_loss'] + .2*ax1_scale),
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment='center'
            )
    ax2_scale = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    ax2.annotate('best=%.1f'%train_data['best_val_acc_perc'],
            (best_epoch, train_data['best_val_acc_perc']),
            (best_epoch, train_data['best_val_acc_perc'] - .2*ax2_scale),
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment='center'
            )

    # save and display to computer
    fig.savefig(
            str(diary_dir / "training_metrics.png"),
            dpi=200,
            bbox_inches="tight"
            )


def gen_experiment_html(diary_dir, experiment, global_data):
    data_subdir = experiment['info']['datadir']
    train_data = experiment['train_data']
    experiment_info = experiment['info']

    model_name = experiment_info['model_name']
    job_id = experiment_info['job_id']

    diary_subdir = diary_dir / model_name
    diary_subdir.mkdir(parents=True, exist_ok=True)

    # make plot png
    gen_data_plots(data_subdir, diary_subdir, train_data, global_data)

    # make model structure png
    my_model = load_model(str(data_subdir / 'saved_models' / 'weights.best.hdf5'))
    # keras is using dpi=96 and PIL is using dpi=72, thus PIL rasterizes
    #   keras eps to 3/4 the size keras png
    # We multiply by 16/6 to get 2x size png for detail on retina screens
    # PIL resize allows us to anti-alias (PIL doesn't antialias rasterizing.)
    plot_model(
            my_model,
            to_file=str(diary_subdir / 'model.eps'),
            show_shapes=True
            )
    eps_model = PIL.Image.open(diary_subdir / 'model.eps')
    eps_model.load(scale=16)
    new_size = (int(eps_model.size[0]/6), int(eps_model.size[1]/6))
    eps_model = eps_model.resize(new_size, PIL.Image.LANCZOS)
    eps_model.save(diary_subdir / 'model.png')

    # create html report
    diary_entry_template = JINJA_ENV.get_template("diary_entry.html")
    diary_section_template = JINJA_ENV.get_template("diary_experiment_section.html")

    # find pixel-size of images
    plot_img = PIL.Image.open(diary_subdir / 'training_metrics.png')
    plot_img_size = [int(x/2) for x in plot_img.size]
    plot_img_size_str = "width:{0[0]}px;height:{0[1]}px;".format(plot_img_size)
    model_img = PIL.Image.open(diary_subdir / 'model.png')
    model_img_size = [int(x/2) for x in model_img.size]
    model_img_size_str = "width:{0[0]}px;height:{0[1]}px;".format(model_img_size)

    job = {}
    job['model_name'] = model_name
    job['job_id'] = job_id
    job['model_diagram_img'] = 'model.png'
    job['model_diagram_img_style'] = model_img_size_str
    job['model_metrics_img'] = 'training_metrics.png'
    job['model_metrics_img_style'] = plot_img_size_str
    job['best_val_acc_perc'] = '{0:.1f}'.format(train_data['best_val_acc_perc'])
    job['best_val_acc_epoch'] = train_data['best_epoch'] 

    report_path = diary_subdir / 'report.html'
    with report_path.open("w") as report_fh:
        report_fh.write(diary_entry_template.render(job=job))

    # convert all paths to relative to diary_dir
    job['model_diagram_img'] = (diary_subdir / 'model.png').relative_to(diary_dir)
    job['model_metrics_img'] = (diary_subdir / 'training_metrics.png').relative_to(diary_dir)
    
    # downsize images for closer to thumbnail size
    model_img_size = [int(x/2) for x in model_img_size]
    model_img_size_str = "width:{0[0]}px;height:{0[1]}px;".format(model_img_size)
    plot_img_size = [int(x/2) for x in plot_img_size]
    plot_img_size_str = "width:{0[0]}px;height:{0[1]}px;".format(plot_img_size)
    job['model_diagram_img_style'] = model_img_size_str
    job['model_metrics_img_style'] = plot_img_size_str

    report_path = report_path.relative_to(diary_dir)
    section = diary_section_template.render(
            job=job, report_path=report_path
            )
    return section


def catalog_dir(data_subdir):
    if data_subdir.name.startswith("data_"):
        job_name = "Local Job"
        datadir = data_subdir
    else:
        job_name = data_subdir.name
        try:
            datadir = data_subdir.glob('data_*')[0]
        except IndexError:
            return None

    model_name = data_subdir.name.lstrip("data_")

    # extract data
    train_data_path = data_subdir / 'train_history.json'
    with train_data_path.open("r") as train_data_fh:
        train_data = json.load(train_data_fh)
    train_data['epochs'] = range(1, len(train_data['acc'])+1)

    # convert data to numpy arrays
    for varname in train_data:
        train_data[varname] = np.array(train_data[varname])

    train_data['val_acc_perc'] = 100*train_data['val_acc']
    train_data['acc_perc'] = 100*train_data['acc']

    # find bests and maxs
    best_i = np.argmin(train_data['val_loss'])
    train_data['best_epoch'] = train_data['epochs'][best_i]
    train_data['best_val_loss'] = train_data['val_loss'][best_i]
    train_data['best_val_acc_perc'] = train_data['val_acc_perc'][best_i]

    train_data['max_loss'] = np.max(
            np.stack((train_data['val_loss'], train_data['loss']))
            )
    train_data['max_acc_perc'] = np.max(
            np.stack((train_data['val_acc_perc'], train_data['acc_perc']))
            )
    train_data['max_epoch'] = np.max(train_data['epochs'])

    experiment_info = {}
    experiment_info['model_name'] = model_name
    experiment_info['job_id'] = job_name
    experiment_info['datadir'] = datadir
    experiment_info['topdir'] = data_subdir

    return {'info':experiment_info, 'train_data':train_data}


def catalog_all_dirs(data_dir):
    # Find max loss over all datasets, so we can adjust all
    #   loss plots from 0 to max loss (consistent ylim for all plots)
    global_data = {}
    experiments = []
    for data_subdir in [x for x in data_dir.iterdir() if x.is_dir()]:
        data_subdir_data = catalog_dir(data_subdir)
        if data_subdir_data is None:
            continue
        experiments.append(data_subdir_data)
        global_data['max_loss'] = max(
                global_data.get('max_loss', 0),
                data_subdir_data['train_data']['max_loss']
                )
        global_data['max_acc_perc'] = max(
                global_data.get('max_acc_perc', 0),
                data_subdir_data['train_data']['max_acc_perc']
                )
        global_data['min_best_epoch'] = min(
                global_data.get('min_best_epoch', 1e6),
                data_subdir_data['train_data']['best_epoch']
                )

    return (experiments, global_data)


def main(argv=None):
    args = process_command_line(argv)
    data_dir = pathlib.Path(args.datadir)
    diary_dir = pathlib.Path(args.diarydir)

    (experiments, global_data) = catalog_all_dirs(data_dir)

    sections = []
    for experiment in experiments:
        sections.append(gen_experiment_html(diary_dir, experiment, global_data))

    # create summaries
    summaries = []
    diary_ranking_section_template = JINJA_ENV.get_template('diary_ranking_section.html')

    expts_val_acc_ranked = sorted(
            experiments,
            key=lambda x: x['train_data']['best_val_acc_perc'],
            reverse=True
            )
    expts_val_acc_out = [
            {
                'name':x['info']['model_name'],
                'criteria_value':"{0:.1f}%".format(x['train_data']['best_val_acc_perc'])
                }
            for x in expts_val_acc_ranked
            ]
    summaries.append(
            diary_ranking_section_template.render(
                criteria="Best Validation Accuracy",
                models=expts_val_acc_out
                )
            )

    expts_epoch_ranked = sorted(
            experiments,
            key=lambda x: x['train_data']['best_epoch'],
            reverse=False
            )
    expts_epoch_out = [
            {
                'name':x['info']['model_name'],
                'criteria_value':"Epoch {0}".format(x['train_data']['best_epoch'])
                }
            for x in expts_epoch_ranked
            ]
    summaries.append(
            diary_ranking_section_template.render(
                criteria="Quickest Training (Minimum Best Epoch)",
                models=expts_epoch_out
                )
            )


    # create html report
    master_diary = diary_dir / 'index.html'
    diary_index_template = JINJA_ENV.get_template("diary.html")
    with master_diary.open("w") as master_diary_fh:
        master_diary_fh.write(
                diary_index_template.render(
                    title='Data Diary for {0}'.format(data_dir.parent.resolve()),
                    summaries=summaries,
                    experiments=sections
                    )
                )

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
