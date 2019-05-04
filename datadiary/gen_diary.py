#!/usr/bin/env python3
#
# Post-training view of loss, accuracy metrics

# Jinja2 tutorial:
# https://dev.to/goyder/automatic-reporting-in-python---part-1-from-planning-to-hello-world-32n1
# https://dev.to/goyder/automatic-reporting-in-python---part-2-from-hello-world-to-real-insights-8p3
# https://dev.to/goyder/automatic-reporting-in-python---part-3-packaging-it-up-1185


import argparse
import datetime
import json
import os
import pathlib
import sys
import warnings
from contextlib import redirect_stderr

import imagesize
import jinja2
import tqdm
# Mute Tensorflow chatter messages ('1' means filter out INFO messages.)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Mute Keras chatter messages
with redirect_stderr(open(os.devnull, "w")):
    from keras.models import load_model
#from keras.utils.generic_utils import serialize_keras_object
import numpy as np
import matplotlib
# png-generation only, no interactive GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# custom version of keras.utils.vis_utils to get dpi argument
from datadiary.keras_vis_utils import plot_model


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
    parser.add_argument('datadir', nargs='+',
            help="Directory containing all experiment data subdirectories."
            )

    # switches/options:
    parser.add_argument(
            '-d', '--diary', action='store', default='diary',
            help='Directory to put diary html doc tree.'
            )
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


def render_experiment_html(diary_dir, experiment, global_data):
    data_subdir = experiment['info']['datadir']
    train_data = experiment['train_data']

    diary_subdir = diary_dir / experiment['info']['topdirname']
    diary_subdir.mkdir(parents=True, exist_ok=True)

    # make plot png
    gen_data_plots(diary_subdir, train_data, global_data)

    # make model structure png
    best_weights_file = data_subdir / 'saved_models' / 'weights.best.hdf5'
    if not best_weights_file.is_file():
        best_weights_file = list((data_subdir / 'saved_models').glob('*.hdf5'))[0]
    my_model = load_model(str(best_weights_file))

    # get model info
    #new_model_opt = serialize_keras_object(my_model.optimizer)
    model_opt_name = my_model.optimizer.__class__.__module__ + \
            "." + my_model.optimizer.__class__.__name__
    model_opt_config = my_model.optimizer.get_config()
    model_opt_config_fmt = [
            "{0}: {1:.3g}".format(x, model_opt_config[x]) for x in model_opt_config
            ]
    model_opt_str = ", ".join(
            [
            "{0}={1:.3g}".format(x, model_opt_config[x]) for x in model_opt_config
                ]
            )
    model_loss_type = my_model.loss
    #model_metrics = my_model.metrics

    # Use dpi=192 for 2x size.
    # Size image in html down by 1/2x to get same size with 2x dpi.
    plot_model(
            my_model,
            to_file=str(diary_subdir / 'model.png'),
            show_shapes=True,
            dpi=192
            )

    # create html report
    diary_entry_template = JINJA_ENV.get_template("diary_entry.html")
    diary_section_template = JINJA_ENV.get_template("diary_experiment_section.html")

    # find pixel-size of images
    plot_img_size = imagesize.get(diary_subdir / 'training_metrics.png')
    plot_img_size = [int(x/2) for x in plot_img_size]
    plot_img_size_str = "width:{0[0]}px;height:{0[1]}px;".format(plot_img_size)
    model_img_size = imagesize.get(diary_subdir / 'model.png')
    model_img_size = [int(x/2) for x in model_img_size]
    model_img_size_str = "width:{0[0]}px;height:{0[1]}px;".format(model_img_size)

    job = {}
    job['model_name'] = experiment['info']['model_name']
    job['job_id'] = experiment['info']['job_id']
    job['topdir'] = experiment['info']['topdir']
    job['model_diagram_img'] = 'model.png'
    job['model_diagram_img_style'] = model_img_size_str
    job['model_metrics_img'] = 'training_metrics.png'
    job['model_metrics_img_style'] = plot_img_size_str
    job['best_val_acc_perc'] = '{0:.1f}'.format(train_data['best_val_acc_perc'])
    job['best_val_acc_epoch'] = train_data['best_epoch']
    job['model_optimizer_name'] = model_opt_name
    job['model_optimizer_config'] = model_opt_config_fmt
    job['model_optimizer_config_str'] = model_opt_str
    job['model_loss'] = model_loss_type

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
            datadir = list(data_subdir.glob('data_*'))[0]
        except IndexError:
            return None

    model_name = datadir.name.lstrip("data_")

    # extract training data
    train_data_path = datadir / 'train_history.json'
    if not train_data_path.is_file():
        return None
    with train_data_path.open("r") as train_data_fh:
        train_data = json.load(train_data_fh)
    train_data['epochs'] = range(1, len(train_data['acc'])+1)

    # extract test data
    test_data_path = datadir / 'test.json'
    try:
        with test_data_path.open("r") as test_data_fh:
            test_data = json.load(test_data_fh)
    except IOError:
        test_data = {}

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
    experiment_info['topdirname'] = data_subdir.name

    return {'info':experiment_info, 'train_data':train_data, 'test_data':test_data}


def catalog_all_dirs(data_dirs):
    # Find max loss over all datasets, so we can adjust all
    #   loss plots from 0 to max loss (consistent ylim for all plots)
    global_data = {}
    experiments = []
    for data_dir in data_dirs:
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


def create_ranking(experiments, title, sort_key, reverse, info_dict_create):
    diary_ranking_section_template = JINJA_ENV.get_template('diary_ranking_section.html')

    if not experiments:
        return ""

    expts_val_acc_ranked = sorted(
            experiments,
            key=sort_key,
            reverse=reverse
            )
    expts_val_acc_out = [
            info_dict_create(x)
            for x in expts_val_acc_ranked
            ]
    return diary_ranking_section_template.render(
            criteria=title,
            models=expts_val_acc_out
            )


def main(argv=None):
    args = process_command_line(argv)
    data_dirs = [pathlib.Path(dir) for dir in args.datadir]
    diary_dir = pathlib.Path(args.diary)

    (experiments, global_data) = catalog_all_dirs(data_dirs)
    # sort by validation accuracy
    experiments.sort(key=lambda x: x['train_data']['best_val_acc_perc'], reverse=True)
    experiments_subtitle = '(Sorted by Validation Accuracy)'

    print("Diary output to: {0}".format(diary_dir))
    print("Rendering HTML summaries of all jobs...")
    sections = []
    for experiment in tqdm.tqdm(experiments, leave=False, unit='job'):
        sections.append(
                render_experiment_html(diary_dir, experiment, global_data)
                )

    # create summaries
    summaries = []

    # tolerate empty test_data dict in data dirs
    summaries.append(
            create_ranking(
                [exp for exp in experiments if 'test_acc_perc' in exp['test_data']],
                title="Best Test Accuracy",
                sort_key=lambda x: x['test_data']['test_acc_perc'],
                reverse=True,
                info_dict_create=lambda x: {
                    'name':x['info']['model_name'],
                    'topdir':x['info']['topdir'],
                    'criteria_value':"{0:.1f}%".format(x['test_data']['test_acc_perc'])
                    },
                )
            )
    summaries.append(
            create_ranking(
                experiments,
                title="Best Validation Accuracy",
                sort_key=lambda x: x['train_data']['best_val_acc_perc'],
                reverse=True,
                info_dict_create=lambda x: {
                    'name':x['info']['model_name'],
                    'topdir':x['info']['topdir'],
                    'criteria_value':"{0:.1f}%".format(x['train_data']['best_val_acc_perc'])
                    },
                )
            )
    summaries.append(
            create_ranking(
                experiments,
                title="Quickest Training (Minimum Best Epoch)",
                sort_key=lambda x: x['train_data']['best_epoch'],
                reverse=False,
                info_dict_create=lambda x: {
                    'name':x['info']['model_name'],
                    'topdir':x['info']['topdir'],
                    'criteria_value':"Epoch {0}".format(x['train_data']['best_epoch'])
                    },
                )
            )

    # create index/summary html report
    master_diary = diary_dir / 'index.html'
    diary_index_template = JINJA_ENV.get_template("diary.html")
    datetime_generated = datetime.datetime.now().strftime("%Y-%m-%d %a %I:%M%p")
    data_dirs_str = ", ".join([str(d.parent.resolve()) for d in data_dirs])
    with master_diary.open("w") as master_diary_fh:
        master_diary_fh.write(
                diary_index_template.render(
                    title='Data Diary for {0}'.format(data_dirs_str),
                    datetime_generated=datetime_generated,
                    summaries=summaries,
                    experiments_subtitle=experiments_subtitle,
                    experiments=sections
                    )
                )
    print("Finished.")

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
