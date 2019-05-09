#!/usr/bin/env python3
#
# Post-training view of loss, accuracy metrics

# Jinja2 tutorial:
# https://dev.to/goyder/automatic-reporting-in-python---part-1-from-planning-to-hello-world-32n1
# https://dev.to/goyder/automatic-reporting-in-python---part-2-from-hello-world-to-real-insights-8p3
# https://dev.to/goyder/automatic-reporting-in-python---part-3-packaging-it-up-1185


import argparse
import datetime
import hashlib
import json
import os
import os.path
import pathlib
#import pprint # debug
import sys
from contextlib import redirect_stderr

import imagesize
import jinja2
import tqdm
# Mute Tensorflow chatter messages ('1' means filter out INFO messages.)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Mute Keras chatter messages
with redirect_stderr(open(os.devnull, "w")):
    import keras
import numpy as np

# custom version of keras.utils.vis_utils to get dpi argument
import datadiary.keras_vis_utils
import datadiary.plotting as plotting

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


def hash_string(in_str, hash_len=6):
    """Create a hash string based on input string

    Arguments:
        in_str (str): input string
        hash_len (int): length of returned hex hash string

    Returns:
        (str): hex ([0-9a-f]+) string of length hash_len, corresponding to hash
            of in_str
    """
    model_hash = hashlib.md5(in_str.encode('utf8'))
    return model_hash.hexdigest()[:hash_len]


def plot_model_get_info(data_subdir, diary_subdir):
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

    # get model info
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
    datadiary.keras_vis_utils.plot_model(
            my_model,
            to_file=str(diary_subdir / 'model.png'),
            show_shapes=True,
            dpi=192,
            transparent_bg=True
            )

    del my_model
    # NEED TO DO THIS or else memory leak and slowdown happen
    #   (keras 2.2.4, tensorflow 1.13.1)
    keras.backend.clear_session()

    return (model_opt_name, model_opt_config_fmt, model_opt_str, model_loss_type)


def render_experiment_html(diary_dir, experiment, global_data):
    data_subdir = experiment['info']['datadir']
    train_data = experiment['train']

    # get hash of datadir
    dirhash = hash_string(str(experiment['info']['datadir']), hash_len=16)

    diary_subdir = diary_dir / (experiment['info']['model_name'] + "_" + dirhash)
    diary_subdir.mkdir(parents=True, exist_ok=True)

    # make plot png
    plotting.gen_data_plots(diary_subdir, train_data, global_data)

    # model diagram and get model info
    (model_opt_name, model_opt_config_fmt, model_opt_str, model_loss_type) = plot_model_get_info(
            data_subdir, diary_subdir
            )

    test_acc_perc = experiment.get('test',{}).get('test_acc_perc', None)

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
    job['datadir'] = experiment['info']['datadir']
    job['model_diagram_img'] = 'model.png'
    job['model_diagram_img_style'] = model_img_size_str
    job['model_metrics_img'] = 'training_metrics.png'
    job['model_metrics_img_style'] = plot_img_size_str
    if test_acc_perc is not None:
        job['test_acc_perc'] = '{0:.1f}'.format(test_acc_perc)
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


def get_training_data(train_data_path):
    if not train_data_path.is_file():
        return {}
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
    return train_data


def get_test_data(test_data_path):
    try:
        with test_data_path.open("r") as test_data_fh:
            test_data = json.load(test_data_fh)
    except IOError:
        test_data = {}
    return test_data


def get_info_data(info_data_path):
    model_dir = info_data_path.parent
    try:
        with info_data_path.open("r") as info_data_fh:
            info_data = json.load(info_data_fh)
    except IOError:
        info_data = {}
    # datetime finished
    if 'datetime_utc' in info_data:
        datetime_finished_utc = datetime.datetime.strptime(
                info_data['datetime_utc'],
                "%Y-%m-%d %H:%M:%S"
                )
        datetime_finished_utc = datetime_finished_utc.replace(tzinfo=datetime.timezone.utc)
        info_data['datetime_finished'] = datetime_finished_utc.astimezone()
        info_data['datetime_formatted'] = info_data['datetime_finished'].strftime("%Y-%m-%d %I:%M%p")
        info_data['datetime_sortable'] = info_data['datetime_finished'].strftime("%Y%m%d%H%M")
    else:
        info_data['datetime_finished'] = None
    # model_name
    if 'model_name' not in info_data:
        info_data['model_name'] = model_dir.name.lstrip("data_")
    # extract job name
    if model_dir.parent.name.startswith("j"):
        info_data['job_id'] = model_dir.parent.name
    else:
        info_data['job_id'] = "Local Job"
    # record directory of data
    info_data['datadir'] = model_dir
    return info_data


def catalog_dir(model_dir):
    experiment = {}

    # Training data
    experiment['train'] = get_training_data(model_dir / 'train_history.json')
    # Test data
    experiment['test'] = get_test_data(model_dir / 'test.json')
    # Info data
    experiment['info'] = get_info_data(model_dir / 'info.json')

    # DEBUG only
    #pprint.pprint(experiment)

    return experiment


def catalog_all_dirs(model_dirs):
    # Find max loss over all datasets, so we can adjust all
    #   loss plots from 0 to max loss (consistent ylim for all plots)
    global_data = {}
    experiments = []
    for model_dir in model_dirs:
        model_dir_data = catalog_dir(model_dir)
        if model_dir_data is None:
            continue
        experiments.append(model_dir_data)
        global_data['max_loss'] = max(
                global_data.get('max_loss', 0),
                model_dir_data['train']['max_loss']
                )
        global_data['max_acc_perc'] = max(
                global_data.get('max_acc_perc', 0),
                model_dir_data['train']['max_acc_perc']
                )
        global_data['min_best_epoch'] = min(
                global_data.get('min_best_epoch', 1e6),
                model_dir_data['train']['best_epoch']
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


def get_model_dirs(data_topdirs):
    """
    Args:
        datadirs (list): top-level list of directories to search for model
            directories

    Returns (list): list of directories containing saved_models dir,
        file train_history.json, and possibly file test.json
    """
    all_subdirs = []
    for data_topdir in data_topdirs:
        all_subdirs.extend(data_topdir.glob('**'))

    all_subdirs = set(all_subdirs)

    model_dirs = []
    for subdir in all_subdirs:
        if (subdir / "train_history.json").is_file() and (subdir / "saved_models").is_dir():
            model_dirs.append(subdir)
    return model_dirs


def render_diary(diary_dir, experiments, global_data, data_topdirs):
    # sort by validation accuracy
    experiments.sort(key=lambda x: x.get('test', {}).get('test_acc_perc', 0), reverse=True)
    experiments_subtitle = '(Sorted by Test Accuracy)'

    print("Diary output to: {0}".format(diary_dir))
    print("Rendering HTML summaries of all jobs...")
    experiment_summaries = []
    for experiment in tqdm.tqdm(experiments, leave=False, unit='job'):
        experiment_summaries.append(
                render_experiment_html(diary_dir, experiment, global_data)
                )

    # create rankings
    rankings = []

    # tolerate empty test dict in data dirs
    rankings.append(
            create_ranking(
                [exp for exp in experiments if 'test_acc_perc' in exp['test']],
                title="Best Test Accuracy",
                sort_key=lambda x: x['test']['test_acc_perc'],
                reverse=True,
                info_dict_create=lambda x: {
                    'name':x['info']['model_name'],
                    'datadir':x['info']['datadir'],
                    'criteria_value':"{0:.1f}%".format(x['test']['test_acc_perc'])
                    },
                )
            )
    rankings.append(
            create_ranking(
                experiments,
                title="Best Validation Accuracy",
                sort_key=lambda x: x['train']['best_val_acc_perc'],
                reverse=True,
                info_dict_create=lambda x: {
                    'name':x['info']['model_name'],
                    'datadir':x['info']['datadir'],
                    'criteria_value':"{0:.1f}%".format(x['train']['best_val_acc_perc'])
                    },
                )
            )
    rankings.append(
            create_ranking(
                experiments,
                title="Quickest Training (Minimum Best Epoch)",
                sort_key=lambda x: x['train']['best_epoch'],
                reverse=False,
                info_dict_create=lambda x: {
                    'name':x['info']['model_name'],
                    'datadir':x['info']['datadir'],
                    'criteria_value':"Epoch {0}".format(x['train']['best_epoch'])
                    },
                )
            )

    # copy sortable.js into diary dir
    with (diary_dir / 'sorttable.js').open("w") as sorttable_js_fh:
        sorttable_js_fh.write(
                JINJA_ENV.get_template("sorttable.js").render()
                )

    # create index/summary html report
    master_diary = diary_dir / 'index.html'
    diary_index_template = JINJA_ENV.get_template("diary.html")
    datetime_generated = datetime.datetime.now().strftime("%Y-%m-%d %a %I:%M%p")
    local_timezone=datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Z")

    data_dirs_parent = [d.parent.resolve() for d in data_topdirs]
    data_dirs_commonpath = os.path.commonpath([str(d) for d in data_dirs_parent])
    if len(data_topdirs) > 1:
        data_subdirs_str = ", ".join(
                [str(d.relative_to(data_dirs_commonpath)) for d in data_dirs_parent]
                )
        data_subdirs_str = "/{" + data_subdirs_str + "}"
    else:
        data_subdirs_str = ""

    # TODO 2019-05-08: it's nice to put timezone in header of column instead of
    #   each row, but this is strictly not right with daylight savings, as
    #   old dates may be different daylight savings than now.
    with master_diary.open("w") as master_diary_fh:
        master_diary_fh.write(
                diary_index_template.render(
                    title='{0}{1}'.format(
                        data_dirs_commonpath, data_subdirs_str
                        ),
                    local_timezone=local_timezone,
                    datetime_generated=datetime_generated,
                    rankings=rankings,
                    experiments_subtitle=experiments_subtitle,
                    experiment_summaries=experiment_summaries,
                    experiments=experiments
                    )
                )


def main(argv=None):
    args = process_command_line(argv)
    diary_dir = pathlib.Path(args.diary)

    data_topdirs = [pathlib.Path(dir) for dir in args.datadir]
    model_dirs = get_model_dirs(data_topdirs)

    (experiments, global_data) = catalog_all_dirs(model_dirs)

    render_diary(diary_dir, experiments, global_data, data_topdirs)
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
