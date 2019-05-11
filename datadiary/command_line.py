#!/usr/bin/env python3


import argparse
import sys

import datadiary


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
            description="Create html tree of describing data of many jobs.")

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


def main():
    try:
        args = process_command_line(sys.argv)
        status = datadiary.generate_diary(args.diary, args.datadir)
    except KeyboardInterrupt:
        # Make a very clean exit (no debug info) if user breaks with Ctrl-C
        print("Stopped by Keyboard Interrupt", file=sys.stderr)
        # exit error code for Ctrl-C
        status = 130

    sys.exit(status)
