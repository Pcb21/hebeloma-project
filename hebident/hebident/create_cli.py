import argparse
import sys
import os

import pandas

from .util import PrintToScreenAndFile
from .creator import create_classifiers


def main():
    parser = argparse.ArgumentParser(description='Hebeloma species identifier creator tool.')
    parser.add_argument('file', type=str, nargs='?', help='Path to a .csv file containing Hebeloma collection data')
    parser.add_argument('-o', '--output', type=str, help='The name of a directory to output the identifiers to')
    args = parser.parse_args()
    collections_df = pandas.read_csv(args.file)

    output_directory = args.output
    os.makedirs(output_directory, exist_ok=True)
    output_log_file = os.path.join(output_directory, "identifier_creator.log")
    logger = PrintToScreenAndFile(output_log_file)

    create_classifiers(collections_df=collections_df, output_directory=output_directory, logger=logger)


if __name__ == '__main__':
    sys.exit(main())
