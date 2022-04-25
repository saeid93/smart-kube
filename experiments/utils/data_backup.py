"""
to bakcup data folder when necessary
source: https://www.geeksforgeeks.org/working-zip-files-python/
"""
from zipfile import ZipFile
import os
import sys
import argparse

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    CLUSTERS_PATH,
    TRAIN_RESULTS_PATH,
    DATA_PATH,
    BACKUP_PATH
)

parser = argparse.ArgumentParser()
parser.add_argument('--backup', type=str, default='cluster')
args = parser.parse_args()


def get_all_file_paths(directory):

    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # returning all file paths
    return file_paths


def main():

    # path to folder which needs to be zipped
    if args.backup == 'cluster':
        directory = CLUSTERS_PATH
        saved_file_path = os.path.join(BACKUP_PATH, 'cluster.zip')
    elif args.backup == 'results':
        directory = TRAIN_RESULTS_PATH
        saved_file_path = os.path.join(BACKUP_PATH, 'results.zip')
    elif args.backup == 'data':
        directory = DATA_PATH
        saved_file_path = os.path.join(BACKUP_PATH, 'data.zip')
    else:
        raise ValueError(f"invalid input for backup <{args.backup}>")
    # calling function to get all file paths in the directory
    file_paths = get_all_file_paths(directory)

    # printing the list of all files to be zipped
    print('Following files will be zipped:')
    for file_name in file_paths:
        print(file_name)

    # writing files to a zipfile
    with ZipFile(saved_file_path, 'w') as zip:
        # writing each file one by one
        for file in file_paths:
            zip.write(file)

    print('All files zipped successfully!')


if __name__ == "__main__":
    main()
