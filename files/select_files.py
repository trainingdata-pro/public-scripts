#!/usr/bin/env python3

import argparse
import os
import random
import shutil
from pathlib import Path


def get_args() -> argparse.Namespace:
    """Return a Namespace containing the arguments.

    Returns:
        argparse.Namespace: Namespace with arguments.
    """
    parser = argparse.ArgumentParser(
        description='''This script selects n files, m from each subdirectory
                       in source directory, and put in target directory''')

    parser.add_argument('srcdir', type=str, help='Source dir with files')
    parser.add_argument('outdir', type=str, help='Output dir for files')
    parser.add_argument('-n', type=int, default=10000, help='Number of files')
    parser.add_argument('-m',
                        type=int,
                        default=1,
                        help='Number if files from subdirectory')

    args = parser.parse_args()
    return args


def make_dirs(source_dir: str, output_dir: str) -> tuple[Path, Path]:
    """Return Path objects with paths to source and output directories.

    Args:
        source_dir (str): source directory name.
        output_dir (str): output directory name.

    Returns:
        tuple[Path, Path]: Path objects with paths.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    Path.mkdir(output_path, exist_ok=True)
    return source_path, output_path


if __name__ == '__main__':
    args = get_args()

    source_dir = args.srcdir
    output_dir = args.outdir
    max_num = args.n
    subdir_number = args.m

    source_path, output_path = make_dirs(source_dir, output_dir)

    common_counter = 0
    cur_path = Path.cwd()

    try:
        tree = list(os.walk(source_path))
        copied_files = set()

        while common_counter < max_num:
            for path, _, files in tree[1:]:
                if files:
                    file = random.choice(files)
                    if file in copied_files:
                        files.remove(file)
                        continue
                    file_path = Path(path, file)
                    files.remove(file)
                    copied_files.add(file)
                    shutil.copy2(file_path, output_path)
                    common_counter += 1
                    if common_counter == max_num:
                        break

    except Exception as err:
        print(err)
    else:
        shutil.make_archive(source_dir, 'zip', output_path)
    finally:
        print('Script is stopped!')
