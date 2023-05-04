#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import ffmpeg
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    """Return a Namespace containing the arguments.

    Returns:
        argparse.Namespace: Namespace with arguments.
    """
    parser = argparse.ArgumentParser(
        description='''This script gets frames from video''')

    parser.add_argument('-fps2save',
                        '--frames_per_second_to_save',
                        type=float,
                        help='Frames per second to save',
                        required=True)

    parser.add_argument('-vid_dir',
                        '--videos_directory',
                        type=Path,
                        help="Path to directory with input video files",
                        required=True)

    parser.add_argument('-img_dir',
                        '--images_directory',
                        type=Path,
                        help="Path to directory with output images",
                        required=True)

    parser.add_argument('-f',
                        '--formats',
                        type=list,
                        help="List of videoformats to convert",
                        default=['.avi', '.mp4', '.mov'],
                        required=False)

    args = parser.parse_args()
    return args


def extract_frames(video_file: Path, fps: float, output: Path):
    output = output / Path(f"{video_file.stem}_frames")
    if not output.exists():
        output.mkdir(exist_ok=True, parents=True)

    image_file = f'{str(output / video_file.stem)}_%04d.jpg'
    (ffmpeg.input(str(video_file)).filter('fps', fps=fps).output(
        image_file, start_number=0).overwrite_output().run(quiet=True))


if __name__ == "__main__":
    args = get_args()
    fps_to_save: int = args.frames_per_second_to_save
    vid_dir: Path = args.videos_directory
    img_dir: Path = args.images_directory
    formats: List[str] = args.formats
    for file in tqdm(list(vid_dir.iterdir())):
        if file.suffix in formats:
            extract_frames(file, fps_to_save, img_dir)
