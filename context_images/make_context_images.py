#!/usr/bin/env python
import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    """Return a Namespace containing the arguments.

    Returns:
        argparse.Namespace: Namespace with arguments.
    """
    parser = argparse.ArgumentParser(
        description='''This script make context images.''')

    parser.add_argument('-dirs',
                        '--directories',
                        type=str,
                        nargs='*',
                        help='Source directories',
                        required=True)

    parser.add_argument('-s',
                        '--suffix',
                        type=str,
                        default='.jpg',
                        help="Target image's suffix")
    parser.add_argument('-con',
                        '--context',
                        type=str,
                        default='plate.txt',
                        help="Context txt-file's name")

    args = parser.parse_args()
    return args


def make_dir(path: Path, name: str) -> Path:
    path = path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def contains_image(path: Path, suffix: str) -> Path | None:
    image = list(path.glob(f'*{suffix}'))
    if image:
        return image[0]


def make_context_image(text: str, height: int, width: int, x0: int, y0: int):
    background = np.zeros((height, width, 3), np.uint8)

    # fontpath = f'/Ubuntu_B.ttf'
    font = ImageFont.truetype('Ubuntu-B.ttf', 22)
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    draw.text((x0, y0), text, font=font, fill=(255, 255, 255))
    image = np.array(image)
    return image


if __name__ == '__main__':
    args = get_args()
    dirs: list[str] = args.directories
    image_suffix: str = args.suffix
    context: str = args.context

    current_dir = Path.cwd()
    src_dirs: list[Path] = [Path(dir) for dir in dirs]
    result_dir = make_dir(current_dir, 'result')
    target_dirs = [make_dir(result_dir, dir) for dir in dirs]

    for src_dir, target_dir in tqdm(zip(src_dirs, target_dirs)):
        src_tree = os.walk(src_dir)

        images_counter = 0
        target_subdir_id = 0

        target_txt_dir_name = context[:-4] + 's'
        target_txt_dir = make_dir(target_dir, target_txt_dir_name)

        for src_subdir, _, src_files in tqdm(src_tree, leave=False):
            src_subdir = Path(src_subdir)
            src_image = contains_image(src_subdir, image_suffix)
            if src_image:
                if images_counter % 30000 == 0:
                    target_subdir_id += 1
                    target_subdir_name = str(target_subdir_id)
                    target_subdir = make_dir(target_dir, target_subdir_name)

                    related_images_name = 'related_images'
                    related_images = make_dir(target_subdir,
                                              related_images_name)

                images_counter += 1

                shutil.copy(src_image, target_subdir)
                context_dir_name = f'{src_image.stem}_{src_image.suffix[1:]}'
                context_dir = make_dir(related_images, context_dir_name)

                src_txt = src_subdir / context
                target_txt_name = src_image.with_suffix('.txt').name
                target_txt = shutil.copyfile(src_txt,
                                             target_txt_dir / target_txt_name)

                text = target_txt.read_text()
                context_image = make_context_image(text, 100, 220, 10, 25)
                
                context_image_name = \
                    f'context_image_{src_image.with_suffix(".jpg").name}'
                save_image_as = str(context_dir / context_image_name)
                cv2.imwrite(save_image_as, context_image)
