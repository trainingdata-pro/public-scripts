import os
from PIL import Image, ExifTags
from tqdm import tqdm

for root, dirs, files in os.walk('original'):
    for im in tqdm(files):
        original_img = Image.open(os.path.join(root, im))
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = original_img._getexif()
        try:
            if exif[orientation] == 3:
                original_img = original_img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                original_img = original_img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                original_img = original_img.rotate(90, expand=True)
        except (TypeError, KeyError):
            print(f'error: {im}')
        finally:
            full_path = os.path.join('result', root)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            original_img.save(os.path.join(full_path, im))

