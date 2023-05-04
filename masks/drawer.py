import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, overload

import cv2
import numpy as np
from cvat_sdk import Client
from cvat_sdk.api_client.exceptions import NotFoundException
from cvat_sdk.api_client.models import IDataMetaRead, ILabeledShape
from cvat_sdk.core.proxies.jobs import Job
from numpy import ndarray

# logger = logging.getLogger(__name__)


class Drawer:
    """Provides session management, implements authentication operations and
    simplifies access to CVAT server APIs and additional methods of working
    with the CVAT API.

    Masks creating example:
    >>> masks = drawer.draw_masks(id=15000,
                                  destination=Path('target_dir'),
                                  is_job=True,
                                  on_photo=True)
    """

    def __init__(self, host: str, username: str, password: str) -> None:
        self._host = host
        self._username = username
        self._password = password

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.logout()
        return

    @property
    def host(self) -> str:
        return self._host

    @property
    def username(self) -> str:
        return self._username

    @property
    def password(self) -> str:
        return self._password

    @property
    def client(self):
        self._client = Client(url=self.host)
        self._client.login((self._username, self._password))

        return self._client

    def _get_jobs(self,
                  id: int,
                  is_job: Optional[bool] = None,
                  is_task: Optional[bool] = None) -> List[Job]:
        """Returns a list of Job objects for task or
        a list with a single Job object for job.

        Args:
            id (int): task or job id.
            is_job (Optional[bool], optional): True if job id.
            Defaults to None.
            is_task (Optional[bool], optional): True if tas id.
            Defaults to None.

        Returns:
            List[Job]: list of Job objects.
        """

        jobs = []

        if is_job:
            job = self.client.jobs.retrieve(obj_id=id)
            jobs.append(job)

        elif is_task:
            task = self.client.tasks.retrieve(obj_id=id)
            jobs = task.get_jobs()

        return jobs

    @staticmethod
    def _make_task_dir(destination: Path, jobs: List[Job]) -> Path:
        """Creates a directory with task id as directory name and returns path
        to created directoty.

        Args:
            destination (Path): destination as Path object.
            jobs (List[Job]): list of Job objects.

        Returns:
            Path: path to created directoty as Path object.
        """
        task_id = str(jobs[0].task_id)
        task_dir = destination / Path(task_id)
        task_dir.resolve()
        task_dir.mkdir(parents=True, exist_ok=True)

        return task_dir

    @staticmethod
    def _get_colors(job: Job, ignore_label: str) -> Dict[int, Tuple[int]]:
        """Returns a dictionary with label ids as keys and RGBA color tuples
        as values.

        Args:
            job (Job): Job object.
            ignore_label (str): label name to ignore (will be set to black).

        Returns:
            Dict[int, Tuple[int]]: dictionary with label ids and colors.
        """
        colors = {}

        for label in job.labels:
            if label.name.lower() == ignore_label:
                colors[label.id] = (0, 0, 0, 0)

            else:
                color = label.color[1:]

                # Convert RGB to BGR for OpenCV.
                color = list(int(color[i:i + 2], 16) for i in (4, 2, 0))
                color.append(255)
                colors[label.id] = tuple(color)

        return colors

    @staticmethod
    def _make_job_dir(destination: Path, job: Job) -> Path:
        """Creates a directory with job id as directory name and returns path
        to created directoty.

        Args:
            destination (Path): destination as Path object.
            jobs (Job): Job object.

        Returns:
            Path: path to created directoty as Path object.
        """
        job_id = str(job.id)
        job_dir = destination / Path(job_id)
        job_dir.resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        return job_dir

    @staticmethod
    def _parse_meta(meta: IDataMetaRead) -> Dict[int, Dict]:
        """Returns job frames metadata as dictionary with frame id as key
        and dictionary wtih name, height, width as value.

        Args:
            meta (IDataMetaRead): job's frames metada.

        Returns:
            Dict[int, Dict]: dictionary with converted metadata.
        """
        frames = {}
        start_frame = meta.start_frame
        for id, frame in enumerate(meta.frames, start=start_frame):
            name: str = frame.name
            height: int = frame.height
            width: int = frame.width
            frames[id] = {'name': name, 'height': height, 'width': width}

        return frames

    def _parse_shapes(self,
                      shapes: List[ILabeledShape]) -> Dict[int, List[Dict]]:
        """Returns a dictionary with job's frames ids as keys and lists of
        shapes sorted by z_order as values.

        Args:
            shapes (List[ILabeledShape]): list with job's shapes sorted by
            z_order.

        Returns:
            Dict[int, List[Dict]]: dictionary with shapes grouped by frame ids.
        """
        labeled_data = {}

        for shape in shapes:
            frame_id = shape.frame
            shape_type = shape.type.value
            shape_points = shape.points
            z_order = shape.z_order
            label_id = shape.label_id

            shape_data = {
                'label_id': label_id,
                'type': shape_type,
                'points': shape_points,
                'z_order': z_order
            }

            if not labeled_data.get(frame_id):
                labeled_data[frame_id] = [shape_data]

            else:
                labeled_data[frame_id].append(shape_data)

        labeled_data = self._sort_by_z_order(labeled_data=labeled_data)

        return labeled_data

    @staticmethod
    def _sort_by_z_order(
            labeled_data: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
        """Returns shapes dictionary with shapes sorted by z_order.

        Args:
            labeled_data (Dict[int, List[Dict]]): dictionary with unsorted
            shapes lists.

        Returns:
            Dict[int, List[Dict]]: dictionary with sorted shapes lists.
        """
        for _, shapes in labeled_data.items():
            shapes.sort(key=lambda x: x.get('z_order'))

        return labeled_data

    @staticmethod
    def _mix(frames: Dict[int, Dict],
             shapes: Dict[int, List[Dict]]) -> Dict[int, Dict]:
        """Returns mix of frames and shapes dictionaries grouped by frames ids.

        Args:
            frames (Dict[int, Dict]): dictionary with frames metadata.
            shapes (Dict[int, List[Dict]]): dictionary with shapes data.

        Returns:
            Dict[int, Dict]: mixed dictionary with frames metadata and
            shapes data.
        """
        for id in frames:
            frames[id]['shapes'] = shapes.get(id, list())

        return frames

    @staticmethod
    def _has_shapes(labeled_frame: Dict[str, Any]) -> bool:
        """Returns True if frame has shapes else False.

        Args:
            labeled_frame (Dict[str, Any]): dictionary with frame data.

        Returns:
            bool: result.
        """
        if labeled_frame.get('shapes'):

            for shape in labeled_frame['shapes']:
                if not shape.get('points'):
                    return False

        else:
            return False

        return True

    def _make_background(self, labeled_frame: Dict[str, Any],
                         photo: ndarray | None) -> ndarray:
        """Returns background as np.ndarray (black or image) in RGBA format.

        Args:
            labeled_frame (Dict[str, Any]): dictionary with labeled frame data.
            photo (ndarray | None): image as np.ndarray or None for black.

        Returns:
            ndarray: background as np.ndarray.
        """
        height = labeled_frame.get('height')
        width = labeled_frame.get('width')

        if not isinstance(photo, ndarray):
            background: ndarray = np.zeros((height, width, 3), np.uint8)
            background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
        else:
            background = cv2.cvtColor(photo, cv2.COLOR_RGB2RGBA)

        return background

    @staticmethod
    def _make_overlay(colors: Dict[int, Tuple[int]],
                      labeled_frame: Dict[str, Any]) -> ndarray:
        """Returns overlay for frame with drawed shapes as np.ndarray.

        Args:
            colors (Dict[int, Tuple[int]]): dictionary with labels colors.
            labeled_frame (Dict[str, Any]): dictionary with labeled frame data.

        Returns:
            ndarray: iverlay image with drawed shapes.
        """
        height = labeled_frame.get('height')
        width = labeled_frame.get('width')
        shapes = labeled_frame.get('shapes')

        overlay: ndarray = np.zeros((height, width, 4), np.uint8)

        # TODO Add other type of shapes
        for shape in shapes:
            points = shape.get('points')
            points = [
                tuple(points[i:i + 2]) for i in range(0,
                                                      len(points) - 1, 2)
            ]
            points = np.array(points).astype(int)

            if shape['type'] == 'polygon':
                overlay = cv2.fillPoly(overlay, [points],
                                       color=(colors[shape['label_id']]))

            elif shape['type'] == 'polyline':
                overlay = cv2.polylines(overlay, [points],
                                        isClosed=False,
                                        color=colors[shape['label_id']],
                                        thickness=3)

            elif shape['type'] == 'points':
                for center in points:
                    overlay = cv2.circle(overlay,
                                         center=center,
                                         radius=0,
                                         color=colors[shape['label_id']],
                                         thickness=-1)

            # TODO Add rotation support
            elif shape['type'] == 'rectangle':
                overlay = cv2.rectangle(overlay,
                                        *points,
                                        color=colors[shape['label_id']],
                                        thickness=1)
        return overlay

    @staticmethod
    def _apply_mask(background: ndarray, overlay: ndarray,
                    transparent: bool) -> ndarray:
        """Returns result image as np.ndarray after adding overlay to
        background with or without overlay transparency.

        Args:
            background (ndarray): background image.
            overlay (ndarray): foreground image.
            transparent (bool): True if needs in transparent overlay.

        Returns:
            ndarray: result image as np.ndarray.
        """
        roi = background

        # Create a mask of overlay and create its inverse mask also
        img_to_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_to_gray, 1, 255, cv2.THRESH_BINARY)
        inverse_mask = cv2.bitwise_not(mask)

        # Take only region of overlay from overlay image.
        fg = cv2.bitwise_and(overlay, overlay, mask=mask)

        # Black-out the area of overlay in ROI
        # Put overlay in ROI and modify the background
        if not transparent:
            bg = cv2.bitwise_and(roi, roi, mask=inverse_mask)
            background = cv2.add(bg, fg)

        else:
            bg = background
            background = cv2.addWeighted(bg, 1, fg, 0.75, 0)

        return background

    def _make_masks(self, job: Job, target_dir: Path, on_photo: bool,
                    transparent: bool, visualize: bool,
                    ignore_label: str) -> None:
        """Makes masks or visualisations as collages CVAT job.

        Args:
            job (Job): Job object.
            target_dir (Path): path to save result images.
            on_photo (bool): True for masks on photos.
            transparent (bool): True for transparent masks on photos.
            visualize (bool): True for collages creating. If True, 'on_photo'
            and 'transparent' will be ignored.
            ignore_label (str): label name to cut.
        """
        colors = self._get_colors(job=job, ignore_label=ignore_label)

        job_dir = self._make_job_dir(destination=target_dir, job=job)

        meta = job.get_meta()
        annotations = job.get_annotations()
        shapes = annotations.shapes
        frames = self._parse_meta(meta=meta)
        shapes = self._parse_shapes(shapes=shapes)

        labeled_frames = self._mix(frames=frames, shapes=shapes)

        if not visualize:
            for id, data in labeled_frames.items():
                if self._has_shapes(data):
                    photo = None
                    if on_photo:
                        photo = np.frombuffer(job.get_frame(
                            id, quality='original').read(),
                                              dtype=np.uint8)
                        photo = cv2.imdecode(photo, 1)

                    background = self._make_background(labeled_frame=data,
                                                       photo=photo)
                    overlay = self._make_overlay(colors=colors,
                                                 labeled_frame=data)
                    mask = self._apply_mask(background=background,
                                            overlay=overlay,
                                            transparent=transparent)

                    name = Path(data.get('name')).name
                    mask_path = job_dir / Path(name).with_suffix('.png')

                    cv2.imwrite(str(mask_path), mask)

        else:
            for id, data in labeled_frames.items():
                if self._has_shapes(data):
                    photo = np.frombuffer(job.get_frame(
                        id, quality='original').read(),
                                          dtype=np.uint8)
                    photo = cv2.imdecode(photo, 1)
                    photo = cv2.cvtColor(photo, cv2.COLOR_RGB2RGBA)

                    black_background = self._make_background(
                        labeled_frame=data, photo=None)
                    background = self._make_background(labeled_frame=data,
                                                       photo=photo)
                    overlay = self._make_overlay(colors=colors,
                                                 labeled_frame=data)
                    transparent_mask = self._apply_mask(background=background,
                                                        overlay=overlay,
                                                        transparent=True)
                    mask = self._apply_mask(background=black_background,
                                            overlay=overlay,
                                            transparent=False)

                    name = Path(data.get('name')).name
                    collage_path = job_dir / Path(name).with_suffix('.jpg')

                    collage = np.concatenate((photo, transparent_mask, mask),
                                             axis=1)

                    cv2.putText(collage, f'frame_id: {str(id)}',
                                (10, data.get('height') - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0, 255),
                                2)
                    cv2.imwrite(str(collage_path), collage)

    @overload
    def draw_masks(self,
                   id: int,
                   destination: Path,
                   is_job: bool = None,
                   on_photo: bool = False,
                   transparent: bool = False,
                   visualize: bool = False,
                   ignore_label: str = 'ignore') -> Path:
        pass

    @overload
    def draw_masks(self,
                   id: int,
                   destination: Path,
                   is_task: bool = None,
                   on_photo: bool = False,
                   transparent: bool = False,
                   visualize: bool = False,
                   ignore_label: str = 'ignore') -> Path:
        pass

    def draw_masks(self,
                   id: int,
                   destination: Path,
                   is_job: Optional[bool] = None,
                   is_task: Optional[bool] = None,
                   on_photo: bool = False,
                   transparent: bool = False,
                   visualize: bool = False,
                   ignore_label: str = 'ignore') -> Path:
        """Draws masks (on black background or on photo) or visualizations
        (collages from original photo, transparent mask on photo and mask
        on black background), packes into zip-archive and returns path to it.
        Draws masks on black background by default.

        Args:
            id (int): CVAT task or job id.

            destination (Path): path to save result archive.

            is_job (Optional[bool], optional): True if job id was passed.
            Defaults to None.

            is_task (Optional[bool], optional): True if task id was passed.
            Defaults to None.

            on_photo (bool, optional): True if masks on photo are needed.
            Defaults to False.

            transparent (bool, optional): True if transparent masks on photo
            are needed. Defaults to False.

            visualize (bool, optional): True if visualizations are needed.
            Defaults to False.

            ignore_label (str, optional): name of label to ignore (to cut).
            Defaults to 'ignore'.

        Raises:
            ValueError: raises if both of 'is_job' and 'is_task' are set to
            True or False or if 'transparent' is set to True, but 'on_photo'
            isn't set to True.

        Returns:
            Path: path to result archive.
        """

        args = [arg for arg in (is_job, is_task) if arg]
        if len(args) > 1:
            raise ValueError(
                'Only one of the optional arguments can be specified.')
        elif len(args) < 1:
            raise ValueError(
                'At least one of the optional arguments must be specified.')

        if (not on_photo) and transparent:
            raise ValueError(
                "Transparent can be set to True only with 'on_photo'=True")

        root: Path
        if visualize:
            root = Path('collages')
        elif on_photo:
            root = Path('masks_on_photo')
        else:
            root = Path('')

        destination = destination / root
        destination.resolve()

        jobs = self._get_jobs(id, is_job=is_job, is_task=is_task)

        task_dir = self._make_task_dir(destination, jobs)

        for job in jobs:
            self._make_masks(job=job,
                             target_dir=task_dir,
                             on_photo=on_photo,
                             transparent=transparent,
                             visualize=visualize,
                             ignore_label=ignore_label)

        archive_name = shutil.make_archive(task_dir.parent, 'zip',
                                           task_dir.parent, task_dir.name)
        archive_path = destination.parent / Path(archive_name)

        shutil.rmtree(task_dir.parent)

        return archive_path

    def exists(self, id: int, type: str) -> bool:
        """Returns True if task or job with passed id exists or Flase if
        doesn't exist.

        Args:
            id (int): CVAT task or job id.
            type (str): type ('task' or 'job)

        Returns:
            bool: True if exists.
        """
        try:
            if type == 'task':
                self.client.tasks.retrieve(id)
            elif type == 'job':
                self.client.jobs.retrieve(id)
        except NotFoundException as e:
            # logger.exception(e)
            return False
        else:
            return True
