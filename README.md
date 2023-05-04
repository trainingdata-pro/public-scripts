## [Создание контекстных изображений для CVAT](context_images/make_context_images.py)

### Суть задачи

В CVAT для удобства разметки данных есть функция отображения контекстных изображений, которые по сути являются подсказкой-указателем для исполнителя.

[Context images for 2d task](https://opencv.github.io/cvat/docs/manual/advanced/contextual-images/)

Для их создания необходимо сформировать изображения с необходимой информацией и создать определенную файловую структуру, как описано в документации выше.

### Инструменты

Как и в случае с созданием масок можно использовать библиотеки

[Pillow](https://pypi.org/project/Pillow/)  
[OpenCV: OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)  
[opencv-python](https://pypi.org/project/opencv-python/)  

------------------

## [Случайная выборка файлов из датасета](files/select_files.py)

### Суть задачи

Выбрать случайным образом n файлов по m из кадой дочерней директории

### Инструменты

Инструменты для работы с файлами могут быть самыми разными в зависимости от задачи, для манипуляций с файлами датасета, как правило, используются библиотеки из Python Core:

[os - Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html)  
[shutil - High-level file operations](https://docs.python.org/3/library/shutil.html)  
[pathlib - Object-oriented filesystem paths](https://docs.python.org/3/library/pathlib.html)  

------------------

## [Очистка метаданных у изображений](images/clear_meta_and_rotate_image.py)

### Суть задачи

В данной задаче необходимо почистить метаданные у заданных изображений. После очистки мета данных может случиться так, что изображение “ляжет на бок”. В этом случае необходимо восстановить правильную ориентацию изображения.

### Инструменты

Для этого можно использовать python-библиотеку Pillow.

[Pillow](https://pillow.readthedocs.io/en/stable/)  

------------------

## [Создание масок и визуализаций](masks/drawer.py)

### Суть задачи

В этой задаче необходимо нарисовать маски разных типов по заданной разметке.

### Инструменты

Для парсинга файлов с разметкой можно использовать Python-библиотеки

[Beautiful Soup Documentation - Beautiful Soup 4.9.0 documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)  
[xml.etree.ElementTree - The ElementTree XML API](https://docs.python.org/3/library/xml.etree.elementtree.html)  

Для работы с самими изображениями

[Pillow](https://pypi.org/project/Pillow/)  
[OpenCV: OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)  
[opencv-python](https://pypi.org/project/opencv-python/)  

------------------

## [Кадрирование видео для создания датасета](video/get_frames_from_video_ffmpeg.py)

### Суть задачи

Создание набора изображений из кадров видео, взятых с определенной частотой.

### Инструменты

[OpenCV: OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)  
[opencv-python](https://pypi.org/project/opencv-python/)  
[moviepy](https://pypi.org/project/moviepy/)  
