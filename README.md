# [Создание контекстных изображений для CVAT](context_images/make_context_images.py)

## Суть задачи

В CVAT для удобства разметки данных есть функция отображения контекстных изображений, которые по сути являются подсказкой-указателем для исполнителя.

[Context images for 2d task](https://opencv.github.io/cvat/docs/manual/advanced/context-images/)

Для их создания необходимо сформировать изображения с необходимой информацией и создать определенную файловую структуру, как описано в документации выше.

## Инструменты

Как и в случае с созданием масок можно использовать библиотеки

[Pillow](https://pypi.org/project/Pillow/)

[OpenCV: OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

[opencv-python](https://pypi.org/project/opencv-python/)