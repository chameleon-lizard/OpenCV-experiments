# Склеивалка картинок с денойзером

Хелловорлд для OpenCV. Умеет сшивать горизонтальные фото и видео, денойзить эти фото и видео и искать оптический поток. В папке с картинками находится две папки - исходные картинки и картинки с шумом. Шум добавлялся тупо в гимпе. В папке с видео находятся три видео - исходное видео и две её части, полученные тупо разрезанием исходного видео.

## Зависимости

```
opencv-python
argparse
imutils
numpy
```

## Использование
У `image_stitching.py` три параметра:
- `-m --method` - метод денойзинга. Может быть Gaussian, Bilateral, FastNl, none.
- `-i --images` - путь к директории с картинками, которые будем сшивать.
- `-o --output` - путь к выходной картинке.

В моём примере чтобы увидеть всё, что есть, надо сделать так:
```
$ python image_stitching.py -i .\images\room_noisy -o output-denoised-FastNl.jpg -m FastNl
$ python image_stitching.py -i .\images\room_noisy -o output-denoised-Gaussian.jpg -m Gaussian
$ python image_stitching.py -i .\images\room_noisy -o output-denoised-Bilateral.jpg -m Bilateral
$ python image_stitching.py -i .\images\room_noisy -o output-noisy.jpg -m none
$ python image_stitching.py -i .\images\room_default -o output-default.jpg -m none
```

У `video_stitching.py` три параметра:
- `-m --method` - метод денойзинга. Может быть Gaussian, Bilateral, FastNl, none. На моём железе none выдаёт ~2 фпс, с FastNl на производительность смотреть грустно. Рекомендую запускать с none или с фильтрами. В OpenCV есть метод, который осуществляет Fast Non-Local Mean Denoising по нескольким кадрам, но я боюсь за свой компьютер, учитывая, что он будет ещё медленнее однокадрового, а однокадровый тратит на рендер одного кадра по 5-6 секунд.
- `-l --left` - путь к левому видео.
- `-r --right` - путь к правому видео.

```
$ python video_stitching.py -l ./videos/not_noisy/left.mp4 -r ./videos/not_noisy/right.mp4 -m none
$ python video_stitching.py -l ./videos/noisy/left.mp4 -r ./videos/noisy/right.mp4 -m Gaussian
$ python video_stitching.py -l ./videos/noisy/left.mp4 -r ./videos/noisy/right.mp4 -m Bilateral
$ python video_stitching.py -l ./videos/noisy/left.mp4 -r ./videos/noisy/right.mp4 -m FastNl
```

У `optical_flow.py` три параметра:
- `-m --method` - метод денойзинга. Может быть Gaussian, Bilateral, FastNl, none. В принципе, всё, что написано в предыдущем пункте справедливо, но без денойзинга поиску потока будет *очень* грустно, так что рекомендую запускать хотя бы с билатеральным размытием. Там всё вроде оптимально.
- `-l --left` - путь к левому видео.
- `-r --right` - путь к правому видео.

```
$ python video_stitching.py -l ./videos/not_noisy/left.mp4 -r ./videos/not_noisy/right.mp4 -m none
$ python video_stitching.py -l ./videos/noisy/left.mp4 -r ./videos/noisy/right.mp4 -m Gaussian
$ python video_stitching.py -l ./videos/noisy/left.mp4 -r ./videos/noisy/right.mp4 -m Bilateral
$ python video_stitching.py -l ./videos/noisy/left.mp4 -r ./videos/noisy/right.mp4 -m FastNl
```

## Как это работает?

Сначала мы получаем фотографии, потом создаём объект `cv2.stitcher`. Он принимает в себя фотографии, возвращает одну фотографию и результат (либо 0, то есть `OK`, либо сообщение об ошибке).

### Сшивка изображений

![output-default](https://github.com/chameleon-lizard/OpenCV-stitcher/raw/master/output-default.jpg "Результат склейки")


1) Сначала ищутся **ключевые точки**. Это делается по методу **SIFT** (*scale-invariant feature transform*). Происходит свёртка изображений **фильтрами Гаусса** (домножение каждого пикселя на матрицу), потом вычисляется разность размытых изображений. Ключевые точки - это максимумы/минимумы **разности гауссианов** (*DoG*). Это делается путём сравнения каждого пикселя по разности гауссианов изображений для её восьми соседей в том же масштабе и девяти соответствующих соседних пикселей в каждом из соседних масштабов. Если значение пикселя является максимумом или минимумом среди всех сравниваемых точек, оно выбирается как кандидат ключевой точки.
2) Происходит **локализация ключевых точек**. Соседние интерполируются, чтобы выкинуть лишние, выкидываются точки с низким контрастом (они неустойчивы к шумам), исключаются точки, не имеющие хорошо определённого местоположения, но имеющие большой вклад от рёбер потому что такие точки будут, опять же, неустойчивы к шумам.
3) **Назначается ориентация**. Смотрим на направление градиентов в локальном изображении, назначаем ориентацию ключевым точкам, добиваясь инвариантности по вращению.
4) Создаём **Дескрипторы ключевых точек**. Это нужно, чтобы наши ключевые точки были инвариантны не только относительно вращения, сдвига и изменения масштаба, а ещё и шума, освещения, точки обзора и так далее. В первую очередь создаётся набор гистограмм направлений на 4×4 соседних пикселях с 8 областями в каждой. Эти гистограммы вычисляются из значений величины и ориентации элементов в области 16×16 вокруг ключевой точки, так что каждая гистограмма содержит элементы из 4×4 подобласти исходной области соседства. Величины далее взвешиваются функцией Гаусса с сигмой равной половине ширины окна дескриптора. Дескриптор затем становится вектором всех значений этих гистограмм. Поскольку имеется 4 × 4=16 гистограмм с 8 областями в каждой, вектор имеет 128 элементов. Этот вектор нормализуется до единичной длины, чтобы обеспечить инвариантность аффинным изменениям в освещении. Чтобы сократить эффект нелинейного освещения, применяется порог величиной 0,2 и вектор снова нормализуется. 
5) Находятся **одинаковые дескрипторы** у двух изображений, эти изображения сопоставляются.
6) С помощью **RANSAC** (*RANdom SAmple Consensus*), **k-NN** (*метод k-ближайших соседей*) находится **матрица гомографии**.
7) К изображениям применяется матрица гомографии, чтобы исказить их и изображения сшиваются.

*P.S. Стоит отметить, что возможно имело бы смысл обрезать лишние части картинки, например, с помощью вычисления наибольшего прямоугольника, включающегося в получившуюся картинку после искажения и обрезания по нему. Но поскольку получившаяся картинка будет терять данные (в том числе, может терять много данных), я решил пока что на это забить. Мало ли, вдруг что то более умное в голову придёт.*

### Денойзинг
После сшивки изображений, я передаю результат одному из методов, который производит денойзинг изображения. Реализовано три метода: размытие **Гауссом**, **Билатеральное** размытие и денойзинг методом **Fast Non-Local Means Denoising**. Рассмотрим, как работают каждый вариант денойзинга и его результаты.
1) Размытие **Гауссом**.
![output-denoised-gaussian](https://github.com/chameleon-lizard/OpenCV-stitcher/raw/master/output-denoised-gaussian.jpg "Результат денойзинга гауссом")
Мы берём функцию Гаусса, выражающую нормальное распределение для двухмерного случая:
![equation](https://latex.codecogs.com/gif.latex?G(x,&space;y)&space;=&space;\frac{1}{2\pi\sigma^2}e^{-\frac{x^2&space;&plus;&space;y^2}{2\sigma^2}}, "G(x, y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2 + y^2}{2\sigma^2}},")
где *x* - это расстояние от точки применения по оси абсцисс, *y* - расстояние от точки применения на оси ординат, а ![equation](https://latex.codecogs.com/gif.latex?\sigma "\sigma") - среднеквадратичное отклонение распределения Гаусса. С помощью этой функции мы получаем матрицу свёртки, которую мы применяем к каждому пикселю картинки, чтобы размыть её. Алгоритм эффективно удаляет шум, но в моём случае разрешение итоговой картинки оказалось слишком мало и в итоге после применения гаусса с ![equation](https://latex.codecogs.com/gif.latex?\sigma&space;=&space;3 "\sigma = 3") картинка получилась слишком мутной.

2) **Билатеральное** размытие.
![output-denoised-bilateral](https://github.com/chameleon-lizard/OpenCV-stitcher/raw/master/output-denoised-bilateral.jpg "Результат денойзинга билатеральным размытием")
Этот алгоритм отличается от размытия **Гауссом** тем, что он не размывает границы. Идея состоит в том, что два пикселя могут быть близкими *по расстоянию*, а могут быть близкими *по значению*, то есть похожими. Использование двух фильтров, фильтра по расстоянию и по значению, одновременно, даёт возможность заменить пиксель со значением *x* пикселем, со значением, равным усреднённому значению близких и похожих пикселей. Таким образом, в "гладких" регионах, где единственной разницей между значениями соседних пикселей будет шум, этот алгоритм эффективно его сгладит, тогда как в регионах с резкой сменой контраста (например, границей), граница останется нетронутой.
Проблема этого алгоритма заключается в том, что любая разница пикселей считается шумом, то есть текстурированная поверхность будет сглаживаться. В моём примере это мне показалось не так важно, поскольку питон упирался в недостаток оперативной памяти в компьютере и мне пришлось снизить разрешение изображений до 500х375, так что деталей и текстур там и так и так не осталось.
3) **Fast Non-Local Means Denoising**.
![output-denoised-FastNl](https://github.com/chameleon-lizard/OpenCV-stitcher/raw/master/output-denoised-FastNl.jpg "Результат денойзинга с помощью FNLM")
Это стандартный метод денойзинга, применяющийся в OpenCV. Алгоритм состоит в том, что ищутся наиболее похожие пиксели на тот пиксель, который мы обрабатываем, но не только вблизи от пикселя, но и в других местах картинки. В итоге, алгоритм отлично убирает шум, но более требователен к железу. В моём случае, он справился с шумом в правой части стола и на занавесках, но при этом не справился с шумом а области коврика и мыши.
4) **А что будет если не денойзить?**
![output-noisy](https://github.com/chameleon-lizard/OpenCV-stitcher/raw/master/output-noisy.jpg "Результат шумная картинка")
Да ничего хорошего не будет.

### Сшивка видео
![video_stitching_result](https://github.com/chameleon-lizard/OpenCV-stitcher/raw/master/video_stitching_result.png "Результат сшивки видео")

В целом, алгоритм абсолютно тот же. Видео покадрово считываются, при надобности обрабатываются, сшиваются воедино и выводятся на экран.

### Поиск оптического потока
![optical_flow_result](https://github.com/chameleon-lizard/OpenCV-stitcher/raw/master/optical_flow_result.png "Результат поиска оптического потока для сшитых видео")

Это - надстройка над сшивкой видео. Сначала сшивается два кадра из двух видео, переводятся в чб (для повышения производительности это делается до денойза), производится денойз при надобности и картинки сохраняются в виде numpy-массива. Затем происходит то же самое со следующими по порядку кадрами. После этого картинки сравниваются с помощью метода Гуннара-Фарнебака (встроено в OpenCV, подробнее о самом алгоритме чуть дальше). Получившиеся векторы переводятся в цвет (чем ярче, тем длиннее вектор) и, наконец, выводятся.

#### Алгоритм Гуннара-Фарнебака

Описания алгоритма я так нигде и не нашёл (ну, кроме изначальной докторской в 150 страниц), так что я опишу основы.

У нас есть $I(x, y, t)$ - точка, которую мы рассматриваем на первом кадре и $I(x + \delta x, y + \delta y, t + \delta t)$ - точка, которую мы рассматриваем на втором кадре. После разложения в ряд Тейлора и деления на $\delta t$, мы получаем **уравнение оптического потока**:

![equation](https://latex.codecogs.com/svg.latex?\frac{dI}{dx}%20u%20+%20\frac{dI}{dy}%20v%20+%20\frac{dI}{dt}%20=%200, "\frac{dI}{dx} u + \frac{dI}{dy} v + \frac{dI}{dt} = 0,") 

где 

![equation](https://latex.codecogs.com/svg.latex?u%20=%20\frac{dx}{dt},%20v%20=%20\frac{dx}{dt}. "u = \frac{dx}{dt}, v = \frac{dx}{dt}.")

После этого мы смотрим на все точки картинки (а не только на ключевые, как в алгоритме *Лукаса-Канады*), измеряем изменение насыщенности пикселей, переводим в HSV и выводим полученную картинку.

*P.S. Я тут подумал, видео из примера в 30 кадрах в секунду в 720p. В конце, где я ручкой машу, там есть кадры с низким качеством распознавания. Может быть, если поиграться с пирамидкой (снизить разрешение входных кадров, чтобы перемещение пикселей было меньше, есть параметр в функции* `cv2.calcOpticalFlowFarneback`*), качество распознавания будет лучше?*

## Источники
- [Туториал по склейке с pyimagesearch](https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/)
- [Другой туториал, оттуда же, но с разбором того, как оно работает изнутри](https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/)
- [Статья на Википедии про SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)
- [Документация OpenCV с описанием алгоритмов фильтрования картинок](https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html)
- [Документация OpenCV с примером применения FNLMD](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html)
- [Статья с описанием алгоритма FNLMD](http://www.ipol.im/pub/art/2011/bcm_nlm/)
- [Статья с описанием алгоритма билатерального фильтра](http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html)
- [Чей то репозиторий с гитхаба, где тоже склеиваются два видео, но менее красиво, чем у меня и без денойзинга](https://github.com/Toemazz/VideoStitcher)
- [Туториал по OpenCV'шному Dense Optical Flow](https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/?ref=rp)
