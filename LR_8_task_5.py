import cv2 as cv  # Імпортуємо бібліотеку OpenCV
import numpy as np  # Імпортуємо бібліотеку NumPy
from matplotlib import pyplot as plt  # Імпортуємо pyplot з бібліотеки Matplotlib

img = cv.imread('test_foto.jpg',0)  # Зчитуємо основне зображення в градаціях сірого
img2 = img.copy()  # Створюємо копію основного зображення
template = cv.imread('test_foto_fase.jpg',0)  # Зчитуємо шаблон зображення в градаціях сірого
w, h = template.shape[::-1]  # Отримуємо ширину та висоту шаблону зображення

# Список всіх 6 методів для порівняння
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for meth in methods:  # Проходимо кожен метод
 img = img2.copy()  # Копіюємо основне зображення
 method = eval(meth)  # Оцінюємо метод

 # Застосовуємо шаблонний матчинг
 res = cv.matchTemplate(img,template,method)  # Застосовуємо шаблонний матчинг з вибраним методом
 min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)  # Знаходимо мінімальні та максимальні значення пікселів та їх розташування

 # Якщо метод TM_SQDIFF або TM_SQDIFF_NORMED, беремо мінімум
 if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
  top_left = min_loc
 else:
  top_left = max_loc
 bottom_right = (top_left[0] + w, top_left[1] + h)

 cv.rectangle(img,top_left, bottom_right, 255, 2)  # Малюємо прямокутник на знайденій області

 plt.subplot(121),plt.imshow(res,cmap = 'gray')  # Відображаємо результат матчингу
 plt.title('Результат матчингу'), plt.xticks([]), plt.yticks([])  # Встановлюємо заголовок для осей
 plt.subplot(122),plt.imshow(img,cmap = 'gray')  # Відображаємо зображення з виявленою точкою
 plt.title('Виявлена точка'), plt.xticks([]), plt.yticks([])  # Встановлюємо заголовок для осей
 plt.suptitle(meth)  # Встановлюємо заголовок для всього вікна
 plt.show()  # Показуємо всі відкриті фігури
