import numpy as np
import cv2
from matplotlib import pyplot as plt

# Завантажте зображення
img = cv2.imread('coins_2.jpg')
cv2.imshow("coins_2",img)
cv2.waitKey(0)

# Перетворіть зображення в сіре
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Бінаризуйте зображення
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("coins bin ",thresh)
cv2.waitKey(0)

# Видалення шуму
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# Певна фонова область
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Пошук впевненої області переднього плану
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Пошук невідомого регіону
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

cv2.imshow("coins ", opening)
cv2.waitKey(0)

# Маркування міток
ret, markers = cv2.connectedComponents(sure_fg)

# Додайте один до всіх міток, щоб впевнений фон був не 0, а 1
markers = markers+1

# Тепер позначте область невідомого нулем
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

# Створіть словник для зберігання площі кожної монети
areas = {}
for i in range(np.max(markers)):
    areas[i] = np.sum(markers == i)

# Визначте порогове значення для розміру монети
coin_size_threshold = 1000

# Пройдіться по всіх монетах і змініть колір відповідно до їх розміру
for i in range(1, np.max(markers)+1):
    if areas[i] > coin_size_threshold:
        img[markers == i] = [255, 0, 0]  # Великі монети
    else:
        img[markers == i] = [0, 255, 0]  # Малі монети

cv2.imshow("coins_markers",img)
cv2.waitKey(0)
