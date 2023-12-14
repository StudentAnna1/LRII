# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from mplfinance.original_flavor import candlestick_ohlc
import json
# Завантаження даних фондового ринку з пакету mplfinance
start_date = (2008, 1, 1)
end_date = (2008, 3, 1)
quotes = candlestick_ohlc('INTC', start_date, end_date)
# Вирахування варіації котирувань між відкриттям і закриттям біржі
variation = np.array([quote.open - quote.close for quote in quotes])
# Створення матриці подібності на основі відстані між варіаціями
similarity = -np.abs(np.subtract.outer(variation, variation))
# Створення об'єкта моделі поширення подібності
af = AffinityPropagation(affinity='precomputed')
# Навчання моделі на даних
af.fit(similarity)
# Отримання міток кластерів для даних
y_pred = af.labels_
# Отримання координат центрів кластерів
centroids = af.cluster_centers_indices_
# Вивід кількості кластерів
print("Кількість кластерів =", len(centroids))
# Завантаження файлу з прив'язками символічних позначень компаній до повних назв
with open('company_symbol_mapping.json') as f:
    company_names = json.load(f)
# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot([quote.date for quote in quotes], variation, 'o')
plt.xlabel('Дата')
plt.ylabel('Варіація котирувань')
plt.title('Кластеризація фондового ринку за допомогою моделі поширення подібності')
for i in range(len(centroids)):
    # Виділення точок, що належать одному кластеру
    cluster = np.where(y_pred == i)[0]
    # Вивід назв компаній, що належать одному кластеру
    print("Кластер", i+1, ":")
    for j in cluster:
        print(company_names[quotes[j].ticker])
    # Підписання центрів кластерів
    plt.text(quotes[centroids[i]].date, variation[centroids[i]], 'C'+str(i+1), fontsize=16, color='red')
# Побудова свічкового графіку з шириною свічок 0.2
candlestick_ohlc(plt.gca(), quotes, width=0.2)
plt.show()
