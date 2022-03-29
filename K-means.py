import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

dataset = np.array(
#matriz com as coordenadas dos item
[[-25, -46], 
[-22, -43],
[-25, -49], 
[-30, -51],
[-19, -43], 
[-15, -47], 
[-12, -38], 
[-8, -34], 
[-16, -49], 
[-3, -60], 
[-22, -47], 
[-3, -38],
[-21, -47], 
[-23, -51], 
[-27, -48], 
[-21, -43], 
[-1, -48], 
[-10, -67], 
[-8, -63] 
])

#Printa o grafico

plt.scatter(dataset[:,1], dataset[:,0]) #posicionamento dos eixos x e y
plt.xlim(-75, -30) #range do eixo x
plt.ylim(-50, 10) #range do eixo y
plt.grid() #função que desenha a grade no nosso gráfico

kmeans = KMeans(n_clusters = 4, #numero de clusters
init = 'k-means++', n_init = 10, #algoritmo que define a posição dos clusters de maneira mais assertiva
max_iter = 300) #numero máximo de iterações
pred_y = kmeans.fit_predict(dataset)
plt.scatter(dataset[:,1], dataset[:,0], c = pred_y) #posicionamento dos eixos x e y
plt.xlim(-75, -30) #range do eixo x
plt.ylim(-50, 10) #range do eixo y
plt.grid() #função que desenha a grade no nosso gráfico
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0], s = 70, c = 'red') #posição de cada centroide no gráfico
plt.show()
