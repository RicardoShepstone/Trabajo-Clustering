# Autor: Ricardo Shepstone Aramburu


# Librerias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

import Funtions_Clustering as FC

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Se Carga el dataset
postures_raw = pd.read_csv('input/Postures.csv', na_values=['?', '', 'NA'])

# Eliminamos la primera observación de índice 0 por tener valores nulos
postures = postures_raw.drop(axis=0, index=0)
postures.head()

# missing values
msno.matrix(postures)

plt.clf()

# Representación de algunas observaciones (útil para el EDA)
FC.visualize2D(postures, [2873], 'X', 'Y',12)

plt.clf()

# Quitamos instancias que tengan las 12 variables
postures = postures.drop(axis=0, index=postures[~postures['X11'].isna()].index)
postures.shape

# Dropeamos las tres útimas variables 
postures=postures.drop(columns=['X11','Y11','Z11'])


####################################### 

# Pruebas para la función de reasignación
# Partimos de un dataframe más pequeño para simplificar el problema
# Filtramos por posición 1 y 2
postures_pos1 = postures[postures['Class'] == 1]
postures_pos2 = postures[postures['Class'] == 2]
# Filtramos por Usuario el primer usuario
postures_pos1_user0 = postures_pos1[postures_pos1['User'] == 0]
postures_pos2_user0 = postures_pos2[postures_pos2['User'] == 0]
# Creamos copia en df
df1 = postures_pos1_user0.copy()
df2 = postures_pos2_user0.copy()

# Trabajamos con df para hacer las pruebas para desarrollar la función
# Con for
X = []
for i in range(df2.shape[0]):
    X.append(df2.iloc[i,2::3].values.tolist())
# Sin for


Xlist = df1.iloc[:,2::3].values.tolist()
X = [value for list in Xlist for value in list]

Ylist = df1.iloc[:,3::3].values.tolist()
Y = [value for list in Ylist for value in list]

Zlist = df1.iloc[:,4::3].values.tolist()
Z = [value for list in Zlist for value in list]

point = [p for i in range(df1.shape[0]) for p in range(1,12)]

df = pd.DataFrame(data=list(zip(point,X,Y,Z)),columns=['point','X','Y','Z'])

train = df[~df['X'].isna()]

##########################################

# Pruebas para kmeans



kmeans = KMeans(n_clusters=5, random_state=1234).fit(train.drop(columns='point'))
centros=kmeans.cluster_centers_
etiquetas=kmeans.labels_

FC.mostrar(train.drop(columns='point') , etiquetas, centros, i=0,j=1)

sum(etiquetas==0)
sum(etiquetas==1)
sum(etiquetas==2)
sum(etiquetas==3)

# Funciona aunque hay que optimizar los parámetros para crear los cluteres

####################################

# Prueba de las funciones creadas para dividir el dataset y modificar su formato

postures_pos1_dict = FC.divide_df(postures[postures['Class'] == 1], 'User', 'User')
postures_pos2_dict = FC.divide_df(postures[postures['Class'] == 2], 'User', 'User')
postures_pos3_dict = FC.divide_df(postures[postures['Class'] == 3], 'User', 'User')
postures_pos4_dict = FC.divide_df(postures[postures['Class'] == 4], 'User', 'User')
postures_pos5_dict = FC.divide_df(postures[postures['Class'] == 5], 'User', 'User')


postures_pos1_transformed = {}
for key in postures_pos1_dict:
    postures_pos1_transformed[key] = FC.transform3D_df(postures_pos1_dict[key])

# Probamos kmeans con distintos datasets del diccionario

# Justificar el uso de distancia eucliniana

# Para user 0 estudiamos los valores de K y vemos que no hay un codo claro 
# User 0 tiene casi 9000 observaciones
error = []
for k in np.arange(1,12):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1234).fit(postures_pos1_transformed['User0'].drop(columns=['point','instance']))
    error.append(kmeans.inertia_)
    centros=kmeans.cluster_centers_
    etiquetas=kmeans.labels_
    FC.mostrar(postures_pos1_transformed['User0'].drop(columns=['point','instance']) , etiquetas, centros, i=0,j=1)

plt.figure(figsize=(10, 8))
plt.plot(error)
plt.show()

# probamos otro usuario con menos observaciones
# User 7 tiene 432
error = []
for k in np.arange(1,12):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1234).fit(postures_pos1_transformed['User7'].drop(columns=['point','instance']))
    error.append(kmeans.inertia_)
    centros=kmeans.cluster_centers_
    etiquetas=kmeans.labels_
    FC.mostrar(postures_pos1_transformed['User7'].drop(columns=['point','instance']) , etiquetas, centros, i=0,j=1)

plt.figure(figsize=(10, 8))
plt.plot(error)
plt.show()

# Probamos otra vez con user 7 normalizado



scaler = MinMaxScaler()
scaler.fit(postures_pos1_transformed['User7'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos1_transformed['User7'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos1_transformed['User7'].drop(columns=['point','instance']).index, 
                       columns=postures_pos1_transformed['User7'].drop(columns=['point','instance']).columns)

# Aplicamos kmeans de nuevo

error = []
for k in np.arange(1,12):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1234).fit(df_norm)
    error.append(kmeans.inertia_)
    centros=kmeans.cluster_centers_
    etiquetas=kmeans.labels_
    FC.mostrar(df_norm , etiquetas, centros, i=0,j=1)

plt.figure(figsize=(10, 8))
plt.plot(error)
plt.show()


# Probamos con la postura 2

postures_pos2_transformed = {}
for key in postures_pos2_dict:
    postures_pos2_transformed[key] = FC.transform3D_df(postures_pos2_dict[key])
    
error = []
for k in np.arange(1,12):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1234).fit(postures_pos2_transformed['User13'].drop(columns=['point','instance']))
    error.append(kmeans.inertia_)
    centros=kmeans.cluster_centers_
    etiquetas=kmeans.labels_
    FC.mostrar(postures_pos2_transformed['User13'].drop(columns=['point','instance']) , etiquetas, centros, i=0,j=1)

plt.figure(figsize=(10, 8))
plt.plot(error)
plt.show()

np.unique(postures_pos2_transformed['User13'].point)


# Probamos con user 0

    
error = []
for k in np.arange(1,12):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1234).fit(postures_pos2_transformed['User0'].drop(columns=['point','instance']))
    error.append(kmeans.inertia_)
    centros=kmeans.cluster_centers_
    etiquetas=kmeans.labels_
    FC.mostrar(postures_pos2_transformed['User0'].drop(columns=['point','instance']) , etiquetas, centros, i=0,j=1)

plt.figure(figsize=(10, 8))
plt.plot(error)
plt.show()

np.unique(postures_pos2_transformed['User0'].point)



# Probamos con normalizado
scaler = MinMaxScaler()
scaler.fit(postures_pos2_transformed['User0'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos2_transformed['User0'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos2_transformed['User0'].drop(columns=['point','instance']).index, 
                       columns=postures_pos2_transformed['User0'].drop(columns=['point','instance']).columns)


# Aplicamos kmeans de nuevo

error = []
for k in np.arange(1,12):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1234).fit(df_norm)
    error.append(kmeans.inertia_)
    centros=kmeans.cluster_centers_
    etiquetas=kmeans.labels_
    FC.mostrar(df_norm , etiquetas, centros, i=0,j=1)

plt.figure(figsize=(10, 8))
plt.plot(error)
plt.show()


# Con user 4


error = []
for k in np.arange(1,12):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1234).fit(postures_pos2_transformed['User4'].drop(columns=['point','instance']))
    error.append(kmeans.inertia_)
    centros=kmeans.cluster_centers_
    etiquetas=kmeans.labels_
    FC.mostrar(postures_pos2_transformed['User4'].drop(columns=['point','instance']) , etiquetas, centros, i=0,j=1)

plt.figure(figsize=(10, 8))
plt.plot(error)
plt.show()

np.unique(postures_pos2_transformed['User4'].point)


# Probamos con normalizado
scaler = MinMaxScaler()
scaler.fit(postures_pos2_transformed['User4'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos2_transformed['User4'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos2_transformed['User4'].drop(columns=['point','instance']).index, 
                       columns=postures_pos2_transformed['User4'].drop(columns=['point','instance']).columns)


# Aplicamos kmeans de nuevo

error = []
for k in np.arange(1,12):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1234).fit(df_norm)
    error.append(kmeans.inertia_)
    centros=kmeans.cluster_centers_
    etiquetas=kmeans.labels_
    FC.mostrar(df_norm , etiquetas, centros, i=0,j=1)

plt.figure(figsize=(10, 8))
plt.plot(error)
plt.show()


#####################################################################

# Si se incluye la variable instance

# Para la postura 2 y user 0
# Aplicamos kmeans de nuevo


error = []
for k in np.arange(1,12):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1234).fit(postures_pos2_transformed['User0'].drop(columns=['point','instance']))
    error.append(kmeans.inertia_)
    centros=kmeans.cluster_centers_
    etiquetas=kmeans.labels_
    FC.mostrar(postures_pos2_transformed['User0'].drop(columns=['point','instance']) , etiquetas, centros, i=0,j=1)

plt.figure(figsize=(10, 8))
plt.plot(error)
plt.show()

np.unique(postures_pos2_transformed['User0'].point)    

# el normalizado hace la variable instance muy pequeña por lo que se hace una clasificación temporal, no queremos eso


# probamos postura 1 usuario 4
error = []
for k in np.arange(1,12):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1234).fit(postures_pos1_transformed['User4'].drop(columns=['point','instance']))
    error.append(kmeans.inertia_)
    centros=kmeans.cluster_centers_
    etiquetas=kmeans.labels_
    FC.mostrar(postures_pos1_transformed['User4'].drop(columns=['point','instance']) , etiquetas, centros, i=0,j=1)

plt.figure(figsize=(10, 8))
plt.plot(error)
plt.show()


#####################################################################


# Hacer para el resto de posturas

# Hacer función que aplique kmeans sobre todos los usuarios y todas las posturas para sacar las medidas de calidad




#####################################################################
# DBSCAN sin la variable instance


scaler = MinMaxScaler()
scaler.fit(postures_pos2_transformed['User0'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos2_transformed['User0'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos2_transformed['User0'].drop(columns=['point','instance']).index, 
                       columns=postures_pos2_transformed['User0'].drop(columns=['point','instance']).columns)


clustering = DBSCAN(eps=0.025, min_samples=6).fit(df_norm)
etiquetas=clustering.labels_
FC.mostrar(df_norm, etiquetas, i=0,j=1)
plt.clf()

# probamos otra postura
# postura 1 usuario 0

scaler = MinMaxScaler()
scaler.fit(postures_pos1_transformed['User0'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos1_transformed['User0'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos1_transformed['User0'].drop(columns=['point','instance']).index, 
                       columns=postures_pos1_transformed['User0'].drop(columns=['point','instance']).columns)


# parece que min_samples va bien con 2*ndim
clustering = DBSCAN(eps=0.025, min_samples=6).fit(df_norm)
etiquetas=clustering.labels_
FC.mostrar(df_norm, etiquetas, i=0,j=1)
plt.clf()

# para devolver las etiquetas y el número de veces que aparece cada una
value, counts = np.unique(etiquetas, return_counts=True)
len(etiquetas)
counts>0.1*len(etiquetas)


np.unique(postures_pos1_transformed['User0'].iloc[:,1].values, return_counts=True)

# probamos min_samples con un valor pequeño para que descarte outliers, queremos que tenga 7 clusters

clustering = DBSCAN(eps=0.04, min_samples=round(0.005*df_norm.shape[0])).fit(df_norm)
etiquetas=clustering.labels_
FC.mostrar(df_norm, etiquetas, i=0,j=1)
plt.clf()

# vemos cuántos clusters se han formado
np.unique(etiquetas, return_counts=True)[1]


#S método del codo
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=20)
neighbors_fit = neighbors.fit(df_norm)
distances, indices = neighbors_fit.kneighbors(df_norm)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)



# como encontrar el radio óptimo

optimum_radius = []

for radius in np.arange(0.01, 0.1,0.001):
    clustering = DBSCAN(eps=radius, min_samples=6).fit(df_norm)
    etiquetas=clustering.labels_
    FC.mostrar(df_norm, etiquetas, i=0,j=1)
    
    if sum(np.unique(etiquetas, return_counts=True)[1]>0.01*len(etiquetas)) == 7:
        optimum_radius.append(radius)




# probamos la optimización con usuario 4
scaler = MinMaxScaler()
scaler.fit(postures_pos1_transformed['User4'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos1_transformed['User4'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos1_transformed['User4'].drop(columns=['point','instance']).index, 
                       columns=postures_pos1_transformed['User4'].drop(columns=['point','instance']).columns)

sum(np.unique(postures_pos1_transformed['User4'].iloc[:,1].values, return_counts=True)[1]>0.01*postures_pos1_transformed['User4'].shape[0])

optimum_radius = []
for radius in np.arange(0.01, 0.2,0.001):
    clustering = DBSCAN(eps=radius, min_samples=6).fit(df_norm)
    etiquetas=clustering.labels_
    FC.mostrar(df_norm, etiquetas, i=0,j=1)
    
    if sum(np.unique(etiquetas, return_counts=True)[1]>0.01*len(etiquetas)) == 5:
        optimum_radius.append(radius)


# probamos user 5

scaler = MinMaxScaler()
scaler.fit(postures_pos1_transformed['User5'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos1_transformed['User5'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos1_transformed['User5'].drop(columns=['point','instance']).index, 
                       columns=postures_pos1_transformed['User5'].drop(columns=['point','instance']).columns)

sum(np.unique(postures_pos1_transformed['User5'].iloc[:,1].values, return_counts=True)[1]>0.01*postures_pos1_transformed['User5'].shape[0])


optimum_radius = []
for radius in np.arange(0.01, 0.3,0.001):
    clustering = DBSCAN(eps=radius, min_samples=6).fit(df_norm)
    etiquetas=clustering.labels_
    FC.mostrar(df_norm, etiquetas, i=0,j=1)
    
    if sum(np.unique(etiquetas, return_counts=True)[1]>0.01*len(etiquetas)) == 6:
        optimum_radius.append(radius)
        
clustering = DBSCAN(eps=np.mean(optimum_radius), min_samples=6).fit(df_norm)
etiquetas=clustering.labels_
FC.mostrar(df_norm, etiquetas, i=0,j=1)

# User 5 de la postura 2
np.unique(postures_pos2_transformed['User5'].iloc[:,1].values, return_counts=True)[1]
postures_pos2_transformed['User5'].shape[0]
sum(np.unique(postures_pos2_transformed['User5'].iloc[:,1].values, return_counts=True)[1]>0.01*postures_pos2_transformed['User5'].shape[0])

scaler = MinMaxScaler()
scaler.fit(postures_pos2_transformed['User5'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos2_transformed['User5'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos2_transformed['User5'].drop(columns=['point','instance']).index, 
                       columns=postures_pos2_transformed['User5'].drop(columns=['point','instance']).columns)

optimum_radius = []
for radius in np.arange(0.01, 0.03,0.001):
    clustering = DBSCAN(eps=radius, min_samples=6).fit(df_norm)
    etiquetas=clustering.labels_
    FC.mostrar(df_norm, etiquetas, i=0,j=1)
    
    if sum(np.unique(etiquetas, return_counts=True)[1]>0.01*len(etiquetas)) == 10:
        optimum_radius.append(radius)
        
clustering = DBSCAN(eps=np.mean(optimum_radius), min_samples=6).fit(df_norm)
etiquetas=clustering.labels_
FC.mostrar(df_norm, etiquetas, i=0,j=1)



# Para el usuario 7

np.unique(postures_pos2_transformed['User7'].iloc[:,1].values, return_counts=True)[1]
postures_pos2_transformed['User7'].shape[0]
sum(np.unique(postures_pos2_transformed['User7'].iloc[:,1].values, return_counts=True)[1]>0.01*postures_pos2_transformed['User7'].shape[0])


scaler = MinMaxScaler()
scaler.fit(postures_pos2_transformed['User7'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos2_transformed['User7'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos2_transformed['User7'].drop(columns=['point','instance']).index, 
                       columns=postures_pos2_transformed['User7'].drop(columns=['point','instance']).columns)

optimum_radius = []
for radius in np.arange(0.01, 0.1,0.001):
    clustering = DBSCAN(eps=radius, min_samples=6).fit(df_norm)
    etiquetas=clustering.labels_
    FC.mostrar(df_norm, etiquetas, i=0,j=1)
    
    if sum(np.unique(etiquetas, return_counts=True)[1]>0.001*len(etiquetas)) == 10:
        optimum_radius.append(radius)
        
clustering = DBSCAN(eps=np.mean(optimum_radius), min_samples=6).fit(df_norm)
etiquetas=clustering.labels_
FC.mostrar(df_norm, etiquetas, i=0,j=1)






from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

np.unique(postures_pos2_transformed['User12'].iloc[:,1].values, return_counts=True)[1]
postures_pos2_transformed['User12'].shape[0]
sum(np.unique(postures_pos2_transformed['User12'].iloc[:,1].values, return_counts=True)[1]>0.01*postures_pos2_transformed['User12'].shape[0])


scaler = MinMaxScaler()
scaler.fit(postures_pos2_transformed['User12'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos2_transformed['User12'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos2_transformed['User12'].drop(columns=['point','instance']).index, 
                       columns=postures_pos2_transformed['User12'].drop(columns=['point','instance']).columns)

optimum_radius = []
optimum_min_samples = []
for radius in np.arange(0.01, 0.05,0.005):
    for samples in range(6,58,4):
        clustering = DBSCAN(eps=radius, min_samples=50).fit(df_norm)
        etiquetas=clustering.labels_
        FC.mostrar(df_norm, etiquetas, i=0,j=1)

    
        if sum(np.unique(etiquetas, return_counts=True)[1]>0.01*len(etiquetas)) == 11:
            optimum_radius.append(radius)
            optimum_min_samples.append(samples)

clustering = DBSCAN(eps=np.mean(optimum_radius), min_samples=round(np.mean(optimum_min_samples))).fit(df_norm)
etiquetas=clustering.labels_
FC.mostrar(df_norm, etiquetas, i=0,j=1)
###############################

# prueba para automatizar kmeans


kmeans = KMeans(n_clusters=8, n_init=100, random_state=1234).fit(df_norm)
centros=kmeans.cluster_centers_
etiquetas=kmeans.labels_
davies_bouldin_score(df_norm, etiquetas)
silhouette_score(df_norm, etiquetas)

##############################

# más pruebas

np.unique(postures_pos2_transformed['User7'].iloc[:,1].values, return_counts=True)[1]
postures_pos2_transformed['User7'].shape[0]
sum(np.unique(postures_pos2_transformed['User7'].iloc[:,1].values, return_counts=True)[1]>0.01*postures_pos2_transformed['User7'].shape[0])


scaler = MinMaxScaler()
scaler.fit(postures_pos2_transformed['User7'].drop(columns=['point','instance']))
results_norm = scaler.transform(postures_pos2_transformed['User7'].drop(columns=['point','instance']))
df_norm = pd.DataFrame(results_norm, index=postures_pos2_transformed['User7'].drop(columns=['point','instance']).index, 
                       columns=postures_pos2_transformed['User7'].drop(columns=['point','instance']).columns)

optimum_radius = []
optimum_min_samples = []
for radius in np.arange(0.01, 0.3,0.002):
    for samples in range(6,58,4):
        clustering = DBSCAN(eps=radius, min_samples=50).fit(df_norm)
        etiquetas=clustering.labels_
        FC.mostrar(df_norm, etiquetas, i=0,j=1)

    
        if sum(np.unique(etiquetas, return_counts=True)[1]>0.01*len(etiquetas)) == 10:
            optimum_radius.append(radius)
            optimum_min_samples.append(samples)

clustering = DBSCAN(eps=np.mean(optimum_radius), min_samples=round(np.mean(optimum_min_samples))).fit(df_norm)
etiquetas=clustering.labels_
FC.mostrar(df_norm, etiquetas, i=0,j=1)
