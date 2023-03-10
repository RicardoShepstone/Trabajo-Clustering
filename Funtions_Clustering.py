# Autor: Ricardo Shepstone Aramburu

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# Función propia para la visualización de los puntos
def visualize2D(data, instance, coord1, coord2, npoints):
    coord_dict={'X':2,'Y':3,'Z':4}
    plt.ylim(min(data.iloc[instance[0], coord_dict[coord2]::3])-20, max(data.iloc[instance[0], coord_dict[coord2]::3])+20)
    plt.xlim(min(data.iloc[instance[0], coord_dict[coord1]::3])-100, max(data.iloc[instance[0], coord_dict[coord1]::3])+100)
    for i in instance:
        plt.scatter(data.iloc[i, coord_dict[coord1]::3], data.iloc[i, coord_dict[coord2]::3],
                    c=range(npoints), cmap='rainbow')

    plt.colorbar()    
    plt.show()
    

#Función vista en clase para visualizar los clústers
def mostrar(X, c=None, centroids=None,i=0,j=0):

    # Creamos los mapas de colores a utilizar
    cmap_bold = ListedColormap('rainbow')

    # Creamos la figura
    plt.figure(figsize=(10,8))

    # Pintamos los puntos
    plt.scatter(X.iloc[:, i], X.iloc[:, j], c=c, cmap='rainbow', s=60)

    # Pintamos los centroides si los hay 
    # ligero cambios con respecto a la función del notebook anterior
    if centroids is not None:
        plt.scatter(centroids[:,i], centroids[:,j], marker='*', c=range(centroids.shape[0]), s=500)

    # Mostramos la figura
    plt.show()
    

# Función propia para crear un diccionario con los distintos dataframes según 
# el usuario
def divide_df(df, column, keyname):
    df_dict={}
    for value in df[column].unique():
        df_dict[keyname+str(value)]=df[df[column]==value]
    return df_dict


# Función para transformar un dataset al otro formato con las coordenadas
def transform3D_df(df):
    Xlist = df.iloc[:,2::3].values.tolist()
    X = [value for list in Xlist for value in list]
    Ylist = df.iloc[:,3::3].values.tolist()
    Y = [value for list in Ylist for value in list]
    Zlist = df.iloc[:,4::3].values.tolist()
    Z = [value for list in Zlist for value in list]
    point = [p for i in range(df.shape[0]) for p in range(1,12)]
    instance = [i for i in range(df.shape[0]) for j in range(1,12)]
    transformed_df = pd.DataFrame(data=list(zip(instance,point,X,Y,Z)),columns=['instance','point','X','Y','Z'])
    return transformed_df.dropna()