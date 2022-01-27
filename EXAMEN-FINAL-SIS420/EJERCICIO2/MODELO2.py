# utilizado para manejos de directorios y rutas
import os

# Computacion vectorial y cientifica para python
import numpy as np

# Librerias para graficación (trazado de gráficos)
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # Necesario para graficar superficies 3D


#al cargar los datos cambiar a la direccion a la actual  donde se encuentra el texto 
data = np.loadtxt(os.path.join(r'C:\Users\Gabo\Desktop\Nueva carpeta\EXAMEN-FINAL-SIS420\EJERCICIO2','DATASETAUTOSCLASIFICADOS11.txt'), delimiter=',')

X=data[:,0:16]
Y=data[:,16]

def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma

X_norm,mu,sigma=featureNormalize(X)

#print(X)
#print(Y)

m=Y.size
print('m = ',m)

X = np.concatenate([np.ones((m, 1)),X_norm], axis=1)


def computeCost(X, y, theta):
    # inicializa algunos valores importantes
    m = y.size  # numero de ejemplos de entrenamiento
    
    J = 0
    #h = np.dot(X, theta)
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
        # Inicializa algunos valores importantes
    m = y.shape[0]  # numero de ejemplos de entrenamiento
   
    # hace una copia de theta, para evitar cambiar la matriz original, 
    # ya que las matrices numpy se pasan por referencia a las funciones

    theta = theta.copy()
    
    J_history = [] # Lista que se utiliza para almacenar el costo en cada iteración
    
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history


theta = np.zeros(17)
# configuraciones para el descenso por el gradiente
iterations = 15000
alpha = 0.003

theta, J_history = gradientDescent(X ,Y, theta, alpha, iterations)

#print('Theta encontrada por descenso gradiente: {:.4f}, {:.4f}, {:.4f}'.format(*theta))


print('costo calculado con theta = ', computeCost(X,Y,theta))

# pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
# pyplot.xlabel('Numero de iteraciones')
# pyplot.ylabel('Costo J')
#pyplot.show()



print('ECUACION DE LA NORMAL')

thetaa = np.zeros(17)
def normalEqn(X, y):
  
    thetaa = np.zeros(X.shape[1])
    
    thetaa = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    
    return thetaa

thetaa=normalEqn(X,Y)
J=computeCost(X,Y,thetaa)
print('costo = ', J)

#y_h=np.dot(X,thetaa)
#print(y_h)


#https://www.autopia.com.bo/auto/hatchback-usados/suzuki-sx4-2011-61f0985406a5d26618234e23

x1=[1,33,2011,5,5,2,1,2,1,47400,8,3,1600,1,1,1,2]
x1[1:17]=(x1[1:17]-mu)/sigma
print('auto suzuki 2011')
print(np.dot(x1,thetaa))

#https://www.autopia.com.bo/auto/suv-usados/bmw-x3-2011-61e98b7cfd703041232cd383

x2=[1,17,2011,5,5,2,2,1,1,78916,14,4,3000,1,1,1,1]
x2[1:17]=(x2[1:17]-mu)/sigma
print('vagoneta bmw 2011')
print(np.dot(x2,thetaa))

#https://www.autopia.com.bo/auto/hatchback-usados/toyota-corolla-1998-61b83bbd9c25881efd324653

x3=[1,2,1998,5,5,4,1,1,1,281260,9,4,1600,1,3,2,2]
x3[1:17]=(x3[1:17]-mu)/sigma
print('aurto toyota 1998')
print(np.dot(x3,thetaa))