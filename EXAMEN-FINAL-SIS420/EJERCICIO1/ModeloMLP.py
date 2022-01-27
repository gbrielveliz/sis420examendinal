# utilizado para manejos de directorios y rutas
import os

# Computacion vectorial y cientifica para python
import numpy as np

# Librerias para graficación (trazado de gráficos)
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # Necesario para graficar superficies 3D


class MLP:
    def __init__(self, layers):
        # el MLP es una lista de capas
        self.layers = layers

    def __call__(self, x):
        # calculamos la salida del modelo aplicando
        # cada capa de manera secuencial
        for layer in self.layers:
            x = layer(x)
        return x


class Layer():
    def __init__(self):
        self.params = []
        self.grads = []

    def __call__(self, x):
        # por defecto, devolver los inputs
        # cada capa hará algo diferente aquí
        return x

    def backward(self, grad):
        # cada capa, calculará sus gradientes
        # y los devolverá para las capas siguientes
        return grad

    def update(self, params):
        # si hay parámetros, los actualizaremos
        # con lo que nos de el optimizer
        return
class Linear(Layer):
    def __init__(self, d_in, d_out):
        # pesos de la capa
        self.w = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2/(d_in+d_out)),
                                  size=(d_in, d_out))
        self.b = np.zeros(d_out)

    def __call__(self, x):
        self.x = x
        self.params = [self.w, self.b]
        # salida del preceptrón
        return np.dot(x, self.w) + self.b    
    
    def backward(self, grad_output):
        # gradientes para la capa siguiente (BACKPROP)
        grad = np.dot(grad_output, self.w.T)
        self.grad_w = np.dot(self.x.T, grad_output)
        # gradientes para actualizar pesos
        self.grad_b = grad_output.mean(axis=0)*self.x.shape[0]
        self.grads = [self.grad_w, self.grad_b]
        return grad

    def update(self, params):
        self.w, self.b = params
class ReLU(Layer):
    def __call__(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad = self.x > 0
        return grad_output*grad


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=-1,keepdims=True)

class Sigmoid(Layer):    
    def __call__(self, x):
        self.x = x
        return sigmoid(x)

    def backward(self, grad_output):
        grad = sigmoid(self.x)*(1 - sigmoid(self.x))
        return grad_output*grad


class SGD():
    def __init__(self, net, lr):
        self.net = net
        self.lr = lr

    def update(self):
        for layer in self.net.layers:
            layer.update([
                params - self.lr*grads
                for params, grads in zip(layer.params, layer.grads)
            ])



class Loss():
    def __init__(self, net):
        self.net = net

    def backward(self):
        # derivada de la loss function con respecto 
        # a la salida del MLP
        grad = self.grad_loss()
        # BACKPROPAGATION
        for layer in reversed(self.net.layers):
            grad = layer.backward(grad)        
class MSE(Loss):
    def __call__(self, output, target):
        self.output, self.target = output, target.reshape(output.shape)
        loss = np.mean((self.output - self.target)**2)
        return loss.mean()

    def grad_loss(self):
        return self.output -  self.target  
class BCE(Loss):
    def __call__(self, output, target):
        self.output, self.target = output, target.reshape(output.shape)
        loss = - np.mean(self.target*np.log(self.output) - (1 - self.target)*np.log(1 - self.output))
        return loss.mean()

    def grad_loss(self):
        return self.output -  self.target          
class CrossEntropy(Loss):
    def __call__(self, output, target):
        self.output, self.target = output, target
        logits = output[np.arange(len(output)), target]
        loss = - logits + np.log(np.sum(np.exp(output), axis=-1))
        loss = loss.mean()
        return loss

    def grad_loss(self):
        answers = np.zeros_like(self.output)
        answers[np.arange(len(self.output)), self.target] = 1
        return (- answers + softmax(self.output)) / self.output.shape[0]


#al cargar los datos cambiar a la direccion a la actual  donde se encuentra el texto 
data = np.loadtxt(os.path.join(r'C:\Users\Gabo\Desktop\Nueva carpeta\EXAMEN-FINAL-SIS420\EJERCICIO1','DATASETAUTOSCLASIFICADOS22.txt'), delimiter=',')


X=data[:,0:17]
Y=data[:,17]

def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma

X_norm,mu,sigma=featureNormalize(X)
X=X_norm
#print(X)
#print(Y)
m=Y.size
print('m = ',m)


#H=20, LR=0.0003 , EPOCHS=200, BATCH SIZE = 64, 



D_in, H, D_out = 17, 100, 1
# añadimos más capas
mlp = MLP([

    #Linear(D_in, D_out),
    Linear(D_in, H),
    ReLU(),
    Linear(H, H),
    ReLU(),Linear(H, H),
    ReLU(),
    Linear(H, H),
    ReLU(),
    Linear(H, D_out),
    Sigmoid()
    
])

optimizer = SGD(mlp, lr=0.001)

loss=BCE(mlp)

epochs = 300
batch_size = 64

batches = len(X) // batch_size
log_each = 10
l = []
for e in range(1,epochs+1):
    _l = []
    for b in range(batches):
        x = X[b*batch_size:(b+1)*batch_size]
        y = Y[b*batch_size:(b+1)*batch_size] 
        y_pred = mlp(x)    
        _l.append(loss(y_pred, y))
        loss.backward()    
        optimizer.update()
    l.append(np.mean(_l))
    if not e % log_each:
        print(f'Epoch {e}/{epochs}, Loss: {np.mean(l):.4f}')


y_predic=mlp(X)

def predicion(Threshold,y_h):
    for i in range (0,len(y_h)):
        if y_h[i] >Threshold:
            y_h[i]=1
        else:
            y_h[i]=0
    return y_h


Threshold=0.5
y_p=y_predic.copy()
y_h=predicion(Threshold,y_p)

print(y_h.T)
print(Y)
def accuracy(y_h,Y):
    return np.mean(y_h.T== Y)

print('accuracy = ',accuracy(y_h,Y))
#print('Precision del conjuto de entrenamiento: {:.2f}%'.format(np.mean(y_h.T == Y) ))

def evaluacion(y_h,Y):
    #verdaderoPositivo
    TP=0
    #verdaderoNegativo
    TN=0
    #falsoPositivo
    FP=0
    #falsoNegativo
    FN=0

    for i in range (0,len(y_h)):
        if y_h[i]==1 and Y[i]==1:
            TP+=1
        if y_h[i]==0 and Y[i]==0:
            TN+=1
        if y_h[i]==1 and Y[i]==0:
            FP+=1
        if y_h[i]==0 and Y[i]==1:
            FN+=1
    return TP,TN,FP,FN


TP,TN,FP,FN=evaluacion(y_h,Y)
#matriz de confusion 
CM=[[TN,FP],
    [FN,TP]]

print('MATRIZ DE CONFUSION')
print('[TN,FP],[FN,TP]')
print(CM)

def precision(TP,FP):
    PRECISION = TP /(TP+FP)
    return PRECISION

def recall(TP,FN):
    RECALL = TP /(TP+FN)
    return RECALL

PRECISION=precision(TP,FP)
RECALL=recall(TP,FN)
print('precision  reacall ')
print(PRECISION,RECALL)


for threshold in np.linspace(0.1,0.8,5):
    y_predicion=y_predic.copy()
    y_hh=predicion(threshold,y_predicion)
    tp,tn,fp,fn=evaluacion(y_hh,Y)
    precisionn=precision(tp,fp)
    recalll=recall(tp,fn)
    print('threshold = ',threshold,'  precision = ', precisionn,' recall = ', recalll )


#nueva prediccion
threshold=0.625
y_predicion=y_predic.copy()
y_hh=predicion(threshold,y_predicion)
tp,tn,fp,fn=evaluacion(y_hh,Y)


cm=[[tn,fp],
    [fn,tp]]

print('MATRIZ DE CONFUSION CON THRESHOLD = 0.625')
print(cm)



# #https://www.autopia.com.bo/auto/hatchback-usados/suzuki-sx4-2011-61f0985406a5d26618234e23
# x1=[33,2011,5,5,2,1,2,1,47400,8,3,1600,1,1,1,2,12400]
# x1=(x1-mu)/sigma
# y1=mlp(x1)

# #https://www.autopia.com.bo/auto/suv-usados/bmw-x3-2011-61e98b7cfd703041232cd383
# x2=[17,2011,5,5,2,2,1,1,78916,14,4,3000,1,1,1,1,28000]
# x2=(x2-mu)/sigma
# y2=mlp(x2)

# #https://www.autopia.com.bo/auto/hatchback-usados/toyota-corolla-1998-61b83bbd9c25881efd324653
# x3=[2,1998,5,5,4,1,1,1,281260,9,4,1600,1,3,2,2,10000]
# x3=(x3-mu)/sigma
# y3=mlp(x3)


# x4=[10,2000,5,5,4,1,1,1,28126,9,4,1000,1,3,2,2,35000]
# x4=(x4-mu)/sigma
# y4=mlp(x4)
# x5=[17,1990,5,5,2,2,1,1,78916,14,4,1000,1,1,1,1,28000]
# x5=(x5-mu)/sigma
# y5=mlp(x5)
# x6=[33,2011,5,5,2,1,2,1,47400,8,3,1000,1,1,1,2,40400]
# x6=(x6-mu)/sigma
# y6=mlp(x6)
# #print(y1,y2,y3,y4,y5,y6)
# print(predicion(threshold,y1))
# print(predicion(threshold,y2))
# print(predicion(threshold,y3))
# print(predicion(threshold,y4))
# print(predicion(threshold,y5))
# print(predicion(threshold,y6))




