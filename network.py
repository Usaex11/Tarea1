
import random

import numpy as np

class Network(object):






    def __init__(self, sizes):  # Inicializa una instancia de la red neuronal con capas especificadas por sizes
        self.num_layers = len(sizes)
        self.sizes = sizes
        # crea biases y weights de la red de manera aleatoria
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #es una lista de matrices de bias una para cada capa excepto la primera
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #es una lista de matrices de los weights una para cada conexión entre capas 
        self.v_w = [np.zeros(w.shape) for w in self.weights] #hay una velocidad asosiada a cada peso y a cada bias con lo que deberian de tener la misma forma
        self.v_b = [np.zeros(b.shape) for b in self.biases]






    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights): #es un bucle que itera sobre las listas self.biases y 
            #self.weights b son bias de una capa y w son los pesos de las conexiones entre capas
            a = sigmoid(np.dot(w, a)+b) #calcula la salida de la capa utilizando la función de activación sigmoide, np.dot(w, a) 
            #es el producto punto entre la matriz de pesos w y la entrada a y se le suma la matriz de bias b y el resultado 
            #se vuelve a introducir al siguiente bucle
        return a #seria el resultado final en la ultima capa







    def SGD(self, training_data, epochs, mini_batch_size, eta, mo, test_data=None):
        if test_data:
            test_data = list(test_data) #crea ina lista de los datos de prueba
            n_test = len(test_data) #obtenemos la longitus de los datos de prueba

        training_data = list(training_data) #convierte los datos de entrenamiento en una lista
        n = len(training_data) #longitus de los datos de entrenamiento
        
        for j in range(epochs): #es un ciclo que durara la cantidad de epocas que le pongamos 
            random.shuffle(training_data) #barajea aleatoriamente los datos de entrenamiento
            
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] #divide los datos de entrenamiento en pequeños paqquetes de tamaño "mini_batch_size"
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, mo) #se le tuvo que agregar el momento para que pues funcione
            
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))#imprime cuantos aciertos tuvo la red en cada epoca
            
            else:
                print("Epoch {0} complete".format(j)) #si no hay datos de entrenamiento dice que ya termino






    def update_mini_batch(self, mini_batch, eta, mo): #parametro mo de momento xd
        nabla_b = [np.zeros(b.shape) for b in self.biases] #crea una matriz de 0 de la misma forma que la matriz de biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] #crea una matriz de 0 de la misma forma que la matriz de pesos
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #calcula los gradientes con la funcion backdrop
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #una lista de los gradientes de los bias
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #un alista de los gradientes de los pesos
        
        self.v_w= [mo*vw-(eta/len(mini_batch))*nw for vw, nw in zip(self.v_w, nabla_w)] #es la lista que actualiza las velocidades esta construida
        #de esta mnera porque segun yo se tiene que actualizar al mismo tiempo que se modifican los pesos
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] #modifica los pesos 
        
        self.v_b= [mo*vb-(eta/len(mini_batch))*nb for vb, nb in zip(self.v_b, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)] #modifica los bias 






    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases] #igual crea una lista de matrices con 0 de la forma de los biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] #hace lo mismo pero con los pesos
        # feedforward
        activation = x 
        activations = [x] #guanda todas las activaciones capa por capa
        zs = [] #lista de todos los vectores z
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b #es la entrada de la funcion de activacion
            zs.append(z) #va guardadno las Z en la lista zs
            activation = sigmoid(z) #se le mete a la funcion sigmoide la z calculada
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  #calcula el error en la capa de salida multiplicando la 
        #derivada de la función de costo por la función de activacion
        nabla_b[-1] = delta #guasrda el error
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #transpone y hace producto punto entre delta y las activaciones
        for l in range(2, self.num_layers):
            z = zs[-l] #el valor de z mas a la "derecha" o del final
            sp = sigmoid_prime(z) #evalua z en la derivada de la funcion de costo
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #error de la capa en la que va el ciclo
            nabla_b[-l] = delta #almacena el error
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) 
        return (nabla_b, nabla_w)



    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]  #crea una tupla(x,y) con x siendo la neurona con mayor activacion
        return sum(int(x == y) for (x, y) in test_results) #compara si es que X es igual a Y y regresa esos datos





    #def cost_derivative(self, output_activations, y):
        #return (output_activations-y) #derivada de la funcion de costo como para los pesos y los bias es la misma solo se necesita una funcion
    def cost_derivative(self, output_activations, y, ep=1e-9): 
        N=output_activations.shape[0]
        cost=np.sum(y*np.log(output_activations+ep))/N #La furmula matematica es sum(todas las anteriores predicciones log de los outputs de las neuronas)
        #al entrenar demasiado lento encontre que si se dividia entre N(numero de outputs de las neuronas) y se le agregaba un epsilon para que no empezara en 0 y fuer mas rapido era una solucion
        return (cost)
        





def sigmoid(z):
    return 1.0/(1.0+np.exp(-z)) #es la neurona sigmoide






def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z)) #es la derivada de la neurona sigmoide
