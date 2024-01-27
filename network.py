"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):  # Inicializa una instancia de la red neuronal con capas especificadas por sizes
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        # crea biases y weights de la red de manera aleatoria
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #es una lista de matrices de bias una para cada capa excepto la primera
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #es una lista de matrices de los weights una para cada conexión entre capas 
        


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights): #es un bucle que itera sobre las listas self.biases y 
            #self.weights b son bias de una capa y w son los pesos de las conexiones entre capas
            a = sigmoid(np.dot(w, a)+b) #calcula la salida de la capa utilizando la función de activación sigmoide, np.dot(w, a) 
            #es el producto punto entre la matriz de pesos w y la entrada a y se le suma la matriz de bias b y el resultado 
            #se vuelve a introducir al siguiente bucle
        return a #seria el resultado final en la ultima capa

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            test_data = list(test_data) #crea ina lista de los datos de prueba
            n_test = len(test_data) #obtenemos la longitus de los datos de prueba

        training_data = list(training_data) #convierte los datos de entrenamiento en una lista
        n = len(training_data) #longitus de los datos de entrenamiento
        for j in range(epochs): #es un ciclo que durara la cantidad de epocas que le pongamos 
            random.shuffle(training_data) #barajea aleatoriamente los datos de entrenamiento
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] #divide los datos de entrenamiento en pequeños paqquetes de tamaño "mini_batch_size"
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) #simplemente invoca al metodo update_mini_batch
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))#imprime cuantos aciertos tuvo la red en cada epoca
            else:
                print("Epoch {0} complete".format(j)) #si no hay datos de entrenamiento dice que ya termino

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #crea una matriz de 0 de la misma forma que la matriz de biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] #crea una matriz de 0 de la misma forma que la matriz de pesos
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #calcula los gradientes con la funcion backdrop
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #una lista de los gradientes de los bias
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #un alista de los gradientes de los pesos
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] #modifica los pesos 
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)] #modifica los bias 

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
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
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) #calcula el error en la capa de salida multiplicando la 
        #derivada de la función de costo por la función de activacion
        nabla_b[-1] = delta #guasrda el error
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #transpone y hace producto punto entre delta y las activaciones
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l] #el valor de z mas a la "derecha" o del final
            sp = sigmoid_prime(z) #evalua z en la derivada de la funcion de costo
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #error de la capa en la que va el ciclo
            nabla_b[-l] = delta #almacena el error
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) 
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]  #crea una tupla(x,y) con x siendo la neurona con mayor activacion
        return sum(int(x == y) for (x, y) in test_results) #compara si es que X es igual a Y y regresa esos datos

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) #resta output_activations "y" ???? 

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z)) #es la neurona sigmoide

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)) #es la derivada de la neurona sigmoide
