from Tensor import *
import numpy as np

# Слой
class Layer(object):
    
    def __init__(self):
        self.parameters = list()
        
    def get_parameters(self):
        return self.parameters



# Линейный слой
class Linear(Layer):

    """
    Количество входов и выходов
    """
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        
		# Рандомные значения
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / (n_inputs))
        self.weight = Tensor(W, autograd=True)
        
		# Смещение
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)
        
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    # Умножаем на веса
    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))

# Слой с функцией активации tanh
class Tanh(Layer):

    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.tanh()
    
# Слой с функцией активации sigmoid
class Sigmoid(Layer):

    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.sigmoid()


# Слой со среднеквадратичной ошибкой
class MSELoss(Layer):

    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return ((pred - target)*(pred - target)).sum(0)


# Последовательные слои
class Sequential(Layer):
    
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers
    
    # добавляем слой
    def add(self, layer):
        self.layers.append(layer)

    # Последовательно идем по слоям
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


# Другой вид ошибки
class CrossEntropyLoss(object):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return input.cross_entropy(target)


#### Test ####
"""np.random.seed(0)

data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)
target = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)

model = Sequential([Linear(2,3), Tanh(), Linear(3,1), Sigmoid()])
criterion = MSELoss()

optim = SGD(parameters=model.get_parameters(), alpha=1)

for i in range(10):
    
    # Predict
    pred = model.forward(data)
    
    # Compare
    loss = criterion.forward(pred, target)
    
    # Learn
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)
    """
