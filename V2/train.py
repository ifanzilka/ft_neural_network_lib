from typing import Tuple
from copy import deepcopy
from network import *
from optimizers import *
from numpy import ndarray

class Trainer(object):
	'''
	Обучение нейросети.
	'''
	def __init__(self,
					net: NeuralNetwork,
					optim: Optimizer) -> None:
		'''
		Для обучения нужны нейросеть и оптимизатор. Нейросеть
		назначается атрибутом экземпляра оптимизатора.
		'''
		self.net = net
		self.optim = optim
		self.best_loss = 1e9
		# Добавил атрбиут
		setattr(self.optim, 'net', self.net)

	def generate_batches(self,
							X: ndarray,
							y: ndarray,
							size: int = 32) -> Tuple[ndarray]:
		'''
		Генерирует пакеты для обучения
		'''
		assert X.shape[0] == y.shape[0], \
		'''
		объекты и цель должны иметь одинаковое количество строк, вместо этого
		функции имеют {0}, а цель имеет {1}
		'''.format(X.shape[0], y.shape[0])

		N = X.shape[0]

		for ii in range(0, N, size):
			X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

			yield X_batch, y_batch


	def fit(self, X_train: ndarray, y_train: ndarray,
			X_test: ndarray, y_test: ndarray,
			epochs: int=100,
			eval_every: int=10,
			batch_size: int=32,
			seed: int = 1,
			restart: bool = True)-> None:
		'''
		Подгонка нейросети под обучающие данные за некоторое число
		эпох. Через каждые eval_every эпох выполняется оценка
		'''

		np.random.seed(seed)
		# Если в первый раз или перезапуск
		if restart:
			for layer in self.net.layers:
				layer.first = True

			self.best_loss = 1e9

		for e in range(epochs):

			if (e+1) % eval_every == 0:
				# for early stopping
				last_model = deepcopy(self.net)

			# Перемешиваем данные
			X_train, y_train = permute_data(X_train, y_train)

			# Выдаем массив данных размера батч сайз
			batch_generator = self.generate_batches(X_train, y_train, batch_size)

			for ii, (X_batch, y_batch) in enumerate(batch_generator):

				self.net.train_batch(X_batch, y_batch)
				self.optim.step()

			if (e+1) % eval_every == 0:

				test_preds = self.net.forward(X_test)
				loss = self.net.loss.forward(test_preds, y_test)

				if loss < self.best_loss:
					print(f"Validation loss after {e+1} epochs is {loss:.3f}")
					self.best_loss = loss
				else:
					print(f"""Loss increased after epoch {e+1}, final loss was {self.best_loss:.3f}, using the model from epoch {e+1-eval_every}""")
					self.net = last_model
					# ensure self.optim is still updating self.net
					setattr(self.optim, 'net', self.net)
					break

lr = NeuralNetwork(
    layers=[Dense(neurons=20,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

# Загрузим данные

from sklearn.datasets import load_boston

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names



# Scaling the data
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)



def to_2d_np(a: ndarray, 
          type: str="col") -> ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
    "Input tensors must be 1 dimensional"
    
    if type == "col":        
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

# make target 2d array
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)
print("ok")
# helper function

def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

# Первая модель

trainer = Trainer(lr, SGD(lr=0.01))


trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()	