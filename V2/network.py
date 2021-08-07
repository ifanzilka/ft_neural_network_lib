# 1. NeuralNetwork будет в качестве атрибута получать список экземпляров Layer. Слои будут такими, как было определено ранее — с прямым и обратным методами. Эти методы принимают объекты ndarray
# и возвращают объекты ndarray.

# 2. Каждый Layer будет иметь список операций Operation, сохраненный
# в атрибуте operations слоя функцией _setup_layer.

# 3. Эти операции, как и сам слой, имеют методы прямого и обратного
# преобразования, которые принимают в качестве аргументов объекты
# ndarray и возвращают объекты ndarray в качестве выходных данных.

# 4. В каждой операции форма output_grad, полученная вметоде backward,
# должна совпадать с формой выходного атрибута Layer. То же самое
# верно для форм input_grad, передаваемых в обратном направлении
# методом backward и атрибутом input_.

# 5. Некоторые операции имеют параметры (которые хранятся в атрибуте param). Эти операции наследуют от класса ParamOperation. Те же
# самые ограничения на входные и выходные формы применяются
# к слоям и их методам forward и backward — они берут объекты ndarray,
# и формы входных и выходных атрибутов и их соответствующие
# градиенты должны совпадать.

# 6. У класса NeuralNetwork также будет класс Loss. Этот класс берет
# выходные данные последней операции из NeuralNetwork и цели,
# проверяет, что их формы одинаковы, и, вычисляя значение потерь
# (число) и ndarray loss_grad, которые будут переданы в выходной
# слой, начинает обратное распространение.

from numpy import ndarray
from typing import List
from layers import*
from losses import *

class NeuralNetwork(object):
	'''
	Класс нейронной сети.
	'''
	def __init__(self, 
					layers: List[Layer],
					loss: Loss,
					seed: int = 1) -> None:
		'''
		Нейросети нужны слои и потери.
		'''
		self.layers = layers
		self.loss = loss
		self.seed = seed
		if seed:
			for layer in self.layers:
				setattr(layer, "seed", self.seed)        

	def forward(self, x_batch: ndarray) -> ndarray:
		'''
		Передача данных через последовательность слоев.
		'''
		x_out = x_batch
		for layer in self.layers:
			x_out = layer.forward(x_out)

		return x_out

	def backward(self, loss_grad: ndarray) -> None:
		'''
		Передача данных назад через последовательность слоев.
		'''

		grad = loss_grad
		for layer in reversed(self.layers):
			grad = layer.backward(grad)

		return None

	def train_batch(self,
					x_batch: ndarray,
					y_batch: ndarray) -> float:
		'''
		Передача данных вперед через последовательность слоев.
		Вычисление потерь.
		Передача данных назад через последовательность слоев.
		'''

		predictions = self.forward(x_batch)

		loss = self.loss.forward(predictions, y_batch)

		self.backward(self.loss.backward())

		return loss

	def params(self):
		'''
		Получение параметров нейросети.
		'''
		for layer in self.layers:
			yield from layer.params

	def param_grads(self):
		'''
		Получение градиента потерь по отношению к параметрам нейросети.
		'''
		for layer in self.layers:
			yield from layer.param_grads