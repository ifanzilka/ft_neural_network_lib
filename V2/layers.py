from numpy import ndarray
from base import *
from typing import List
from activations import *
from dense import *

class Layer(object):
	'''
	Слой нейронов в нейросети.
	'''

	def __init__(self,
	 neurons: int):
		'''
		Число нейронов примерно соответствует «ширине» слоя
		'''
		self.neurons = neurons
		self.first = True
		self.params: List[ndarray] = []
		self.param_grads: List[ndarray] = []
		self.operations: List[Operation] = []

	def _setup_layer(self, num_in: int) -> None:
		'''
		Функция _setup_layer реализуется в каждом слое.
		'''
		raise NotImplementedError()

	def forward(self, input_: ndarray) -> ndarray:
		'''
		Передача входа вперед через серию операций.
		''' 

		# Если в первый раз то инициализируем
		if self.first:
			self._setup_layer(input_)
			self.first = False

		self.input_ = input_

		for operation in self.operations:

			input_ = operation.forward(input_)

		self.output = input_

		return self.output

	def backward(self, output_grad: ndarray) -> ndarray:
		'''
		Передача output_grad назад через серию операций.
		Проверка размерностей.
		'''

		assert_same_shape(self.output, output_grad)

		for operation in reversed(self.operations):
			output_grad = operation.backward(output_grad)

		input_grad = output_grad

		self._param_grads()

		return input_grad

	def _param_grads(self) -> ndarray:
		'''
		Извлечение _param_grads из операций слоя.
		'''

		self.param_grads = []
		for operation in self.operations:
			if issubclass(operation.__class__, ParamOperation):
				self.param_grads.append(operation.param_grad)

	def _params(self) -> ndarray:
		'''
		Извлечение _params из операций слоя.
		'''

		self.params = []
		for operation in self.operations:
			if issubclass(operation.__class__, ParamOperation):
				self.params.append(operation.param)


class Dense(Layer):
	'''
	Полносвязный слой, наследующий от Layer.
	'''
	def __init__(self,
				 neurons: int,
				 activation: Operation = Sigmoid()):
		'''
		 Для инициализации нужна функция активации.
		'''
		super().__init__(neurons)
		self.activation = activation

	def _setup_layer(self, input_: ndarray) -> None:
		'''
		Определение операций для полносвязного слоя.
		'''
		if self.seed:
			np.random.seed(self.seed)

		self.params = []


		# weights. Размера Матрицы весов [количесвто входов на кол-во нейронов]
		self.params.append(np.random.randn(input_.shape[1], self.neurons))

		# bias
		self.params.append(np.random.randn(1, self.neurons))

		self.operations = [WeightMultiply(self.params[0]),
				 		  BiasAdd(self.params[1]),
					   self.activation]

		return None