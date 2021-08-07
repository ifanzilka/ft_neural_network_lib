import numpy as np

from numpy import ndarray
from utils import *

class Loss(object):
	'''
	Потери нейросети.
	'''

	def __init__(self):
		'''Pass'''
		pass

	def forward(self, prediction: ndarray, target: ndarray) -> float:
		'''
		Вычисление значения потерь.
		'''
		assert_same_shape(prediction, target)

		self.prediction = prediction
		self.target = target
		
		loss_value = self._output()

		return loss_value

	def backward(self) -> ndarray:
		'''
		Вычисление градиента потерь по входам функции потерь.
		'''
		self.input_grad = self._input_grad()

		assert_same_shape(self.prediction, self.input_grad)

		return self.input_grad

	def _output(self) -> float:
		'''
		Функция _output должна реализовываться всем подклассами
		класса Loss.
		'''
		raise NotImplementedError()

	def _input_grad(self) -> ndarray:
		'''
		Функция _input_grad должна реализовываться всем подклассами
		класса Loss
		'''
		raise NotImplementedError()

class MeanSquaredError(Loss):

	def __init__(self,
				 normalize: bool = False) -> None:
		super().__init__()
		self.normalize = normalize

	def _output(self) -> float:

		'''
		вычисление среднего квадрата ошибки.
		L = (X - Y)^2
		'''
		if self.normalize:
			self.prediction = self.prediction / self.prediction.sum(axis=1, keepdims=True)

		loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

		return loss

	def _input_grad(self) -> ndarray:
		'''
		Вычисление градиента ошибки по входу MSE.
		dL = 2 * (X - Y)
		'''
		return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]

class SoftmaxCrossEntropy(Loss):
	def __init__(self, eps: float=1e-9) -> None:
		super().__init__()
		self.eps = eps
		self.single_class = False

	def _output(self) -> float:

		# если сеть просто выводит вероятности
		# только из-за принадлежности к одному классу:
		if self.target.shape[1] == 0:
			self.single_class = True

		# если "один класс", примените операцию "нормализовать", определенную 
		if self.single_class:
			self.prediction, self.target = \
				normalize(self.prediction), normalize(self.target)

		# применение функции softmax к каждой строке (наблюдение)
		softmax_preds = softmax(self.prediction, axis=1)

		
		# обрезка вывода softmax для предотвращения числовой нестабильности
		self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

		# actual loss computation
		softmax_cross_entropy_loss = (
			-1.0 * self.target * np.log(self.softmax_preds) - \
				(1.0 - self.target) * np.log(1 - self.softmax_preds)
		)

		return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

	def _input_grad(self) -> ndarray:

		# если "один класс", "ненормализовать" вероятности перед возвращением градиента:
		if self.single_class:
			return unnormalize(self.softmax_preds - self.target)
		else:
			return (self.softmax_preds - self.target) / self.prediction.shape[0]