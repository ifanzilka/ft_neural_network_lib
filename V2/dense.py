import numpy as np
from numpy import ndarray
from base import ParamOperation

"""
Матричное умножение X * W
"""
class WeightMultiply(ParamOperation):
	'''
	Умножение весов в нейронной сети.
	'''

	def __init__(self, W: ndarray):
		'''
		Инициализация класса Operation с self.param = W.
		'''
		super().__init__(W)

	def _output(self,inference: bool) -> ndarray:
		'''
		Вычисление выхода.
		'''
		return np.dot(self.input_, self.param)

	def _input_grad(self, output_grad: ndarray) -> ndarray:
		'''
		Вычисление выходного градиента.

		f(x) = X * W
		df/dx = W.T
		'''
		return np.dot(output_grad, np.transpose(self.param, (1, 0)))

	def _param_grad(self, output_grad: ndarray)  -> ndarray:
		'''
		Вычисление градиента параметров.
		'''
		return np.dot(np.transpose(self.input_, (1, 0)), output_grad)

class BiasAdd(ParamOperation):
	'''
	Прибавление отклонений.
	'''

	def __init__(self, B: ndarray):
		'''
		Инициализация класса Operation с self.param = B.
		Проверка размерностей.
		'''
		assert B.shape[0] == 1

		super().__init__(B)

	def _output(self,inference: bool) -> ndarray:
		'''
		Вычисление выхода.
		'''
		return self.input_ + self.param

	def _input_grad(self, output_grad: ndarray) -> ndarray:
		'''
		Вычисление входного градиента.
		'''
		return np.ones_like(self.input_) * output_grad

	def _param_grad(self, output_grad: ndarray) -> ndarray:
		'''
		Вычисление градиента параметров.
		'''
		param_grad = np.ones_like(self.param) * output_grad
		return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])