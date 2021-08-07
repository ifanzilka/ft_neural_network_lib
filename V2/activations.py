import numpy as np
from numpy import ndarray
from base import Operation

class Linear(Operation):
	
	'''
	 Функция активации 
	'''

	def __init__(self) -> None:
		'''Pass'''
		super().__init__()

	def _output(self,inference: bool) -> ndarray:
		'''Pass through'''
		return self.input_

	def _input_grad(self, output_grad: ndarray) -> ndarray:
		'''Pass through'''
		return output_grad

class Sigmoid(Operation):
	'''
	Сигмоидная функция активации.
	'''

	def __init__(self) -> None:
		'''Pass'''
		super().__init__()

	def _output(self, inference: bool) -> ndarray:
		'''
		Вычисление выхода.
		'''
		return 1.0/(1.0+np.exp(-1.0 * self.input_))

	def _input_grad(self, output_grad: ndarray) -> ndarray:
		'''
		Вычисление входного градиента.
		'''
		""" Производная """
		sigmoid_backward = self.output * (1.0 - self.output)
		input_grad = sigmoid_backward * output_grad
		return input_grad

class Tanh(Operation):
	'''
 	Hyperbolic tangent activation function
	'''
	def __init__(self) -> None:
		super().__init__()

	def _output(self, inference: bool) -> ndarray:
		return np.tanh(self.input_)

	def _input_grad(self, output_grad: ndarray) -> ndarray:
		return output_grad * (1 - self.output * self.output)		

class ReLU(Operation):
	'''
	Hyperbolic tangent activation function
	'''
	def __init__(self) -> None:
		super().__init__()

	def _output(self, inference: bool) -> ndarray:
		return np.clip(self.input_, 0, None)

	def _input_grad(self, output_grad: ndarray) -> ndarray:
		mask = self.output >= 0
		return output_grad * mask
