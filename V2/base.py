#python3.6
import numpy as np
import time
from numpy.lib import utils
from utils import *
from numpy import ndarray # Многомерный массив
from typing import List

class Operation(object):
	'''
	Базовый класс операции в нейросети.
	'''
	def __init__(self):
		pass
	
	def forward(self, input_: ndarray, inference: bool=False) -> ndarray:
		'''
		Сохраняет входные данные в переменной экземпляра self._input
		Вызывает функцию self._output ().
		inference -> dropout
		'''
		self.input_ = input_
		self.output = self._output(inference)
		return self.output
	
	def backward(self, output_grad: ndarray) -> ndarray:
		
		'''
		Вызов функции self._input_grad().
		Проверка совпадения размерностей.
		'''
		
		assert_same_shape(self.output, output_grad)
		
		self.input_grad = self._input_grad(output_grad)
		
		assert_same_shape(self.input_, self.input_grad)
		
		return self.input_grad
		
	def _output(self, inference: bool) -> ndarray:
		
		'''
		Метод _output определяется для каждой операции.
		'''
		raise NotImplementedError()

	def _input_grad(self, output_grad: ndarray) -> ndarray:
		'''
		Метод _input_grad определяется для каждой операции.
		'''
		raise NotImplementedError()

"""
Подобно базовому классу Operation, отдельная операция ParamOperation
должна определять функцию _param_grad в дополнение к функциям
_output и _input_grad.
"""

class ParamOperation(Operation):
	'''
	Операция с параметрами.
	'''

	def __init__(self, param: ndarray) -> ndarray:
		'''
		Метод ParamOperation
		'''
		super().__init__()
		self.param = param

	def backward(self, output_grad: ndarray) -> ndarray:
		'''
		Вызов self._input_grad и self._param_grad.
		Проверка размерностей.
		'''

		assert_same_shape(self.output, output_grad)

		self.input_grad = self._input_grad(output_grad)
		self.param_grad = self._param_grad(output_grad)

		assert_same_shape(self.input_, self.input_grad)
		assert_same_shape(self.param, self.param_grad)

		return self.input_grad

	def _param_grad(self, output_grad: ndarray) -> ndarray:
		'''
		Во всех подклассах ParamOperation должна быть реализация
		метода _param_grad.
		'''
		raise NotImplementedError()