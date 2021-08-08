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
		# У каждой операции свой у умножения np dot
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
		# Пришло 32 на 1 делаем обратную операцию
		# Сделали обратную операцию умножению (точнее производная от матричного умножения)
		# Это мы передали след слою
		self.input_grad = self._input_grad(output_grad)
		#Это как мы будем менять веса
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

class Flatten(Operation):
    def __init__(self):
        super().__init__()

    # делаем вектор из тензора
    def _output(self, inference: bool = False) -> ndarray:
        return self.input_.reshape(self.input_.shape[0], -1)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad.reshape(self.input_.shape)

class Conv2D_Op(ParamOperation):

    def __init__(self, W: ndarray):
        super().__init__(W)
        self.param_size = W.shape[2]
        self.param_pad = self.param_size // 2

    def _pad_1d(self, inp: ndarray) -> ndarray:
        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, inp, z])

    def _pad_1d_batch(self,
                      inp: ndarray) -> ndarray:
        outs = [self._pad_1d(obs) for obs in inp]
        return np.stack(outs)

    def _pad_2d_obs(self,
                    inp: ndarray):
        '''
        Input is a 2 dimensional, square, 2D Tensor
        '''
        inp_pad = self._pad_1d_batch(inp)

        other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2))

        return np.concatenate([other, inp_pad, other])


    # def _pad_2d(self,
    #             inp: ndarray):
    #     '''
    #     Input is a 3 dimensional tensor, first dimension batch size
    #     '''
    #     outs = [self._pad_2d_obs(obs, self.param_pad) for obs in inp]
    #
    #     return np.stack(outs)

    def _pad_2d_channel(self,
                        inp: ndarray):
        '''
        inp has dimension [num_channels, image_width, image_height]
        '''
        return np.stack([self._pad_2d_obs(channel) for channel in inp])

    def _get_image_patches(self,
                           input_: ndarray):
        imgs_batch_pad = np.stack([self._pad_2d_channel(obs) for obs in input_])
        patches = []
        img_height = imgs_batch_pad.shape[2]
        for h in range(img_height-self.param_size+1):
            for w in range(img_height-self.param_size+1):
                patch = imgs_batch_pad[:, :, h:h+self.param_size, w:w+self.param_size]
                patches.append(patch)
        return np.stack(patches)

    def _output(self,
                inference: bool = False):
        '''
        conv_in: [batch_size, channels, img_width, img_height]
        param: [in_channels, out_channels, fil_width, fil_height]
        '''
    #     assert_dim(obs, 4)
    #     assert_dim(param, 4)
        batch_size = self.input_.shape[0]
        img_height = self.input_.shape[2]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        patch_size = self.param.shape[0] * self.param.shape[2] * self.param.shape[3]

        patches = self._get_image_patches(self.input_)

        patches_reshaped = (patches
                            .transpose(1, 0, 2, 3, 4)
                            .reshape(batch_size, img_size, -1))

        param_reshaped = (self.param
                          .transpose(0, 2, 3, 1)
                          .reshape(patch_size, -1))

        output_reshaped = (
            np.matmul(patches_reshaped, param_reshaped)
            .reshape(batch_size, img_height, img_height, -1)
            .transpose(0, 3, 1, 2))

        return output_reshaped


    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        img_height = self.input_.shape[2]

        output_patches = (self._get_image_patches(output_grad)
                          .transpose(1, 0, 2, 3, 4)
                          .reshape(batch_size * img_size, -1))

        param_reshaped = (self.param
                          .reshape(self.param.shape[0], -1)
                          .transpose(1, 0))

        return (
            np.matmul(output_patches, param_reshaped)
            .reshape(batch_size, img_height, img_height, self.param.shape[0])
            .transpose(0, 3, 1, 2)
        )


    def _param_grad(self, output_grad: ndarray) -> ndarray:

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]

        in_patches_reshape = (
            self._get_image_patches(self.input_)
            .reshape(batch_size * img_size, -1)
            .transpose(1, 0)
            )

        out_grad_reshape = (output_grad
                            .transpose(0, 2, 3, 1)
                            .reshape(batch_size * img_size, -1))

        return (np.matmul(in_patches_reshape,
                          out_grad_reshape)
                .reshape(in_channels, self.param_size, self.param_size, out_channels)
                .transpose(0, 3, 1, 2))