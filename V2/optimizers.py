import numpy as np

class Optimizer(object):
	'''
	Базовый класс оптимизатора нейросети.
	'''
	def __init__(self,
				lr: float = 0.01):
		'''
		У оптимизатора должна быть начальная скорость обучения.
		'''
		self.lr = lr

	def step(self) -> None:
		'''
		У оптимизатора должна быть функция "step".
		'''
		pass
	
class SGD(Optimizer):
	'''
	Стохастический градиентный оптимизатор.
	'''    
	def __init__(self,
					lr: float = 0.01) -> None:
		'''Pass'''
		super().__init__(lr)

	def step(self):
		'''
		Для каждого параметра настраивается направление, при этом
		амплитуда регулировки зависит от скорости обучения.
		'''
		for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):

			param -= self.lr * param_grad

class SGDMomentum(Optimizer):
	"""
	
	"""
	def __init__(self,
				lr: float = 0.01,
				final_lr: float = 0,
				decay_type: str = None,
				momentum: float = 0.9) -> None:
		super().__init__(lr, final_lr, decay_type)
		self.momentum = momentum

	def step(self) -> None:
		'''
		если итерация первая: инициализируем «скорости» для всех параметров.
 		иначе вызываем функцию _update_rule.
		'''
		if self.first:
			self.velocities = [np.zeros_like(param)
								for param in self.net.params()]
			self.first = False

		for (param, param_grad, velocity) in zip(self.net.params(),
												self.net.param_grads(),
												self.velocities):
			self._update_rule(param=param,
				grad=param_grad,
				velocity=velocity)
		'''
		grad_t_n = g_n + u * (g_(n-1) ) + u^2 * (g_(n-2)) + ... u^n * (g_(n - n)) 
		'''
	def _update_rule(self, **kwargs) -> None:

			# Update velocity
			kwargs['velocity'] *= self.momentum
			kwargs['velocity'] += self.lr * kwargs['grad']

			# Use this to update parameters
			kwargs['param'] -= kwargs['velocity']			
