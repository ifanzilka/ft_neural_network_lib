import numpy as np

class Optimizer(object):
	'''
	Базовый класс оптимизатора нейросети.
	lr -> скорость оубчения
	FINAL_LR -> конечная скорость обучения до которой будем сниажться
	'''
	def __init__(self,
				lr: float = 0.01,
				final_lr: float = 0,
				decay_type: str = 'exponential') -> None:
		self.lr = lr
		self.final_lr = final_lr
		self.decay_type = decay_type
		self.first = True

	'''
	Как будет lr во время обучения (линейное и экспоненциальное затухание)
	'''
	def _setup_decay(self) -> None:

		if not self.decay_type:
			return
		elif self.decay_type == 'exponential':
			self.decay_per_epoch = np.power(self.final_lr / self.lr,
								1.0 / (self.max_epochs - 1))
		elif self.decay_type == 'linear':
			self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)
	
	def _decay_lr(self) -> None:

		if not self.decay_type:
			return

		if self.decay_type == 'exponential':
			self.lr *= self.decay_per_epoch

		elif self.decay_type == 'linear':
			self.lr -= self.decay_per_epoch
		

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
					lr: float = 0.01,
					final_lr:float = 0,
					decay_type:str = None) -> None:
		'''Pass'''
		super().__init__(lr, final_lr, decay_type)

	def step(self):
		'''
		Для каждого параметра настраивается направление, при этом
		амплитуда регулировки зависит от скорости обучения.
		'''
		for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):

			param -= self.lr * param_grad

	def _update_rule(self, **kwargs) -> None:

		update = self.lr*kwargs['grad']
		kwargs['param'] -= update		





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