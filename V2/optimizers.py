
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
		for (param, param_grad) in zip(self.net.params(),
					self.net.param_grads()):

			param -= self.lr * param_grad
