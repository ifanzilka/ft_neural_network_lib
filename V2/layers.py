from numpy import ndarray
from base import *
from typing import List
from activations import *
from dense import *
from dropout import *

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
		#заполняем когда инициализируем веса 
		self.params: List[ndarray] = []
		# градиент по параметрам. заполняю во время обратного прохода
		self.param_grads: List[ndarray] = []
		self.operations: List[Operation] = []

	def _setup_layer(self, num_in: int) -> None:
		'''
		Функция _setup_layer реализуется в каждом слое.
		'''
		raise NotImplementedError()

	def forward(self, input_: ndarray, inference = False) -> ndarray:
		'''
		Передача входа вперед через серию операций.
		''' 

		# Если в первый раз то инициализируем (пришло (batch_size, len))
		if self.first:
			self._setup_layer(input_)
			self.first = False

		self.input_ = input_

		# проходим по всем операциям в слое
		for operation in self.operations:

			input_ = operation.forward(input_,inference)

		self.output = input_

		return self.output

	def backward(self, output_grad: ndarray) -> ndarray:
		'''
		Передача output_grad назад через серию операций.
		Проверка размерностей.
		'''

		assert_same_shape(self.output, output_grad)
		#  обратный проход по операциям
		for operation in reversed(self.operations):
			output_grad = operation.backward(output_grad)

		input_grad = output_grad

		# список обратных операций	
		self._param_grads()

		return input_grad

	def _param_grads(self) -> ndarray:
		
		'''
		Извлечение _param_grads из операций слоя.
		'''

		self.param_grads = []
		for operation in self.operations:
			if issubclass(operation.__class__, ParamOperation):
				# add gradinet to operation
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
	operations хранит 
	weight_init -> настройка весов чтобы при мзенение всех признаков ответ не менялся уменьшаме дисперсию и чтобы масштаб признков не заичел от количества
	
	'''
	def __init__(self,
				neurons: int,
				activation: Operation = Linear(),
				conv_in: bool = False,
				dropout: float = 1.0,
				weight_init: str = "standard") -> None:
		super().__init__(neurons)
		self.activation = activation
		self.conv_in = conv_in
		self.dropout = dropout
		self.weight_init = weight_init
	
	def _setup_layer(self, input_: ndarray) -> None:
		'''
		Определение операций для полносвязного слоя.
		'''
		if self.seed:
			np.random.seed(self.seed)

		# кол во входов 784 (грубо говоря чтобы отклонение по всему слоя в начале было 1)
		num_in = input_.shape[1]
		if self.weight_init == "glorot":
			scale = 2/(num_in + self.neurons)
		else:
			scale = 1.0
		
		#print(input_.shape)
		#print(scale)
		self.params = []

		# weights. Размера Матрицы весов [количесвто входов на кол-во нейронов]
		#self.params.append(np.random.randn(input_.shape[1], self.neurons))

		# bias
		#self.params.append(np.random.randn(1, self.neurons))

		#Стандартное отклонение (разброс или “ширина”) распределения. Должно быть неотрицательным.
		self.params.append(np.random.normal(loc=0,
											scale=scale,
											size=(num_in, self.neurons)))

		# bias
		self.params.append(np.random.normal(loc=0,
											scale=scale,
											size=(1, self.neurons)))


		# в этом слое из операций добавим умножение и прибавление смещения
		self.operations = [WeightMultiply(self.params[0]),
				 		  BiasAdd(self.params[1]),
					   self.activation]
		if self.dropout < 1.0:
			self.operations.append(Dropout(self.dropout))
		return None

# Читатели, знакомые с понятием вычислительной сложности, могут
# сказать, что такой код катастрофически медленный: для вычисления
# градиента параметра пришлось написать семь вложенных циклов! В этом
# нет ничего плохого, поскольку нам нужно было прочувствовать и понять
# принцип работы CNN, написав все с нуля. Но можно написать по-другому:
# 1) из входных данных извлекаются участки image_height × image_
# width × num_channels размера filter_height × filter_width из набора
# тестов;
# 2) для каждого участка выполняется скалярное произведение на соответствующий фильтр, соединяющий входные каналы с выходными
# каналами;
# 3) складываем результаты скалярных произведений, чтобы сформировать результат.
class Conv2D(Layer):
    '''
    Как только мы определим все операции и контур слоя,
    все, что остается реализовать здесь, - это функция _setup_layer!
    '''
    def __init__(self,
                 out_channels: int,
                 param_size: int,
                 dropout: int = 1.0,
                 weight_init: str = "normal",
                 activation: Operation = Linear(),
                 flatten: bool = False) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.activation = activation
        self.flatten = flatten
        self.dropout = dropout
        self.weight_init = weight_init
        self.out_channels = out_channels
  
    # Инициализация
    def _setup_layer(self, input_: ndarray) -> ndarray:

        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2/(in_channels + self.out_channels)
        else:
            scale = 1.0

        conv_param = np.random.normal(loc=0,
                                      scale=scale,
                                      size=(input_.shape[1],  # input channels
                                     self.out_channels,
                                     self.param_size,
                                     self.param_size))

        self.params.append(conv_param)

        self.operations = []
        self.operations.append(Conv2D_Op(conv_param))
        self.operations.append(self.activation)

        """
        В зависимости от того, хотим ли мы передавать выходные данные этого
        слоя в другой сверточный слой или в полносвязный связанный слой для
        предсказаний, применятся (или нет) операция flatten.
        """
        if self.flatten:
            self.operations.append(Flatten())

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None		