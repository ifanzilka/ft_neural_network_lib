import torch.nn as nn
from torch import Tensor


# Базовый класс для всех модулей нейронной сети.
# Ваши модели также должны относиться к подклассу этого класса.
# Модули также могут содержать другие модули, что позволяет размещать их в древовидной структуре. Вы можете назначить подмодули в качестве обычных атрибутов:
# Настройки модели для оубчения или обычного режима

def inference_mode(m: nn.Module):
	m.eval()


class PyTorchLayer(nn.Module):

	def __init__(self) -> None:
		super().__init__()

	def forward(self, x: Tensor,
		inference: bool = False) -> Tensor:
		raise NotImplementedError()


class DenseLayer(PyTorchLayer):

	'''
	'''
	def __init__(self,
				 input_size: int,
				 neurons: int,
				 dropout: float = 1.0,
				 activation: nn.Module = None) -> None:

		super().__init__()
		self.linear = nn.Linear(input_size, neurons)
		self.activation = activation
		if dropout < 1.0:
			self.dropout = nn.Dropout(1 - dropout)

	def forward(self, x: Tensor,
		inference: bool = False) -> Tensor:
		if inference:
			self.apply(inference_mode)

		x = self.linear(x)  # does weight multiplication + bias 
		if self.activation:
			x = self.activation(x)
		if hasattr(self, "dropout"):
			x = self.dropout(x)

		return x