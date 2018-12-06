from abc import ABC, abstractmethod, abstractproperty


# 각종 el module을 돌리기 위해 input form과 output form을 맞춰주는 모듈
class AbstractWrapper(ABC):
	def __init__(self):
		pass

	@abstractmethod
	def run(self, **kwargs):
		pass

	@abstractmethod
	def input_wrap(self, data):
		pass

	@abstractmethod
	def output_wrap(self, output):
		pass


