from abc import ABC
class AbstractArgument(ABC):
	@classmethod
	def from_json(cls, json_file):
		from . import jsonload
		args = cls()
		if type(json_file) is str:
			json_file = jsonload(json_file)
		for attr, value in json_file.items():
			setattr(args, attr, value)
		return args

	@classmethod
	def from_config(cls, ini_file):
		import configparser
		from .. import GlobalValues as gl
		c = configparser.ConfigParser()
		c.read(ini_file)
		args = cls()
		for attr, section in c.items():
			for k, v in section.items():
				if v in ["True", "False"]:
					v = gl.boolmap(v)
				setattr(args, k, v)
		return args


	def to_json(self):
		return self.__dict__