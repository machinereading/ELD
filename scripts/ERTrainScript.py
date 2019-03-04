from src.er.ERMain import ER
from src.utils import TimeUtil
from src.utils import readfile, writefile


er = ER("ERTest")

er.train("corpus/er/wiki_fix.conll", "corpus/er/cw_fix.conll")