from src.ev.EVMain import EV
from src.utils import TimeUtil
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_CACHE_PATH"] = "/home/minho/cudacache"
module = EV("train", sys.argv[1])
module.train()
TimeUtil.time_analysis()
