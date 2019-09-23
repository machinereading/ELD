from .Encoder import *
from .Transformer import *

encoder_map = {"CNN": CNNEncoder, "RNN": RNNEncoder, "BiRNN": BiContextEncoder, "SelfAttention": SelfAttentionEncoder, "Identity": Ident}
