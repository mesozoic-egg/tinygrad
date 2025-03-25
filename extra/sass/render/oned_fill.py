from tinygrad import Tensor
from tinygrad.helpers import DEBUG, Context, NOOPT

with Context(DEBUG=4, NOOPT=1):
  a = Tensor.ones(32).contiguous().realize()
  print(a.tolist())