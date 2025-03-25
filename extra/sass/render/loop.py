from tinygrad import Tensor
from tinygrad.helpers import DEBUG, Context, NOOPT

a = Tensor.ones(32).contiguous().realize()

with Context(DEBUG=4, NOOPT=1):
  a.sum(0).realize()


