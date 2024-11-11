import os
import pytest
from tinygrad import Tensor, Device, dtypes
from tinygrad.ops import Ops, UOp, Variable, sym_infer, sint, print_uops
from tinygrad.engine.realize import get_kernel
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer
from tinygrad.renderer.ptx2 import PTXRenderer as PTXRenderer2
from tinygrad.renderer.ptx3 import PTXRenderer as PTXRenderer3
from tinygrad.helpers import Context, NOOPT

from typing import List

clang_renderer = ClangRenderer()
cuda_renderer = CUDARenderer("sm_86")
ptx_renderer = PTXRenderer("sm_86")
ptx_renderer2 = PTXRenderer2()
ptx_renderer3 = PTXRenderer3("sm_86")

def schedule(a: Tensor):
  scheduled, vars = a.schedule_with_vars() 
  for si in scheduled:
    if si.ast.op is Ops.SINK:
      return si.ast
  
def render(ast: UOp, renderer: Renderer):
  kernel = get_kernel(renderer, ast)
  kernel.linearize()
  src = renderer.render("rendered", kernel.uops)
  return src

def render2(uops: List[UOp], renderer: Renderer):
  return renderer.render("rendererd", uops)

def test_const():
  store()

def test_mul_float():
  const_1 = UOp(Ops.CONST, dtypes.float64, arg=2.0)
  const_2 = UOp(Ops.CONST, dtypes.float64, arg=3.0)
  mul = UOp(Ops.MUL, dtypes.long, arg=None, src=(const_1, const_2))
  store([const_1, const_2, mul])

def test_unary():
  const_1 = UOp(Ops.CONST, dtypes.float64, arg=2.0)
  exped = UOp(Ops.EXP2, dtypes.float64, src=(const_1,))
  store([const_1, exped])

def test_cast():
  const_1 = UOp(Ops.CONST, dtypes.float32, arg=2.0)
  casted = UOp(Ops.CAST, dtypes.float64, src=(const_1,))
  store([const_1, casted])

def test_load():
  buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1)
  casted = UOp(Ops.CAST, dtypes.long, src=(buf,))
  load = UOp(Ops.LOAD, dtypes.float32, src=(casted,))
  store([buf, casted, load])
  


def store(uops: List[UOp]=[UOp(Ops.CONST, dtypes.uint, arg=2)]):
  define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
  special = UOp(Ops.SPECIAL, dtypes.int, arg=('gidx0', 16), src=())
  added = UOp(Ops.ADD, dtypes.long, arg=None, src=(define_global, special))
  store = UOp(Ops.STORE, dtypes.void, arg=None, src=(added, uops[-1]))
  uops = [define_global, special, added] + uops + [store]
  src0 = render2(uops, ptx_renderer)
  src1 = render2(uops, ptx_renderer3)
  assert src0 == src1



def compare_ptx(a: Tensor):
  ast = schedule(a)
  src0 = render(ast, PTXRenderer("sm_86"))
  src1 = render(ast, ptx_renderer3)
  assert src0 == src1

def test_addition():
  a = Tensor.empty(4, 4)
  b = (a + 1).contiguous()
  compare_ptx(b)

def _sum():
  a = Tensor.empty(16, 16).contiguous()
  b = a.sum(0)
  compare_ptx(b)

def test_sum():
  _sum()

def test_sum_loop():
  with Context(NOOPT=1):
    _sum()

def test_arange():
  a = Tensor.arange(0, 12)
  compare_ptx(a)

def test_matmul():
  a = Tensor.empty(16, 16)
  b = Tensor.empty(16, 16)
  c = a.dot(b)
  compare_ptx(c)

def test_wmma():
  a = Tensor.empty(16, 16, dtype=dtypes.half)
  b = Tensor.empty(16, 8, dtype=dtypes.half)
  r = a.matmul(b, acc_dtype=dtypes.float)
  compare_ptx(r)
