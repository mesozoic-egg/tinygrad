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
  print_uops(kernel.uops)
  print(kernel.uops[-1])
  src = renderer.render("rendered", kernel.uops)
  return src

def render2(uops: List[UOp], renderer: Renderer):
  return renderer.render("rendererd", uops)

def compare2(uops: List[UOp]):
  src0 = render2(uops, PTXRenderer("sm_86"))
  print(src0)
  # src1 = render2(u, PTXRenderer2())
  # print("src1")
  # print(src1)
  # assert src0 == src1
  
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
  print(src0)
  src1 = render2(uops, ptx_renderer3)
  print(src1)
  # src2 = render2(uops, cuda_renderer)
  # print(src2)
  assert src0 == src1



def compare_ptx(a: Tensor):
  ast = schedule(a)
  print(ast)
  src0 = render(ast, PTXRenderer("sm_86"))
  print(src0)
  print(render(ast, cuda_renderer))
  # src1 = render(ast, CUDARenderer("sm_86"))
  # print(src1)
  # clang = render(ast, ClangRenderer())
  # print(clang)

    

def addition():
  a = Tensor.empty(1, 1)
  b = (a + 1).contiguous()
  compare_ptx(b)

def sum():
  a = Tensor.empty(4, 4).contiguous()
  b = a.sum(0)
  compare_ptx(b)

if __name__ == "__main__":
  addition()