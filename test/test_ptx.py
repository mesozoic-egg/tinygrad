import os
import pytest
from tinygrad import Tensor, Device, dtypes, Variable
from tinygrad.ops import Ops, UOp, sym_infer, sint, print_uops
from tinygrad.engine.realize import get_kernel
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer
from tinygrad.renderer.ptx2 import PTXRenderer as PTXRenderer2
from tinygrad.helpers import Context, NOOPT
from tinygrad.codegen.uopgraph import full_graph_rewrite
from tinygrad.codegen.linearize import linearize_uop

from typing import List

clang_renderer = ClangRenderer()
cuda_renderer = CUDARenderer("sm_86")
ptx_renderer = PTXRenderer("sm_86")
ptx_renderer2 = PTXRenderer2("sm_86")

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
  src1 = render2(uops, ptx_renderer2)
  assert src0 == src1

def compare_ptx2(a: UOp):
  uops = linearize_uop(full_graph_rewrite(a, ptx_renderer))  
  ptx_src = ptx_renderer.render("rendered", uops)
  ptx2_src = ptx_renderer2.render("rendered", uops)
  assert ptx_src == ptx2_src
  


def compare_ptx(a: Tensor):
  ast = schedule(a)
  src0 = render(ast, PTXRenderer("sm_86"))
  src1 = render(ast, ptx_renderer2)
  assert src0 == src1

def test_addition():
  a = Tensor.empty(4, 4)
  b = (a + 1).contiguous()
  compare_ptx(b)

def _sum():
  a = Tensor.empty(64, 64).contiguous()
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

def test_acc_vec():
  const_0 = UOp(Ops.CONST, dtypes.int, arg=0, src=())
  const_64 = UOp(Ops.CONST, dtypes.int, arg=64, src=())
  _range = UOp(Ops.RANGE, dtypes.int, arg=(1, True), src=(
    const_0,
    const_64
  ))
  const_0_0 = UOp(Ops.CONST, dtypes.float, arg=0.0, src=())
  vec = UOp(Ops.VECTORIZE, dtypes.float.vec(4), arg=None, src=(
    const_0_0,
    const_0_0,
    const_0_0,
    const_0_0,
  ))
     
  acc = UOp(Ops.DEFINE_ACC, dtypes.float.vec(4), arg=(0, ), src=(
    vec,
    _range 
  ))
  store([const_0, const_64, _range, const_0_0, vec, acc])

def test_var_in_special():
  vi = Variable("i", 1, 10).bind(9)
  a = Tensor.empty(vi, 8)
  b = Tensor.empty(8, 4)
  c = a.dot(b)
  compare_ptx(c)
    
def test_var_in_tensor():
  vi = Variable("i", 1, 10).bind(8)
  a = Tensor(vi) + 1
  compare_ptx(a)

def test_gated_store():
  a = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
  gate_alu = (lidx0:=UOp(Ops.SPECIAL, dtypes.int, (), ('lidx0', 4))).ne(0)
  gated_alu_store = UOp(Ops.STORE, dtypes.void, (a.index(lidx0, gate_alu), UOp.const(dtypes.int, 1)))
  sink = UOp(Ops.SINK, dtypes.void, (gated_alu_store,))
  compare_ptx2(sink)

def test_gated_store_if():
  a = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
  gate_alu = (lidx0:=UOp(Ops.SPECIAL, dtypes.int, (), ('lidx0', 4))).ne(0)
  val = UOp.const(dtypes.int, 1)
  if_uop = UOp(Ops.IF, dtypes.void, (gate_alu,))
  gated_alu_store = UOp(Ops.STORE, dtypes.void, (a.index(lidx0, if_uop), val))
  sink = UOp(Ops.SINK, dtypes.void, (gated_alu_store,))
  compare_ptx2(sink)