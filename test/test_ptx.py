import os
from tinygrad import Tensor, Device
from tinygrad.ops import Ops, UOp, Variable, sym_infer, sint
from tinygrad.engine.realize import get_kernel
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer

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

def compare_ptx(a: Tensor):
  ast = schedule(a)
  print(ast)
  src0 = render(ast, PTXRenderer("sm_86"))
  print(src0)
  src1 = render(ast, CUDARenderer("sm_86"))
  print(src1)
  clang = render(ast, ClangRenderer())
  print(clang)

    

def addition():
  a = Tensor.empty(4, 4)
  b = (a + 1).contiguous()
  compare_ptx(b)

addition()