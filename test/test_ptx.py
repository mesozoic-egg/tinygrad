import os
from tinygrad import Tensor, Device, dtypes
from tinygrad.ops import Ops, UOp, Variable, sym_infer, sint, print_uops
from tinygrad.engine.realize import get_kernel
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer
from tinygrad.renderer.ptx2 import PTXRenderer as PTXRenderer2

def schedule(a: Tensor):
  scheduled, vars = a.schedule_with_vars() 
  for si in scheduled:
    if si.ast.op is Ops.SINK:
      return si.ast
  
def render(ast: UOp, renderer: Renderer):
  kernel = get_kernel(renderer, ast)
  kernel.linearize()
  print_uops(kernel.uops)
  src = renderer.render("rendered", kernel.uops)
  return src

def render2(uop: UOp, renderer: Renderer):
  return renderer.render("rendererd", [uop])

def compare2(u: UOp):
  src0 = render2(u, PTXRenderer("sm_86"))
  src1 = render2(u, PTXRenderer2())
  print("src1")
  print(src1)
  assert src0 == src1
  

def test_const():
  compare2(UOp(Ops.CONST, dtypes.uint, arg=2, src=()))



def compare_ptx(a: Tensor):
  ast = schedule(a)
  print(ast)
  # src0 = render(ast, PTXRenderer("sm_86"))
  # print(src0)
  # src1 = render(ast, CUDARenderer("sm_86"))
  # print(src1)
  # clang = render(ast, ClangRenderer())
  # print(clang)
  src2 = render(ast, PTXRenderer2())
  print(src2)

    

def addition():
  a = Tensor.empty(4, 4)
  b = (a + 1).contiguous()
  compare_ptx(b)

def sum():
  a = Tensor.empty(4, 4).contiguous()
  b = a.sum(0)
  compare_ptx(b)

if __name__ == "__main__":
  sum()