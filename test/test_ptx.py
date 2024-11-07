import os
os.environ["PTX"] = "1"
from tinygrad import Tensor, Device
from tinygrad.ops import Ops, UOp, Variable, sym_infer, sint
from tinygrad.engine.realize import get_kernel
from tinygrad.renderer.ptx import PTXRenderer

def schedule(a: Tensor):
  scheduled, vars = a.schedule_with_vars() 
  for si in scheduled:
    if si.ast.op is Ops.SINK:
      return si.ast
  
def render(ast: UOp, PTX_CSTYLE: bool=False):
  kernel = get_kernel(Device["CUDA"].renderer, ast)
  kernel.linearize()
  renderer = PTXRenderer("sm_86") if PTX_CSTYLE else PTXRenderer("sm_86")
  src = renderer.render("rendered", kernel.uops)
  return src

def compare_ptx(a: Tensor):
  ast = schedule(a)
  src0 = render(ast)
  print(src0)
  src1 = render(ast, PTX_CSTYLE=True)
  if src0 != src0:
    print("src0")
    print(src0)
    print("src1")
    print(src1)
  assert src0 == src1

    

def addition():
  a = Tensor.empty(4, 4)
  b = (a + 1).contiguous()
  compare_ptx(b)

addition()