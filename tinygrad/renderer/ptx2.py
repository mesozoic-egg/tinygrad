from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.ptx import ptx_matcher, render_val
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.ops import GroupOp, UnaryOps, BinaryOps, TernaryOps, Ops, UOp, PatternMatcher, UPat, cast_float_to_bf16

def render_const(ctx, x):
  val = render_val(x, dtypes.uint32)
  return f"mov.b32"

base = PatternMatcher([
  (UPat(Ops.CONST, dtype=dtypes.uint32, name="x"), render_const),
])

class PTXRenderer(CStyleLanguage):
  kernel_prefix = """.version VERSION
.target TARGET
.address_size 64
.visible .entry"""

  extra_matcher = ptx_matcher
  string_rewrite = base
  def render_kernel(self, function_name, kernel, bufs, regs) -> str:
    print(kernel)
    kernel = [f".reg .{reg};" for reg in regs] + kernel + ["ret;"]
    def fmt(line): return line if line[0]=="$" else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1)
    return (f"{self.kernel_prefix} {function_name}(\n\t" +
            ',\n\t'.join([f".param .{'u64' if dtype.__class__ == PtrDType else self.types[dtype]} {name}" for name,dtype in bufs]) + "\n)\n{\n" +
            '\n'.join([fmt(line) for op in kernel for line in op.splitlines()]) +
            "\n}")