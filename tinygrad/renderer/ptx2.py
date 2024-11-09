from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.helpers import strip_parens, getenv, prod, dedup, AMX
from tinygrad.renderer.ptx import ptx_matcher, render_val
from tinygrad.dtype import PtrDType, dtypes, DType
from tinygrad.ops import GroupOp, UnaryOps, BinaryOps, TernaryOps, Ops, UOp, PatternMatcher, UPat, cast_float_to_bf16
from typing import Dict

def render_const(ctx, x):
  val = render_val(x, dtypes.uint32)
  return f"mov.b32"

def render_store(ctx, bidx, var):
  return f"st.global.u32 {ctx[bidx]}, {ctx[var]};"

base = PatternMatcher([
  (UPat(Ops.CONST, dtype=dtypes.uint32, name="x"), render_const),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"{ctx.code_for_workitem[x.arg[0][0]](x.arg[0][-1])}; /* {x.arg[1]} */"),
  (UPat(GroupOp.ALU, name="x"), lambda ctx,x: ctx.code_for_op[x.op](
    *([strip_parens(ctx[v]) if v.op == x.op and x.op in {BinaryOps.ADD, BinaryOps.MUL, BinaryOps.XOR} else ctx[v] for v in x.src]), x.dtype)),
   (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True), render_store),
])

class PTXRenderer(CStyleLanguage):
  kernel_prefix = """.version VERSION
.target TARGET
.address_size 64
.visible .entry"""

  extra_matcher = ptx_matcher
  string_rewrite = base
  code_for_workitem = {"g": lambda x: f"blockIdx.{chr(120+int(x))}", "l": lambda x: f"threadIdx.{chr(120+int(x))}",
                      "i": lambda x: f"(blockIdx.{chr(120+int(x))}*blockDim.{chr(120+int(x))}+threadIdx.{chr(120+int(x))})"}
  types: Dict[DType, str] = { dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
                                dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
                                dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "pred" }

  def render_kernel(self, function_name, kernel, bufs, regs) -> str:
    print("bufs")
    print(bufs)
    def fmt(line): return line if line[0]=="$" else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1)

    kernel = [f".reg .{reg};" for reg in regs] + kernel + ["ret;"]
    ret = (f"{self.kernel_prefix} {function_name}(\n\t" + 
      ',\n\t'.join([f".param {name}" for name,(dtype,mutable) in bufs]) +
      "\n)\n{\n" +
      '\n'.join([fmt(line) for op in kernel for line in op.splitlines()]) +
      "\n}"
    )
    return ret
