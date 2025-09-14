from typing import cast, Callable, Union, Mapping
import struct, dataclasses
from collections import defaultdict
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import flatten, get_single_element, prod, DEBUG

def oneline_uop(u: UOp): return repr(u).split('\n')[0]

rewriter = PatternMatcher([
  (UPat(Ops.DEFINE_GLOBAL, name='x'), lambda x: [f""]),
  (UPat(Ops.SPECIAL, name='x'), lambda x: [
    f"[B------:R-:W-:-:S01]   S2R R6, SR_TID.X ;"
  ]),
  (UPat(Ops.CONST, name='x'), lambda x: [f""]),
  (UPat(Ops.INDEX, name='x'), lambda x: [
    f"[B------:R-:W-:-:S01] IMAD.WIDE R10, R6, R7, c[0x0][0x160];"
  ]),
  (UPat(Ops.LOAD, name='x'), lambda x: [
    f"[B------:R-:W0:-:S05] LDG.E R2, desc[UR4][R2.64];"
  ]),
  (UPat(Ops.MUL, name='x'), lambda x: [
    f"[B0-----:R-:W-:-:S04] FMUL R9, R2, R5;"
  ]),
  (UPat(Ops.STORE, name='x'), lambda x: [
    f"[B------:R-:W-:-:S01] STG.E desc[UR4][R10.64], R9;"
  ]),
  (UPat(Ops.SINK, name='x'), lambda x: [f""]),
])

class SASSRenderer(Renderer):
  device = "CUDA"
  suffix = "SASS"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  tc_sm80 = [x for x in tc.cuda_sm80 if x.dtype_in in [dtypes.half, dtypes.float]]
  def __init__(self, arch:str, device="CUDA"):
    self.device, self.arch = device, arch

  def render(self, uops:list[UOp]) -> str:
    k: list[str] = []
    for u in uops:
      print(u)

    for u in uops:
      if (l:=rewriter.rewrite(u, ctx=self)) is None:
        raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      l = cast(list[str], l)
      k.extend(l)
      pass
    kernel: list[str] = []
    kernel = [
      "[B------:R-:W-:Y:S01]   ULDC.64 UR4, c[0x0][0x118] ;",
      *k,
      "[B------:R-:W-:-:S05]   EXIT ;",
    ]

    text = "\n".join(kernel)
    return text



text = """
  E_3:
  .text.E_3:
// Load the 64-bit descriptor that defines HOW to access global memory
// (e.g., cache policy) into uniform register UR4. This is the same for all threads.
      [B------:R-:W-:Y:S01]   ULDC.64 UR4, c[0x0][0x118] ;

// Get the thread ID (in R6) and the stride for floats (4, in R7)
      [B------:R-:W-:-:S01]   S2R R6, SR_TID.X ;
      [B------:R-:W-:-:S01]   MOV R7, 0x4 ;

// --- Calculate Addresses ---
// Calculate the 64-bit address for this thread's element in each array.
// The constants (0x160, 0x168, 0x170) hold the base pointers (data0, data1, data2).
// These instructions answer "WHERE" to access memory.
      [B------:R-:W-:-:S01]   IMAD.WIDE R10, R6, R7, c[0x0][0x160] ;
      [B------:R-:W-:-:S01]   IMAD.WIDE R2, R6, R7, c[0x0][0x168] ;
      [B------:R-:W-:-:S01]   IMAD.WIDE R4, R6, R7, c[0x0][0x170] ;

// --- Perform the Operation ---
// 1. Load from the calculated addresses using the descriptor in UR4
      [B------:R-:W0:-:S05]   LDG.E R2, desc[UR4][R2.64] ;
      [B------:R-:W0:-:S05]   LDG.E R5, desc[UR4][R4.64] ;

// 2. Multiply the loaded values
      [B0-----:R-:W-:-:S04]   FMUL R9, R2, R5 ;

// 3. Store the result back to memory
      [B------:R-:W-:-:S01]   STG.E desc[UR4][R10.64], R9 ;

   )   [B------:R-:W-:-:S05]   EXIT ;
  """
"""
extern "C" __global__ void __launch_bounds__(3) E_3(float* data0_3, float* data1_3, float* data2_3) {
  int lidx0 = threadIdx.x; /* 3 */
  float val0 = (*(data1_3+lidx0));
  float val1 = (*(data2_3+lidx0));
  *(data0_3+lidx0) = (val0*val1);
}
"""


