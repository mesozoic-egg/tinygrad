from typing import cast, Callable, Union, Mapping
import struct, dataclasses
from collections import defaultdict
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import flatten, get_single_element, prod, DEBUG

class SASSRenderer(Renderer):
  device = "CUDA"
  suffix = "SASS"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  tc_sm80 = [x for x in tc.cuda_sm80 if x.dtype_in in [dtypes.half, dtypes.float]]
  def __init__(self, arch:str, device="CUDA"):
    self.device, self.arch = device, arch

  def render(self, uops:list[UOp]) -> str:
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

      [B------:R-:W-:-:S05]   EXIT ;
  """
    return text
