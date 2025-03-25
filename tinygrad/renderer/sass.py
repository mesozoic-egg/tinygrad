from typing import cast, Callable
import struct
from collections import defaultdict
from tinygrad.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import flatten, get_single_element

class SASSRenderer(Renderer):
  device = "NV"
  suffix = "SASS"
  def __init__(self, arch: str, device="NV"):
    self.device, self.arch = device, arch
    self.tensor_cores = []
    
  def render_kernel(self, kernel, function_name, bufs, regs) -> str:
    return "Rendered kernel" 
  
  def render(self, uops: list[UOp]) -> str:
    return """
.version 8.7
.target sm_86
.address_size 64

.visible .entry E_32(
	.param .u64 oned_fill_param_0
)
{
<sass>
      [B------:R-:W-:-:S02]         MOV R1, c[0x0][0x28] ;
      [B------:R-:W0:-:S01]         S2R R2, SR_CTAID.X ;
      [B------:R-:W-:-:S01]         MOV R3, 0x4 ;
      [B------:R-:W-:-:S01]         ULDC.64 UR4, c[0x0][0x118] ;
      [B------:R-:W-:-:S03]         MOV R5, 0x3f800000 ;
      [B0-----:R-:W-:-:S05]         IMAD.WIDE R2, R2, R3, c[0x0][0x160] ;
      [B------:R-:W-:-:S01]         STG.E desc[UR4][R2.64], R5 ;
      [B------:R-:W-:-:S05]         EXIT ;
  .L_x_0:
      [B------:R-:W-:Y:S00]         BRA `(.L_x_0);
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
      [B------:R-:W-:Y:S00]         NOP;
  .L_x_1:
</sass>
}
"""