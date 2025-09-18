from typing import cast, Callable, Union, Mapping, Optional, List
import struct, dataclasses, re
from collections import defaultdict, OrderedDict
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import flatten, get_single_element, prod, DEBUG

def oneline_uop(u: UOp): return repr(u).split('\n')[0]

class RegMeta(type):
  _class_instances: dict[type, dict[int, 'RegBase']] = {}
  def __call__(cls, id: int):
    if cls not in RegMeta._class_instances:
      RegMeta._class_instances[cls] = {}
    instances = RegMeta._class_instances[cls]
    if instances.get(id) is None:
      instance = super().__call__(id)
      instances[id] = instance
    return instances[id]

class RegBase(metaclass=RegMeta):
  size: int=64 # bits
  def __init__(self, id: int): self.id = id
  def __repr__(self): return self.render64()
  def render8(self): return f"R{self.id}.B8"
  def render16(self): return f"R{self.id}.B16"
  def render32(self): return f"R{self.id}"
  def render64(self): return f"R{self.id}.64"

class IReg(RegBase): pass
class FReg(RegBase): pass

def _index(ctx, x):
  src0, src1 = x.src[0], x.src[1]
  regs = ctx.r.assign_multiple([src1, x], IReg)
  src1_reg, reg = regs
  src1_str = src1_reg.render32()
  reg_str = reg.render32()
  cbank = 0x160 + src0.arg * 8
  return [
    f"[B------:R-:W-:-:S01] IMAD.WIDE {reg_str}, {src1_str}, R0, c[0x0][{cbank:#x}];",
    f"[B------:R-:W-:Y:S00] NOP;",
    ]

def _special(ctx, x):
  idx_pat  = re.compile(r'^(g|l)idx(\d)')
  idx_match = idx_pat.match(x.arg)
  assert idx_match is not None, f"{x.arg=}"
  idx_type, dim_str = idx_match.groups()
  reg_type = "SR_TID" if idx_type == "l" else "SR_CTAID"
  dim = ['X', 'Y', 'Z'][int(dim_str)]
  return [f"[B------:R-:W-:-:S01]   S2R {ctx.r.assign_i32(x)}, {reg_type}.{dim};"]

rewriter = PatternMatcher([
  (UPat(Ops.DEFINE_GLOBAL, name='x'), lambda ctx, x: [f""]),
  (UPat(Ops.SPECIAL, name='x'), _special),
  (UPat(Ops.CONST, name='x'), lambda ctx, x: [
    f"[B------:R-:W-:-:S01]   MOV {ctx.r.assign_i32(x)}, {x.arg:#x} ;"]),
  (UPat(Ops.INDEX, name='x'), _index),
  (UPat(Ops.LOAD, name='x'), lambda ctx, x: [
    f"[B------:R-:W0:-:S05] LDG.E {ctx.r.assign_i32(x)}, desc[UR4][{ctx.r.assign_i32(x.src[0])}.64];"
  ]),
  (UPat(Ops.MUL, name='x'), lambda ctx, x: [
    f"[B0-----:R-:W-:-:S04] FMUL {ctx.r.assign_i32(x)}, {ctx.r.assign_i32(x.src[0])}, {ctx.r.assign_i32(x.src[1])};"
  ]),
  (UPat(Ops.STORE, name='x'), lambda ctx, x: [
    f"[B------:R-:W-:-:S01] STG.E desc[UR4][{ctx.r.assign_i64(x.src[0])}], {ctx.r.assign_i32(x.src[1])};"
  ]),
  (UPat(Ops.SINK, name='x'), lambda ctx, x: [f""]),
])

class Variable:
  def __init__(self, uop: UOp, start: int, end: int):
    """
    Args:
    end:  the index in the linearized uops *after* which the variable is expired
          start and end are both inclusive. 
    size: size in bytes (int32: 4, float64: 8)
    """
    self.uop, self.start = uop, start
    self._end = end
    self._reg: Optional[RegBase] = None
    self._stack: Optional[int] = None
    self.mem: Optional[str] = None
    self.track_reg: bool = False
    self.track_stack: bool = False
    self.track_var_end: bool = False
  
  @property
  def name(self): return repr(self.uop)[:100]

  @property
  def reg(self): return self._reg
  @reg.setter
  def reg(self, v: Union[RegBase, None]):
    if self.track_reg:
      print(f"\033[31m{v} -> {self=}\033[0m")
      print(f"\t{oneline_uop(self.uop)}")
      print(f"====")
    self._reg = v
  @property
  def stack(self): return self._stack
  @stack.setter
  def stack(self, v: int): self._stack = v
  
  @property
  def end(self): return self._end
  @end.setter
  def end(self, v: int):
    prev = self._end
    self._end = v
    if self.track_var_end:
      print(f"\033[31m Interval end :{prev=} -> {self._end=} {self.uop}\033[0m\n")

  def __repr__(self):
    location = f" reg:{self.reg}" if self.reg is not None else f" stack:{self.stack}" if self.stack is not None else ""
    return f"({self.start}-{self.end} reg:{self.reg} stack:{self.stack})"

  def store(self, dst: str="") -> list[str]:
    assert self.reg is not None
    assert self.stack is not None
    return move_reg_mem("str", self.reg, self.stack, self.uop.dtype.itemsize)

  def load(self, reg: RegBase, src: str="") -> list[str]:
    assert self.stack is not None
    return move_reg_mem("ldr", reg, self.stack, self.uop.dtype.itemsize)

  def copy(self, dst: RegBase) -> list[str]:
    assert self.reg is not None
    if Arch.arm:
      if isinstance(self.reg, IReg) and isinstance(dst, IReg):
        op = "mov"
      else:
        op = "fmov"
    else:
      if isinstance(self.reg, IReg) and isinstance(dst, IReg):
        op = "mov"
      else:
        op = "movq"
    return [f"{op} {dst.render64()}, {self.reg.render64()}"]

x86_params: dict[int, int] = {
  0: 7, #R7 (rdi)
  1: 6, #R6 (rsi)
  2: 2, #R2 (rdx)
  3: 1, #R1 (rcx)
  4: 8, #R8
  5: 9, #R9 
}
class AllocatorPool:
  def __init__(self, reg_type: type[RegBase], num: int):
    self.reg_type, self.num = reg_type, num
    self._pool: list[RegBase] = [reg_type(i) for i in range(0, num, 2)]
    self._acquired: dict[RegBase, set[Variable]] = defaultdict(set)

  def __repr__(self):
    l = []
    l.append(f"Pool: {self._pool}")
    l.append(f"Acquired:")
    for reg, vars in self._acquired.items():
      l.append(f"\t{reg}: {vars}")
    return "\n".join(l)

  @property
  def pool(self):
    return self._pool

  def __len__(self): return len(self._pool)

  def pop(self, i):
    reg = self._pool.pop(i)
    #print(f"\033[31m{reg} popped\033[0m")
    return reg
  
  def insert(self, i, v):
    #print(f"\033[32m{v} returned to pool\033[0m")
    self._pool.insert(i, v)

  def index(self, reg):
    return self._pool.index(reg)

  def __getitem__(self, i):
    return self._pool[i]

  def acquire_reg(self, reg: RegBase, var: Variable):
    #print(f"\033[33m{reg} acquired by {var} {oneline_uop(var.uop)}\033[0m")
    self._acquired[reg].add(var)
  def release_reg(self, reg: RegBase, var: Variable):
    assert reg is not None
    #print(f"\033[34m{reg} released from {var} {oneline_uop(var.uop)}\033[0m")
    acquired = self._acquired[reg]
    if var not in acquired: raise Exception(f"Not yet acquired: {var=} {reg=} {acquired=}")
    self._acquired[reg].discard(var)
    if len(self._acquired[reg]) == 0:
      del self._acquired[reg]

  def bookkeeping(self):
    if len(self._pool) + len(self._acquired) != self.num:
      print(f"{self._pool=}")
      for reg, vars in self._acquired.items():
        print(f"\t{reg}: {vars}")
      raise Exception(f"Inconsistent pool + acquired and total reg number")
    for reg, vars in self._acquired.items():
      for var in vars:
        if var.reg != reg:
          print(f"{var=} {reg=}, {self._pool}")
          for _reg, vars in self._acquired.items():
            print(f"\t{_reg}: {vars}")
          raise Exception(f"Inconsistent var.reg: {var.reg} and acquired record: {reg}")
class Allocator:
  def __init__(self, num):
    self.pools: dict[type[RegBase], AllocatorPool] = {
      IReg: AllocatorPool(IReg, num),
    }
    self.uops: dict[UOp, Variable] = {}
    self.reserved: dict[RegBase, int] = {}
    self.blocked: list[RegBase] = [IReg(0)]
    self.stack_size = 0
    self.cur_step = 0
    self.kernel: list[str] = []
    self.tracked_regs: list[RegBase] = []

  def bookkeeping(self):
    for pool in self.pools.values():
      pool.bookkeeping()

  def flush_kernel(self) -> list[str]:
    ret = self.kernel
    self.kernel = []
    return ret

  def alloc(self, reg_type: type[RegBase],
            excludes: list[RegBase]=[],
            debug:bool=False
            ) -> RegBase:
    return self.alloc_multiple(1, reg_type, excludes)[0]

  def alloc_multiple(self, num: int, reg_type: type[RegBase], excludes: list[RegBase]):
    pool = self.pools[reg_type]
    regs = []
    if len(pool):
      idx, count = 0, 0
      while idx < len(pool) and count < num:
        _reg = pool[idx]
        if _reg not in self.blocked and _reg not in excludes and _reg not in self.reserved:
          regs.append(pool.pop(idx))
          count += 1
        else:
          idx += 1
    if len(regs) == num:
      return regs
    num_to_spill = num - len(regs)
    candidates = self._find_spill_candidates(num_to_spill, reg_type, excludes)
    for uop, var in candidates:
      reg = var.reg
      assert reg is not None
      self._spill(reg)
      regs.append(reg)
    return regs

  def share(self, dst: UOp, src: UOp):
    dst_var, src_var = self.uops[dst], self.uops[src]
    reg = src_var.reg
    assert reg, f"Source UOp must already been assigned to register {src} {reg=}"
    dst_var.reg = reg
    pool = self.pools[type(reg)]
    pool.acquire_reg(reg, dst_var)
    

  def return_reg(self, regs: list[RegBase]):
    for reg in regs:
      if reg in self.tracked_regs:
        print(f"\033[31m{reg=} back to pool\033[0m")
      self.pools[type(reg)].insert(0, reg)

  def move_var_to_stack(self, v: Variable):
    reg = v.reg
    assert reg
    assert reg is not None
    self._spill(reg)
    self.return_reg([reg])
    v.reg = None

  def assign(self, _key: UOp,
             reg_type: type[RegBase],
             excludes: list[RegBase]=[], reserve: bool=False,
             debug:bool=False,
             ) -> RegBase:
    var = self.uops[_key]
    if var.reg is not None:
      reg = var.reg
      return reg
    else:
      reg = self.alloc_multiple(1, excludes=excludes, reg_type=reg_type)[0]
    if var.stack is not None:
      self.kernel.extend(var.load(reg))
      var.reg = reg
    if reserve: self.reserved[reg] = 1
    var.reg = reg
    self.pools[reg_type].acquire_reg(reg, var)
    return reg
  def assign_i8(self, _key: UOp, excludes: list[RegBase]=[], reserve: bool = False):
    return self.assign(_key, IReg, excludes, reserve).render8()
  def assign_i32(self, _key: UOp, excludes: list[RegBase]=[], reserve: bool = False):
    return self.assign(_key, IReg, excludes, reserve).render32()
  def assign_i64(self, _key: UOp, excludes: list[RegBase]=[], reserve: bool = False, debug: bool = False):
    return self.assign(_key, IReg, excludes, reserve, debug).render64()
  def assign_f32(self, _key: UOp, excludes: list[RegBase]=[], reserve: bool = False):
    return self.assign(_key, FReg, excludes, reserve).render32()
  def assign_f64(self, _key: UOp, excludes: list[RegBase]=[], reserve: bool = False):
    return self.assign(_key, FReg, excludes, reserve).render64()
  def assign_reg(self, reg: RegBase, _key: UOp, reserve: bool=False) -> None:
    uop = _key
    var = self.uops[uop]
    self.alloc_reg(reg)
    if var.reg is not None:
      self.kernel.extend(var.copy(reg))
    var.reg = reg
    self.pools[type(reg)].acquire_reg(reg, var)

  def alloc_reg(self, reg: RegBase) -> None:
    pool = self.pools[type(reg)]
    if reg in pool:
      pool.pop(pool.index(reg))
    else:
      self._spill(reg)

  def assign_multiple(self, uops: List[UOp], reg_type: type[RegBase], excludes: list[RegBase]=[]) -> list[RegBase]:
    regs: list[Optional[RegBase]] = [None] * len(uops)
    need_alloc: list[int] = []
    for i, uop in enumerate(uops):
      _reg = self.uops[uop].reg
      if _reg is None:
        need_alloc.append(i)
      else:
        regs[i] = _reg
    existing_regs = [reg for reg in regs if reg is not None]
    alloc_regs = self.alloc_multiple(len(need_alloc), reg_type, existing_regs + excludes)
    for i, reg in zip(need_alloc, alloc_regs):
      uop = uops[i]
      var = self.uops[uop]
      if var.stack is not None:
        self.kernel.extend(var.load(reg))
        var.reg = reg
      regs[i] = reg
      var.reg = reg
    for reg, uop_i in zip(alloc_regs, need_alloc):
      var = self.uops[uops[uop_i]]
      self.pools[reg_type].acquire_reg(reg, var)
    for reg in regs:
      assert reg is not None
    regs2 = cast(list[RegBase], regs)
    return regs2

  def release(self, reg: RegBase): del self.reserved[reg] 

  def free_expired(self, i: int):
    expired: list[UOp] = []
    assigned_regs: dict[RegBase, set[Variable]] = defaultdict(set)
    for uop, var in self.uops.items():
      if var.end < i: expired.append(uop)
      if var.reg: assigned_regs[var.reg].add(var)
      if var.reg and var.end < i:
        reg = var.reg
        pool = self.pools[type(reg)]
        pool.release_reg(reg, var)
        assigned_regs[var.reg].remove(var)
    for uop in expired:
      del self.uops[uop]
    for reg, vars in assigned_regs.items():
      if len(vars) == 0:
        pool = self.pools[type(reg)]
        pool.insert(0, reg)
        if self.reserved.get(reg):
          del self.reserved[reg]
  def _spill(self, reg: RegBase) -> None:
    pool = self.pools[type(reg)]
    vars = self.find_vars_holding_reg(reg)
    for var in vars:
      assert var.reg is not None
      if var.stack is None:
        self.stack_size += (var.reg.size // 8)
        var.stack = self.stack_size
      self.kernel.extend(var.store())
    for var in vars:
      assert var.reg is not None
      pool.release_reg(var.reg, var)
      var.reg = None
  def _find_spill_candidates(self, num: int, reg_type: type[RegBase], excludes: list[RegBase]=[]):
    candidates: list[tuple[UOp, Variable]] = []
    for u, v in self.uops.items():
      if v.reg is not None:
        candidates.append((u, v))
    candidates = [(u,v) for u, v in candidates if type(v.reg) == reg_type]
    candidates = [(u,v) for u, v in candidates if v.reg not in self.reserved]
    candidates = [(u,v) for u, v in candidates if v.reg not in self.blocked]
    candidates = [(u,v) for u, v in candidates if v.reg not in excludes]
    assert len(candidates), f"no candidates left {reg_type=} {self.reserved=}"
    candidates = sorted(candidates, key=lambda u_v: u_v[1].end, reverse=True)
    assert len(candidates) >= num, "Not enough registers to fulfill spill"
    candidates = candidates[:num]
    return candidates
  def find_vars_holding_reg(self, reg: RegBase) -> list[Variable]:
    vars: list[Variable] = []
    for v in self.uops.values():
      if v.reg == reg: vars.append(v)
    return vars

extra_matcher = PatternMatcher([
  #(UPat(Ops.INDEX, name='x', src=(UPat(), UPat())), lambda x: x.replace(src=(*x.src, UOp(Ops.CONST, dtype=dtypes.int, arg=4)))),
  #(UPat(Ops.SPECIAL, ),
])

class SASSRenderer(Renderer):
  device = "CUDA"
  suffix = "SASS"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  tc_sm80 = [x for x in tc.cuda_sm80 if x.dtype_in in [dtypes.half, dtypes.float]]
  extra_matcher = extra_matcher
  def __init__(self, arch:str, device="CUDA"):
    self.device, self.arch = device, arch

  def render(self, uops:list[UOp]) -> str:
    self.r = Allocator(32)
    lines: list[str] = []
    var_intervals: dict[UOp, Variable] = OrderedDict()
    for i, u in enumerate(uops):
      var = Variable(u, i, -1)
      var_intervals[u] = var

    for i, u in enumerate(uops):
      if u.op in [Ops.SPECIAL, Ops.SINK]: continue
      for src in u.src:
        if src.dtype is not dtypes.void:
          prev = var_intervals[src].end
          var_intervals[src].end = max(prev, i)

    for v in var_intervals.values():
      if v.end == -1: v.end = len(uops)
    self.r.uops = var_intervals

    name: str=None
    for u in uops:
      if u.op is Ops.DEFINE_GLOBAL:
        pass
      elif u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
      else:
        print(f'\n{u}')
        if (l:=rewriter.rewrite(u, ctx=self)) is None:
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        print(l)
        l = cast(list[str], l)
        lines.extend(l)
        pass
    kernel: list[str] = []
    assert name is not None
    kernel = [
      f"//{name}",
      "[B------:R-:W-:Y:S01]   ULDC.64 UR4, c[0x0][0x118] ;",
      "[B------:R-:W-:-:S01]   MOV R0, 0x4 ;",
      *lines,
      "[B------:R-:W-:-:S05]   EXIT ;",
    ]

    text = "\n".join(kernel)
    #print(text)
    #raise Exception()
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


