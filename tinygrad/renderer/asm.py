from sqlite3 import Binary
from typing import List, Dict, Optional, cast
from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat, UPatAny
from tinygrad.renderer import Renderer
from tinygrad.helpers import DEBUG 
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType
from collections import OrderedDict, defaultdict
from typing import List, Dict, Optional, cast, Literal, Union, Callable
import struct, math, unittest, platform, enum, os
import platform

class ArchType(enum.Enum):
  ARM = "aarch64"
  X86 = "x86_64"
  @classmethod
  def from_platform(cls, value: str):
    candidates = []
    for member in cls:
      if value == member.value:
        candidates.append(member)
    assert len(candidates) == 1, "Platform type is not unique or not found"
    return candidates[0]

class _Arch:
  def __init__(self): self.arch = ArchType.from_platform(platform.machine())
  @property
  def arm(self): return self.arch == ArchType.ARM
  @property
  def x86(self): return self.arch == ArchType.X86
Arch = _Arch()

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
  size: int # bits
  def __init__(self, id: int): self.id = id
  def __repr__(self): return self.render64()
  def render8(self):  raise NotImplementedError()
  def render32(self): raise NotImplementedError()
  def render64(self): raise NotImplementedError()
  def render(self, itemsize: int): raise NotImplementedError()

class IReg(RegBase):
  size = 64
  def render8(self):
    if Arch.arm: return self.render32()
    else:
      if self.id < 8:
        return ["al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil"][self.id]
      else:
        return f"r{self.id}b"
  def render32(self):
    if Arch.arm: return f"w{self.id}"
    else: return ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
      "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"][self.id]
  def render64(self):
    if Arch.arm: return f"x{self.id}"
    else: return ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"][self.id]
  def render(self, itemsize: int):
    """
    itemsize: bytes
    """
    if itemsize == 1: return self.render8()
    if itemsize == 4: return self.render32()
    if itemsize == 8: return self.render64()
    raise Exception(f"Either 4 or 8 bytes for register, received {itemsize}")

class FReg(RegBase):
  size = 128
  def render32(self):
    return f"s{self.id}" if Arch.arm else f"xmm{self.id}"
  def render64(self):
    return f"d{self.id}" if Arch.arm else f"xmm{self.id}"
  def render(self, itemsize: int):
    if itemsize == 4: return self.render32()
    if itemsize == 8: return self.render64()
    raise Exception(f"Either 4 or 8 bytes for register, received {itemsize}")

def oneline_uop(u: UOp): return repr(u).split('\n')[0]

def move_reg_mem(op: Union[Literal["str"], Literal["ldr"]],
                 reg: RegBase, stack: int, size: int):
  if Arch.arm:
    sp = "x29"
    sub = []
    add = []
    while stack > 255:
      sub.append(f"sub x29, x29, #255")
      add.insert(0, f"add x29, x29, #255")
      stack -= 255
    assert stack <=255
    op = "str" if op == "str" else "ldr"
    return [
      *sub,
      f"{op} {reg.render64()}, [{sp}, #-{stack}]",
      *add,
    ]
  else:
    if type(reg) is IReg:
      _op = "mov"
    else:
      _op = "movss" if size == 4 else "movsd"

    if op == "str":
      return [f"{_op} [rbp - {stack}], {reg.render64()}"]
    else:
      return [f"{_op} {reg.render64()}, [rbp - {stack}]"]


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
    self._pool: list[RegBase] = [reg_type(i) for i in range(num)]
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
  def __init__(self, num_ireg: int, num_freg: int):
    self.pools: dict[type[RegBase], AllocatorPool] = {
      IReg: AllocatorPool(IReg, num_ireg),
      FReg: AllocatorPool(FReg, num_ireg),
    }
    self.uops: dict[UOp, Variable] = {}
    self.reserved: dict[RegBase, int] = {}
    self.blocked: list[RegBase] = [IReg(4)]
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

x86_params_mapping: dict[int, int] = {
  0: 7, #R7 (rdi)
  1: 6, #R6 (rsi)
  2: 2, #R2 (rdx)
  3: 1, #R1 (rcx)
  4: 8, #R8
  5: 9, #R9 
}

def stack_all(a: Allocator):
  for u, var in a.uops.items():
    # Previously was also checking var.stack and missed updated value
    if var.reg is not None and var.reg not in a.reserved:
      a.move_var_to_stack(var)

def float32_to_hex(f: float) -> str:
  return hex(int.from_bytes(struct.pack('<f', f), 'little'))

class _AluOps:
  def __init__(self, items: dict[tuple[Union[ArchType, Ops, type[RegBase], int], ...], str]):
    self._root = {}
    for (key, value) in items.items():
      node = self._root
      for i, part in enumerate(key): node = node.setdefault(part, {})
      node[None] = value
    
  def get(self, key_tuple: tuple[Union[ArchType, Ops, type[RegBase], int], ...]) -> str:
    node = self._root
    best_match = None
    for i, part in enumerate(key_tuple):
      if part not in node: break
      node = node[part]
      if None in node: best_match = node[None]
    if best_match is None: raise Exception(f"No match found {key_tuple=}")
    return best_match

AluOps = _AluOps({
  (Ops.ADD, ArchType.X86, IReg): "add",
  (Ops.ADD, ArchType.X86, FReg, 32): "addss",
  (Ops.ADD, ArchType.X86, FReg, 64): "addsd",
  (Ops.ADD, ArchType.ARM, IReg): "add",
  (Ops.ADD, ArchType.ARM, FReg): "fadd",
  (Ops.SUB, ArchType.ARM, IReg): "sub",
  (Ops.SUB, ArchType.ARM, FReg): "fsub",
  (Ops.MUL, ArchType.X86, IReg): "imul",
  (Ops.MUL, ArchType.X86, FReg, 32): "mulss",
  (Ops.MUL, ArchType.X86, FReg, 64): "mulsd",
  (Ops.MUL, ArchType.ARM, IReg): "mul",
  (Ops.MUL, ArchType.ARM, FReg): "fmul",
  (Ops.ASSIGN, ArchType.ARM, IReg): "mov",
  (Ops.ASSIGN, ArchType.ARM, FReg): "fmov",
  (Ops.ASSIGN, ArchType.X86, IReg, 8): "mov",
  (Ops.ASSIGN, ArchType.X86, IReg, 32): "mov",
  (Ops.ASSIGN, ArchType.X86, IReg, 64): "mov",
  (Ops.ASSIGN, ArchType.X86, FReg, 32): "movss",
  (Ops.ASSIGN, ArchType.X86, FReg, 64): "movsd",
  (Ops.SQRT, ArchType.X86, FReg, 32): "sqrtss",
  (Ops.SQRT, ArchType.X86, FReg, 64): "sqrtsd",
  (Ops.SQRT, ArchType.ARM, FReg): "fsqrt",
  (Ops.IDIV, ArchType.X86, FReg, 32): "idiv",
  (Ops.AND,): "and",
  (Ops.OR, ArchType.X86): "or",
  (Ops.OR, ArchType.ARM): "orr",
  (Ops.XOR, ArchType.X86): "xor",
  (Ops.XOR, ArchType.ARM): "eor",
  (Ops.MAX, ArchType.ARM, FReg, 32): "fmax",
  (Ops.MAX, ArchType.ARM, IReg): "smax",
  (Ops.MAX, ArchType.X86, FReg, 32): "maxss",
  (Ops.MAX, ArchType.X86, FReg, 64): "maxsd",
})

def alu(ctx, x):
  dtype = x.src[0].dtype
  reg_type = IReg if dtypes.is_int(dtype) or dtypes.is_bool(dtype) else FReg
  src_regs = ctx.r.assign_multiple(list(x.src), reg_type)

  if ctx.r.uops[x.src[0]].end == ctx.r.cur_step:
    ctx.r.share(x, x.src[0])
    dst = src_regs[0]
  else:
    dst = ctx.r.assign(x, reg_type, src_regs)
  operator = AluOps.get((x.op, Arch.arch, reg_type, 8*x.dtype.itemsize))
  _dst = dst.render(dtype.itemsize)
  src_regs_str = [reg.render(dtype.itemsize) for reg in src_regs]
  if Arch.arm:
    return [f"{operator} {_dst}, {', '.join(src_regs_str)}"]
  else:
    _mov = "mov" if dtypes.is_int(dtype) or dtypes.is_bool(dtype) else "movss" 
    if dst == src_regs[0] and len(src_regs_str) == 2:
      return [f"{operator} {_dst}, {src_regs_str[1]}"]
    elif len(src_regs_str) == 2:
      clear_op = "xor" if reg_type is IReg else "xorps" if dtype.itemsize == 4 else "xorpd"
      return [
        f"{clear_op} {dst}, {dst}",
        f"{_mov} {_dst}, {src_regs_str[0]}",
        f"{operator} {_dst}, {src_regs_str[1]}",]
    elif _dst == src_regs_str[0] and len(src_regs_str) == 1:
      return [f"{operator} {_dst}, {src_regs_str[0]}"]
    elif len(src_regs_str) == 1:
      return [f"{operator} {_dst}, {src_regs_str[0]}"]
    else:
      raise Exception("ALU error handling srcs")

def acc(ctx, x, acc, src):
  dtype = x.src[0].dtype
  reg_type = FReg if dtypes.is_float(acc.dtype) else IReg
  acc_reg, src_reg = ctx.r.assign_multiple([acc, src], reg_type=reg_type)
  _acc, _src = acc_reg.render(dtype.itemsize), src_reg.render(dtype.itemsize)
  ctx.r.share(x, acc)
  reg_type = IReg if dtypes.is_int(dtype) else FReg
  operator = AluOps.get((Ops.ADD, Arch.arch, reg_type, 8*x.dtype.itemsize))
  if Arch.arm:
    return [f"{operator} {_acc}, {_acc}, {_src}"]
  else:
    return [f"{operator} {_acc}, {_src}"]

def const(ctx, x):
  label = f"const_{len(ctx.mem)}"
  if Arch.arm:
    reg_type = FReg if dtypes.is_float(x.dtype) else IReg
    reg = ctx.r.assign(x, reg_type=reg_type)
    reg_str = reg.render(x.dtype.itemsize)
    if x.dtype == dtypes.int64 or x.dtype == dtypes.uint64:
      data_type = ".quad"
    elif x.dtype == dtypes.int32 or x.dtype == dtypes.uint32:
      data_type = ".word"
    elif x.dtype.itemsize == 4:
      data_type = ".single"
    else:
      data_type = ".double"
    ctx.mem.append((label, f"{data_type} {x.arg}"))
    temp_reg = ctx.r.alloc(IReg, [reg])
    ctx.r.return_reg([temp_reg])
    return [f"adrp {temp_reg}, {label}",
      f"ldr {reg_str}, [{temp_reg}, #:lo12:{label}]"]
  else:
    if dtypes.is_int(x.dtype):
      raise Exception("Do not handle integer on x86 in the data section on x86")
    reg = ctx.r.assign(x, reg_type=FReg)
    reg_str = reg.render(x.dtype.itemsize)
    if x.dtype.itemsize == 4:
      data_type = ".float"
      op = "movss"
    else:
      data_type = ".double"
      op = "movsd"
    ctx.mem.append((label, f"{data_type} {x.arg}"))
    return [ f"{op} {reg_str}, [rip+{label}]" ]

def _range(ctx, x):
  stack_all(ctx.r)
  counter = ctx.r.assign(x, reserve=True, reg_type=IReg).render64()
  return [
      f"mov {counter}, #0",
      f".LOOP_{x.arg}:"
  ]

def endrange(ctx, x):
  acc, end = x.src[0], x.src[0].src[0]
  stack_all(ctx.r)
  acc_reg = ctx.r.assign_i64(acc)
  if Arch.arm:
    return [
      f"add {acc_reg}, {acc_reg}, #1",
      f"cmp {acc_reg}, #{end.arg}",
      f"b.lt .LOOP_{acc.arg}"
    ]
  else:
    return [
      f"inc {acc_reg}",
      f"cmp {acc_reg}, {end.arg}",
      f"jl .LOOP_{acc.arg}",
    ]

def _index(ctx, x):
  src0, src1 = x.src[0], x.src[1]
  regs = ctx.r.assign_multiple([src0, src1, x], IReg)
  src0_reg, src1_reg, reg = regs
  assert src0_reg != src1_reg and src0_reg != reg and src1_reg != reg
  src0_str = src0_reg.render64()
  src1_str = src1_reg.render64()
  multiplier = src0.dtype.itemsize
  lsl = int(math.log2(multiplier))
  if Arch.arm:
    return [ f"add {reg}, {src0_str}, {src1_str}, lsl #{lsl}" ]
  else:
    return [ f"lea {reg}, [{src0_str} + {src1_str} * {multiplier}]" ]

def assign(ctx, x):
  reg_type = IReg if dtypes.is_int(x.src[0].dtype) or dtypes.is_bool(x.src[0].dtype) else FReg
  x_src_0_reg = ctx.r.assign(x.src[0], reg_type=reg_type)
  ctx.r.share(x, x.src[0])
  dst, src = ctx.r.assign_multiple([x, x.src[1]], excludes=[x_src_0_reg], reg_type=reg_type)
  
  opcode = AluOps.get((x.op, Arch.arch, reg_type, 8*x.dtype.itemsize))
  ctx.r.uops[x].stack = ctx.r.uops[x.src[0]].stack
  return [f"{opcode} {dst}, {src}"]

def to_bool(ctx, x, a):
  if dtypes.is_int(a.dtype):
    reg_type = IReg
    regs = ctx.r.assign_multiple([x, a], reg_type=IReg)
    dst, src = regs
    temp_reg = ctx.r.alloc(excludes=regs, reg_type=reg_type)
    ctx.r.return_reg([temp_reg])
  else:
    reg_type = FReg
    dst = ctx.r.assign(x, IReg)
    src = ctx.r.assign(a, FReg)
    temp_reg = ctx.r.alloc(FReg, [src])
    ctx.r.return_reg([temp_reg])
  if Arch.arm:
    if dtypes.is_int(a.dtype):
      cmp = f"cmp {src}, #0"
    else:
      cmp = f"fcmp {src}, #0.0"
    return [
      cmp,
      f"cset {dst}, ne"       # Set dst=1 if not equal, else 0
    ]
  else:
    if dtypes.is_int(a.dtype):
      test_op = "cmp"
      reset_op = "xor"
    else:
      test_op = "ucomiss" if a.dtype.itemsize == 4 else "ucomisd"
      reset_op = "xorps"
    return [
      f"xor {dst}, {dst}",
      f"{reset_op} {temp_reg}, {temp_reg}",
      f"{test_op} {temp_reg}, {src}", # ZF=1 => src == 0, ZF=0 => src != 0
      f"setne {dst.render8()}", # set dst to 1 if ZF == 0 => src != 0
    ]


def cmp_int_x86(ctx, x, a, b):
  reg_type=IReg
  regs = ctx.r.assign_multiple([x, a, b], IReg)
  dst, src_a, src_b = regs
  temp_regs = ctx.r.alloc_multiple(2, IReg, [src_a, src_b, dst])
  temp_reg, temp_reg_2 = temp_regs
  ctx.r.return_reg([temp_reg])
  ctx.r.return_reg([temp_reg_2])
  size = a.dtype.itemsize
  mov_op = "mov"
  cmp_op = "cmp"
  set_op = "setl" if x.op is Ops.CMPLT else "setne"
  return [
    f"xor {dst}, {dst}",
    f"{mov_op} {temp_reg.render(size)}, {src_a.render(size)}",
    f"{cmp_op} {temp_reg.render(size)}, {src_b.render(size)}", #CF=1 => src_a < src_b, CF=0 => src_a >= src_b
    f"{set_op} {dst.render8()}", #dst=1 if CF=1 => src_a < src_b
  ]

def cmpne_float_x86(ctx, x, a, b):
  dst = ctx.r.assign(x, IReg)
  if a == b:
    src_a = src_b = ctx.r.assign(a, FReg)
  else:
    src_a, src_b = ctx.r.assign_multiple([a, b], FReg)
  temp_reg = ctx.r.alloc(FReg, [src_a, src_b, dst])
  temp_reg_2 = ctx.r.alloc(IReg, [src_a, src_b, dst])
  ctx.r.return_reg([temp_reg, temp_reg_2])
  size = a.dtype.itemsize
  cmp_op = "ucomiss" if a.dtype.itemsize == 4 else "comisd"
  mov_op = "movss" if a.dtype.itemsize == 4 else "movsd"
  return [
    f"xor {temp_reg_2}, {temp_reg_2}",
    f"xor {dst}, {dst}",
    f"{mov_op} {temp_reg.render(size)}, {src_a.render(size)}",
    f"{cmp_op} {temp_reg.render(size)}, {src_b.render(size)}",
    f"setp {temp_reg_2.render8()}",
    f"setne {dst.render8()}",
    f"or {dst}, {temp_reg_2}",
  ]
def cmplt_float_x86(ctx, x, a, b):
  dst = ctx.r.assign(x, IReg)
  src_a, src_b = ctx.r.assign_multiple([a, b], FReg)
  temp_reg = ctx.r.alloc(FReg, [src_a, src_b, dst])
  temp_reg_2 = ctx.r.alloc(IReg, [src_a, src_b, dst])
  ctx.r.return_reg([temp_reg, temp_reg_2])
  size = a.dtype.itemsize
  cmp_op = "ucomiss" if a.dtype.itemsize == 4 else "ucomisd"
  mov_op = "movss" if a.dtype.itemsize == 4 else "movsd"
  return [
    f"xor {dst}, {dst}",
    f"xor {temp_reg_2}, {temp_reg_2}",
    f"{mov_op} {temp_reg.render(size)}, {src_a.render(size)}",
    f"{cmp_op} {temp_reg.render(size)}, {src_b.render(size)}",
    f"setne {temp_reg_2.render8()}",
    f"setb {dst.render8()}",
    f"and {dst}, {temp_reg_2}",
  ]

def cmp_arm(ctx, x, a, b):
  if dtypes.is_int(a.dtype) or dtypes.is_bool(a.dtype):
    dst, src_a, src_b = ctx.r.assign_multiple([x, a, b], IReg)
    op = "cmp"
  else:
    dst = ctx.r.assign(x, IReg)
    if a == b:
      src_a = src_b = ctx.r.assign(a, FReg)
    else:
      src_a, src_b = ctx.r.assign_multiple([a, b], FReg)
    op = "fcmp"
  size = a.dtype.itemsize
  cmp = "lt" if x.op is Ops.CMPLT else "ne"
  return [
      f"{op} {src_a.render(size)}, {src_b.render(size)}",
      f"cset {dst}, {cmp}"
  ]

def recip(ctx, x):
  dst, src = ctx.r.assign_multiple([x, x.src[0]], FReg)
  temp_reg = ctx.r.alloc(FReg, [dst, src])
  ctx.r.return_reg([temp_reg])
  size = x.dtype.itemsize
  if Arch.arm:
    return [f"fmov {temp_reg.render(size)}, #1.0",
            f"fdiv {dst.render(size)}, {temp_reg.render(size)}, {src.render(size)}"]
  else:
    if size == 4:
      data_type = ".float"
      mov = "movss"
      div = "divss"
    else:
      data_type = ".double"
      mov = "movsd"
      div = "divsd"
    label = f"const_{len(ctx.mem)}"
    ctx.mem.append((label, f"{data_type} 1.0"))
    return [f"{mov} {temp_reg.render(size)}, [rip + {label}]",
            f"{div} {temp_reg.render(size)}, {src.render(size)}",
            f"{mov} {dst.render(size)}, {temp_reg.render(size)}"]


def _where(ctx, x):
  if dtypes.is_int(x.dtype): reg_type = IReg
  else: reg_type = FReg
  cond, t, f = x.src
  _cond = ctx.r.assign(cond, reg_type=IReg)
  exclude_cond = [cond] if reg_type == IReg else []
  _dst, _t, _f = ctx.r.assign_multiple([x,t,f], reg_type=reg_type,
    excludes=exclude_cond)

  if Arch.arm:
    if dtypes.is_int(x.dtype): op = "csel"
    else: op = "fcsel"
    size=x.dtype.itemsize
    return [
	f"cmp {_cond}, #0",          # Test condition ≠0
	f"{op} {_dst.render(size)}, {_t.render(size)}, {_f.render(size)}, ne"  # Select _t if true, _f if false
    ]
  else:
    if dtypes.is_int(x.dtype): mov_op = "mov"
    else: mov_op = "movaps" if x.dtype.itemsize == 4 else "movapd"
    return [
      f"test {_cond}, {_cond}", #ZF=1 if _cond=0 => false
      f"jz .f_case_{ctx.r.cur_step}", #jump if ZF=1 => condition is false
      f"{mov_op} {_dst}, {_t}",
      f"jmp .end_{ctx.r.cur_step}",
      f".f_case_{ctx.r.cur_step}:",
      f"{mov_op} {_dst}, {_f}",
      f".end_{ctx.r.cur_step}:",
    ]

def x86_idiv(ctx, x):
  dividend, divisor = x.src
  vars_holding_eax = ctx.r.find_vars_holding_reg(IReg(0))
  for var in vars_holding_eax:
    if var.stack is None:
      ctx.r.stack_size += (var.reg.size // 8)
      var.stack = ctx.r.stack_size
    ctx.r.kernel.extend(var.store())
    var.reg = None
    ctx.r.pools[IReg].release_reg(IReg(0), var)
  vars_holding_edx = ctx.r.find_vars_holding_reg(IReg(2))
  for var in vars_holding_edx:
    if var.stack is None:
      ctx.r.stack_size += (var.reg.size // 8)
      var.stack = ctx.r.stack_size
    ctx.r.kernel.extend(var.store())
    var.reg = None
    ctx.r.pools[IReg].release_reg(IReg(2), var)
  mov2 = []
  _dividend, _divisor, _dst = ctx.r.assign_multiple(
    [dividend, divisor, x],
    reg_type=IReg, excludes=[IReg(0), IReg(2)])
  if len(vars_holding_eax) >= 1:
    var0 = vars_holding_eax[0]
    mov2.extend([
      *move_reg_mem("ldr", IReg(0), var0.stack, 8)
    ])
    for var in vars_holding_eax:
      if var.reg is not None:
        ctx.r.pools[IReg].release_reg(var.reg, var)
        if var.reg not in ctx.r.pools[IReg]._acquired:
          ctx.r.pools[IReg].insert(0, var.reg)
      var.reg = IReg(0)
      ctx.r.pools[IReg].acquire_reg(IReg(0), var)
  if len(vars_holding_edx) >= 1:
    var0 = vars_holding_edx[0]
    mov2.extend([
      *move_reg_mem("ldr", IReg(2), var0.stack, 8)
    ])
    for var in vars_holding_edx:
      if var.reg is not None:
        ctx.r.pools[IReg].release_reg(var.reg, var)
        if var.reg not in ctx.r.pools[IReg]._acquired:
          ctx.r.pools[IReg].insert(0, var.reg)
      var.reg = IReg(2)
      ctx.r.pools[IReg].acquire_reg(IReg(2), var)
  if x.op is Ops.IDIV:
    result_reg = "rax"
  elif x.op is Ops.MOD:
    result_reg = "rdx"
  else: raise Exception(f"Invalid op {x.op}")
  if x.dtype == dtypes.uint32 or x.dtype == dtypes.uint64:
    op = "div"
    sign_extend = [f"xor {IReg(2).render(x.dtype.itemsize)}, {IReg(2).render(x.dtype.itemsize)}"]
  else:
    sign_extend = ["cdq" if x.dtype.itemsize == 4 else "cqo"]
    op = "idiv"
  ret = [
    f"mov rax, {_dividend.render64()}",
    *sign_extend,
    f"{op} {_divisor.render(x.dtype.itemsize)}",
    f"mov {_dst}, {result_reg}",
    *mov2,
  ]
  return ret

def arm_idiv(ctx, x):
  dividend, divisor = x.src
  _dividend, _divisor, _quotient = ctx.r.assign_multiple(
    [dividend, divisor, x], IReg)
  op = "udiv" if x.dtype == dtypes.uint32 else "sdiv"
  ret = [
    f"{op} {_quotient.render32()}, {_dividend.render32()}, {_divisor.render32()}"
  ]
  return ret

def max_int(ctx, x):
  src1, src2 = x.src
  _dst, _src1, _src2 = ctx.r.assign_multiple([x, src1, src2], IReg)
  size = x.dtype.itemsize
  if Arch.arm:
    return [f"cmp {_src1.render(size)}, {_src2.render(size)}",
            f"csel {_dst.render(size)}, {_src1.render(size)}, {_src2.render(size)}, gt"
            ]
  else:
    return [
      f"mov {_dst.render(size)}, {_src1.render(size)}",
      f"cmp {_src1.render(size)}, {_src2.render(size)}",
      f"cmovl {_dst.render(8)}, {_src2.render(8)}",
    ]

def max_uint(ctx, x):
  src1, src2 = x.src
  _dst, _src1, _src2 = ctx.r.assign_multiple([x, src1, src2], IReg)
  size = x.dtype.itemsize
  if Arch.arm:
    return [f"cmp {_src1.render(size)}, {_src2.render(size)}",
            f"csel {_dst.render(size)}, {_src1.render(size)}, {_src2.render(size)}, hi"
            ]
  else:
    return [
      f"mov {_dst.render(size)}, {_src1.render(size)}",
      f"cmp {_src1.render(size)}, {_src2.render(size)}",
      f"cmovb {_dst.render(8)}, {_src2.render(8)}",
    ]

def cast_bool_to_int(ctx, x, a):
  _x, _a = ctx.r.assign_multiple([x, a], IReg)
  temp_reg = ctx.r.alloc(IReg, excludes=[_x, _a])
  ctx.r.return_reg([temp_reg])
  if Arch.arm:
    return [
      f"cmp {_a.render32()}, xzr",
      f"cset {_x.render32()}, eq"
    ]
  else:
    return [
      f"xor {temp_reg}, {temp_reg}",
      f"xor {_x}, {_x}",
      f"cmp {_a.render32()}, {temp_reg.render32()}",
      f"sete {_x.render8()}",
    ]

def gated_load(ctx, x, bidx, alt, gate):
  reg_type = FReg if dtypes.is_float(x.dtype) else IReg
  _x, _alt = ctx.r.assign_multiple([x, alt], reg_type=reg_type)
  _gate, _bidx = ctx.r.assign_multiple([gate, bidx], reg_type=IReg, excludes=[_x, _alt])
  step = ctx.r.cur_step
  size = x.dtype.itemsize
  if Arch.x86:
    op = "mov" if reg_type is IReg else "movss" if size == 4 else "movsd"
    return [
      f"cmp {_gate}, 1",
      f"jne .ALT{step}",
      f"{op} {_x.render(size)}, [{_bidx}]",
      f"jmp .END{step}",
      f".ALT{step}:",
      f"{op} {_x.render(size)}, {_alt.render(size)}",
      f".END{step}:",
      ]
  else:
    mov_op = "mov" if reg_type is IReg else "fmov"
    mem_op = {1: "ldrb", 2: "ldrh", 4: "ldr", 8: "ldr"}.get(size)
    return [
      f"cmp {_gate}, #1",
      f"b.ne .ALT{step}",
      f"{mem_op} {_x.render(size)}, [{_bidx}]",
      f"b .END{step}",
      f".ALT{step}:",
      f"{mov_op} {_x.render(size)}, {_alt.render(size)}",
      f".END{step}:",
    ]

complex_rewrites = PatternMatcher([
  (UPat(Ops.LOAD, name="x", src=(
    UPat(Ops.INDEX, src=(UPat(), UPat(), UPat.var("gate"))).or_casted("bidx"),
    UPat.var("alt")), allow_any_len=True),
   gated_load),
  (UPat(Ops.MAX, name="x", dtype=dtypes.sints), max_int),
  (UPat(Ops.MAX, name="x", dtype=dtypes.uints), max_uint),
  (UPat(Ops.RECIP, name="x"), recip), 
  (UPat(Ops.WHERE, name="x"), _where),
  (UPat(GroupOp.ALU, name="x"), alu),
  (UPat(Ops.ASSIGN, name="x"), assign),
  (UPat(Ops.INDEX, name="x"), _index),
  (UPat(Ops.RANGE, name="x"), _range),
  (UPat(Ops.ENDRANGE, name="x"), endrange),
  (UPat(Ops.CONST, name="x", dtype=dtypes.floats), const),
  (UPat(Ops.CAST, name="x", dtype=dtypes.bool, src=(UPat(name="a"),)), to_bool),
  (UPat(Ops.CAST, name="x", dtype=dtypes.int32, src=(UPat(name="a", dtype=dtypes.bool),)),
    lambda ctx, x, a: [f"mov {ctx.r.assign_i64(x)}, {ctx.r.assign_i64(a)}"]),
])
x86_rewrite = PatternMatcher([
  (UPat((Ops.IDIV, Ops.MOD), name="x"), x86_idiv),
  (UPat((Ops.CMPNE, Ops.CMPLT), name="x", src=(UPat(name="a", dtype=dtypes.ints + (dtypes.bool,)),
                                  UPat(name="b"))),
   cmp_int_x86),
  (UPat((Ops.CMPLT), name="x", src=(UPat(name="a", dtype=dtypes.floats),
                                  UPat(name="b"))),
   cmplt_float_x86),
  (UPat((Ops.CMPNE), name="x", src=(UPat(name="a", dtype=dtypes.floats),
                                  UPat(name="b"))),
   cmpne_float_x86),
  (UPat(Ops.ADD, name="x", src=(UPat(Ops.DEFINE_REG, name="acc"), UPat(name="src"))), acc),
  (UPat(Ops.CONST, name="x", dtype=(dtypes.int32, dtypes.uint32)), lambda ctx, x: [f"mov {ctx.r.assign_i32(x)}, {x.arg:#x}"]),
  (UPat(Ops.CONST, name="x", dtype=(dtypes.int64, dtypes.uint64)), lambda ctx, x: [f"mov {ctx.r.assign_i64(x)}, {x.arg:#x}"]),
  (UPat(Ops.CONST, name="x", dtype=(dtypes.bool, dtypes.uint8)), lambda ctx, x: [f"mov {ctx.r.assign_i64(x)}, {int(x.arg)}"]),

  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=(dtypes.bool, dtypes.uint8)))),
      lambda ctx, x, addr, src: [f"mov [{ctx.r.assign_i64(addr)}], {ctx.r.assign_i8(src)}"]),

  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=(dtypes.int, dtypes.uint32)))),
      lambda ctx, x, addr, src: [f"mov [{ctx.r.assign_i64(addr)}], {ctx.r.assign_i32(src)}"]),

  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=(dtypes.int64, dtypes.uint64)))),
      lambda ctx, x, addr, src: [f"mov [{ctx.r.assign_i64(addr)}], {ctx.r.assign_i64(src)}"]),

  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.float32))),
      lambda ctx, x, addr, src: [f"movss [{ctx.r.assign_i64(addr)}], {ctx.r.assign_f32(src)}"]),

  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.float64))),
      lambda ctx, x, addr, src: [f"movsd [{ctx.r.assign_i64(addr)}], {ctx.r.assign_f64(src)}"]),

  (UPat(Ops.DEFINE_REG, name="x", dtype=(dtypes.bool, dtypes.uint8), src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"mov {ctx.r.assign_i32(x, reserve=False)}, {ctx.r.assign_i32(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.int32, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"mov {ctx.r.assign_i32(x, reserve=False)}, {ctx.r.assign_i32(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.int64, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"mov {ctx.r.assign_i64(x, reserve=False)}, {ctx.r.assign_i64(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.float32, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"movss {ctx.r.assign_f32(x, reserve=False)}, {ctx.r.assign_f32(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.float64, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"movsd {ctx.r.assign_f64(x, reserve=False)}, {ctx.r.assign_f64(src)}"]),

  (UPat(Ops.LOAD, name="x", dtype=(dtypes.bool, dtypes.uint8), src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"movzx {ctx.r.assign_i32(x)}, byte ptr [{ctx.r.assign_i64(src)}]"]),

  (UPat(Ops.LOAD, name="x", dtype=(dtypes.int32, dtypes.uint32), src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"mov {ctx.r.assign_i32(x)}, [{ctx.r.assign_i64(src)}]"]),

  (UPat(Ops.LOAD, name="x", dtype=dtypes.int64, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"mov {ctx.r.assign_i64(x)}, [{ctx.r.assign_i64(src)}]"]),

  (UPat(Ops.LOAD, name="x", dtype=dtypes.float32, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"movss {ctx.r.assign_f32(x)}, [{ctx.r.assign_i64(src)}]"]),

  (UPat(Ops.LOAD, name="x", dtype=dtypes.float64, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"movsd {ctx.r.assign_f32(x)}, [{ctx.r.assign_i64(src)}]"]),

  (UPat(Ops.BITCAST, name="x", dtype=(dtypes.int32, dtypes.uint32), src=(UPat(name="a", dtype=dtypes.float32),)),
    lambda ctx, x, a: [f"movd {ctx.r.assign_i32(x)}, {ctx.r.assign_f32(a)}"]),

  (UPat(Ops.BITCAST, name="x", dtype=dtypes.float32, src=(UPat(name="a", dtype=(dtypes.int32, dtypes.uint32)),)),
    lambda ctx, x, a: [f"movd {ctx.r.assign_f32(x)}, {ctx.r.assign_i32(a)}"]),

  (UPat(Ops.CAST, name="x", dtype=dtypes.ints, src=(UPat(name="a", dtype=dtypes.ints),)),
    lambda ctx, x, a: [f"mov {ctx.r.assign_i64(x)}, {ctx.r.assign_i64(a)}"]),

  (UPat(Ops.CAST, name="x", dtype=dtypes.ints, src=(UPat(name="a", dtype=dtypes.float32),)),
    lambda ctx, x, a: [f"cvttss2si {ctx.r.assign(x, reg_type=IReg).render(x.dtype.itemsize)}, {ctx.r.assign_f32(a)}"]),

  (UPat(Ops.CAST, name="x", dtype=dtypes.ints, src=(UPat(name="a", dtype=dtypes.float64),)),
    lambda ctx, x, a: [f"cvttsd2si {ctx.r.assign(x, reg_type=IReg).render(x.dtype.itemsize)}, {ctx.r.assign_f64(a)}"]),

  (UPat(Ops.CAST, name="x", dtype=dtypes.float32, src=(UPat(name="a", dtype=(dtypes.int32, dtypes.int64, dtypes.uint32, dtypes.uint64)),)),
    lambda ctx, x, a: [f"cvtsi2ss {ctx.r.assign_f32(x)}, {ctx.r.assign(a, reg_type=IReg).render(a.dtype.itemsize)}"]),

  (UPat(Ops.CAST, name="x", dtype=dtypes.float32, src=(UPat(name="a", dtype=dtypes.bool),)),
    lambda ctx, x, a: [f"cvtsi2ss {ctx.r.assign_f32(x)}, {ctx.r.assign_i32(a)}"]),

  (UPat(Ops.CAST, name="x", dtype=dtypes.float64, src=(UPat(name="a", dtype=dtypes.int64),)),
    lambda ctx, x, a: [f"cvtsi2sd {ctx.r.assign_f64(x)}, {ctx.r.assign_i32(a)}"]),

]) + complex_rewrites

arm_rewrite = PatternMatcher([
  (UPat(Ops.IDIV, name="x"), arm_idiv),
  (UPat((Ops.CMPLT, Ops.CMPNE), name="x", src=(UPat(name="a"),
                                  UPat(name="b"))),
   cmp_arm),
  (UPat(Ops.ADD, name="x", src=(UPat(Ops.DEFINE_REG, name="acc"), UPat(name="src"))), acc),
  (UPat(Ops.CONST, name="x", dtype=(dtypes.int32)), lambda ctx, x: [f"mov {ctx.r.assign_i32(x)}, #{x.arg}"]),
  (UPat(Ops.CONST, name="x", dtype=(dtypes.int64, dtypes.uint64, dtypes.uint32)), const),
  (UPat(Ops.CONST, name="x", dtype=(dtypes.bool, dtypes.uint8)), lambda ctx, x: [f"mov {ctx.r.assign_i64(x)}, {int(x.arg)}"]),

  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=(dtypes.bool, dtypes.uint8)))),
      lambda ctx, x, addr, src: [f"strb {ctx.r.assign_i8(src)}, [{ctx.r.assign_i64(addr)}]"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=(dtypes.int, dtypes.uint32)))),
      lambda ctx, x, addr, src: [f"str {ctx.r.assign_i32(src)}, [{ctx.r.assign_i64(addr)}]"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=(dtypes.int64, dtypes.uint64)))),
      lambda ctx, x, addr, src: [f"str {ctx.r.assign_i64(src)}, [{ctx.r.assign_i64(addr)}]"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.float32))),
      lambda ctx, x, addr, src: [f"str {ctx.r.assign_f32(src)}, [{ctx.r.assign_i64(addr)}]"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.float64))),
      lambda ctx, x, addr, src: [f"str {ctx.r.assign_f64(src)}, [{ctx.r.assign_i64(addr)}]"]),

  (UPat(Ops.DEFINE_REG, name="x", dtype=(dtypes.bool, dtypes.uint8), src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"mov {ctx.r.assign_i32(x, reserve=False)}, {ctx.r.assign_i32(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.int32, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"mov {ctx.r.assign_i32(x, reserve=False)}, {ctx.r.assign_i32(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.int64, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"mov {ctx.r.assign_i64(x, reserve=False)}, {ctx.r.assign_i64(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.float32, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"fmov {ctx.r.assign_f32(x, reserve=False)}, {ctx.r.assign_f32(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.float64, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"fmov {ctx.r.assign_f64(x, reserve=False)}, {ctx.r.assign_f64(src)}"]),

  (UPat(Ops.LOAD, name="x", dtype=(dtypes.bool, dtypes.uint8), src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"ldrb {ctx.r.assign_i32(x)}, [{ctx.r.assign_i64(src)}]"]),
  (UPat(Ops.LOAD, name="x", dtype=(dtypes.int32, dtypes.uint32), src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"ldr {ctx.r.assign_i32(x)}, [{ctx.r.assign_i64(src)}]"]),
  (UPat(Ops.LOAD, name="x", dtype=(dtypes.int64, dtypes.uint64), src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"ldr {ctx.r.assign_i64(x)}, [{ctx.r.assign_i64(src)}]"]),
  (UPat(Ops.LOAD, name="x", dtype=dtypes.float32, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"ldr {ctx.r.assign_f32(x)}, [{ctx.r.assign_i64(src)}]"]),
  (UPat(Ops.LOAD, name="x", dtype=dtypes.float64, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"ldr {ctx.r.assign_f64(x)}, [{ctx.r.assign_i64(src)}]"]),

  (UPat(Ops.BITCAST, name="x", dtype=(dtypes.int32, dtypes.uint32), src=(UPat(name="a", dtype=dtypes.float32),)),
    lambda ctx, x, a: [f"fmov {ctx.r.assign_i32(x)}, {ctx.r.assign_f32(a)}"]),

  (UPat(Ops.BITCAST, name="x", dtype=dtypes.float32, src=(UPat(name="a", dtype=(dtypes.int32, dtypes.uint32)),)),
    lambda ctx, x, a: [f"fmov {ctx.r.assign_f32(x)}, {ctx.r.assign_i32(a)}"]),

  (UPat(Ops.CAST, name="x", dtype=dtypes.ints, src=(UPat(name="a", dtype=dtypes.ints),)),
    lambda ctx, x, a: [f"mov {ctx.r.assign_i64(x)}, {ctx.r.assign_i64(a)}"]),

  (UPat(Ops.CAST, name="x", dtype=dtypes.ints, src=(UPat(name="a", dtype=(dtypes.float32, dtypes.float64)),)),
    lambda ctx, x, a: [f"fcvtzs {ctx.r.assign(x, reg_type=IReg).render(x.dtype.itemsize)}, {ctx.r.assign_f32(a)}"]),

  (UPat(Ops.CAST, name="x", dtype=(dtypes.float32, dtypes.float64),
        src=(UPat(name="a", dtype=(dtypes.int32, dtypes.int64, dtypes.uint32, dtypes.uint64, dtypes.bool),))),
    lambda ctx, x, a:
   [f"scvtf {ctx.r.assign(x, reg_type=FReg).render(x.dtype.itemsize)}, {ctx.r.assign(a, reg_type=IReg).render(a.dtype.itemsize)}"]),

]) + complex_rewrites

def fix_uint(ctx, x: UOp):
  max_val = 0xFFFFFFFF
  effective_value = x.arg & max_val
  if x.arg >= max_val: #4294967295:
    return x.replace(arg=effective_value)
  else:
    return x
def fix_uint8(ctx, x: UOp):
  return x.replace(arg=x.arg % (0xff+1))
extra_matcher = PatternMatcher([
  (UPat(Ops.CONST, dtype=dtypes.uint, name="x"), fix_uint),
  (UPat(Ops.CONST, dtype=dtypes.uint8, name="x"), fix_uint8),
])
if Arch.arm:
  extra_matcher += PatternMatcher([
    (UPat(Ops.MOD, name="x", src=(UPat(name="src1"), UPat(name="src2"))),
      lambda ctx, x, src1, src2:
      UOp(Ops.SUB, dtype=x.dtype, src=(
        src1,
        UOp(Ops.MUL, dtype=x.dtype, src=(
          UOp(Ops.IDIV, dtype=x.dtype, src=(
            src1,
            src2,
          )),
          src2,
       ),
     ))),
     )
  ])

class AsmRenderer(Renderer):
  supports_float4 = False
  has_local = False
  has_shared = False
  global_max = None
  extra_matcher = extra_matcher
  code_for_op = {
    Ops.SQRT: lambda:None
  }

  def __init__(self) -> None:
    super().__init__()
    arch = platform.machine()
    self.arm = arch == "aarch64"
    self.x86 = arch == "x86_64"
    assert self.arm ^ self.x86

  def render(self, uops:List[UOp]) -> str:
    gen_regs = [f"x{i}" for i in range(0, 31)]
    float_regs = [f"D{i}" for i in range(0,32)]
    self.all_regs = gen_regs + float_regs
    self.r = Allocator(num_ireg=16, num_freg=16)
    r = self.r
    mem: list[tuple[str, str]] = [] # ("constant_1", ".float 32.0")
    self.mem = mem
    stack_size: int = 16
    arg_stack_offset: int = 16
    kernel: List[str] = []
    self.uops = uops
    last_use: Dict[UOp, int] = {var:i for i,u in enumerate(uops) for var in (v for v in (u,) + u.src if v.dtype != dtypes.void)}
    if DEBUG >= 6: print(uops[-1])

    name = "test"
    uop_order = {} 
    var_intervals: dict[UOp, Variable] = OrderedDict()
    for i, u in enumerate(uops):
      var = Variable(u, i, -1)
      var_intervals[u] = var
    for i, u in enumerate(uops):
      for src in u.src:
        if src.dtype is not dtypes.void:
          prev = var_intervals[src].end
          var_intervals[src].end = max(prev, i)

    for i, u in enumerate(uops):
      for src in u.src:
        if src.op is Ops.INDEX and len(src.src) > 2:
          gate = src.src[2]
          var_intervals[gate].end = max(var_intervals[gate].end, var_intervals[src].end)

    for v in var_intervals.values():
      if v.end == -1: v.end = len(uops)
    self.r.uops = var_intervals
    if DEBUG.value >= 6:
      for i, u in enumerate(uops):
        v = r.uops[u]
        print(i, v, oneline_uop(u))
    if Arch.x86:
      r.blocked.append(IReg(5))
    
    r.bookkeeping()
    for i,u in enumerate(uops):
      self.r.cur_step = i
      if DEBUG.value >= 6:
        print("=================================")
        print(i, r.uops[u], u)
        print("src intervals:")
        for src in u.src:
          print(self.r.uops[src])
      r.free_expired(i)
      if u.op is Ops.DEFINE_GLOBAL:
        var = r.uops[u]
        if Arch.arm:
          reg = IReg(u.arg)
        else:
          reg = IReg(x86_params[u.arg])
        pool = r.pools[IReg]
        pool.pop(pool.index(reg))
        pool.acquire_reg(reg, var)
        var.reg = reg
        r.move_var_to_stack(var)
        kernel.extend(r.flush_kernel())
          
      elif u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
      else:
        rewriter = arm_rewrite if Arch.arm else x86_rewrite
        if (l:=rewriter.rewrite(u, ctx=self)) is None:
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        l = cast(list[str], l)
        l = [*r.flush_kernel(), *l, ""]
        if DEBUG.value >= 6:
          uop_str = [f".uop_{i}:"] + ["//"+_u for _u in str(u).split("\n")][:]
          l = [*uop_str, *l]
          print("\n".join(kernel)[-100:])
          print("\033[32m", "\n".join(l), "\033[0m", sep="")
        kernel.extend(l)
      r.bookkeeping()

    if Arch.x86:
      stack_alloc = [f"sub rsp, {r.stack_size}"]
    else:
      stack_alloc = []
      stack = r.stack_size
      while stack > 4096:
        stack_alloc.append(f"sub sp, sp, #4096")
        stack -= 4096
      stack_alloc.append(f"sub sp, sp, #{stack}")

    prologue = [
      "stp x29, x30, [sp, #-16]!",
      "mov x29, sp",
      "mov x30, sp",
      "sub x30, x30, #255",
      *stack_alloc
    ] if self.arm else [
      "push rbp",
      "mov rbp, rsp",
      *stack_alloc,
    ]
    epilogue = [
      f"mov sp, x29;",
      f"ldp x29, x30, [sp], #16;",
      f"ret",
    ] if self.arm else [
      "mov rsp, rbp",
      "pop rbp",
      "ret",
    ]
    mem_data = []
    for a,b in mem:
      if b.startswith(".quad"):
        mem_data.append(f".align 3")
      mem_data.append(f"{a}: {b}")
    data_section = [
      ".section .data",
      ".p2align 3",
      *mem_data
    ]
    kernel = [
      *prologue,
      *kernel,
      *epilogue,
      "",
      *data_section
    ]
    _kernel: str = "\n".join(kernel)
    ret = f"""
.text
{'.intel_syntax noprefix' if self.x86 else ''}
.global {name}
{name}:
{_kernel}
    """
    if folder:=os.environ.get("SAVE_ASM"):
      with open(f"../tg-dev/{folder}/{name}.s", "wt") as f: f.write(ret)
    return ret

#TESTS

class Tests(unittest.TestCase):
  def test_to_hex(self):
    assert float32_to_hex(20.0) == "0x41a00000"
    assert float32_to_hex(49.193) == "0x4244c5a2"

class TestAllocatorExpire(unittest.TestCase):
  def setUp(self):
    self.a = Allocator(16, 0)
    uop1 = UOp(Ops.CONST, arg=1)
    uop2 = UOp(Ops.CONST, arg=2)
    self.uop1, self.uop2 = uop1, uop2
    self.a.uops[uop1] = Variable(uop1, 0, 2)
    self.a.uops[uop2] = Variable(uop2, 0, 10)
    self.a.assign(uop1, IReg, reserve=True)
    self.a.assign(uop2, IReg, reserve=True)
    assert len(self.a.uops) == 2
    assert len(self.a.reserved) == 2
  def tearDown(self): del self.a, self.uop1, self.uop2

  def test_expired_uop_none(self):
    assert len(self.a.pools[IReg]) == 14
    self.a.free_expired(2)
    assert len(self.a.uops) == 2 and len(self.a.reserved) == 2
    assert len(self.a.pools[IReg]) == 14

  def test_expired_uop_one(self):
    self.a.free_expired(3)
    assert len(self.a.uops) == 1 and len(self.a.reserved) == 1
    assert len(self.a.pools[IReg]) == 15

  def test_expired_uop_non(self):
    self.a.free_expired(11)
    assert len(self.a.uops) == 0 and len(self.a.reserved) == 0
    assert len(self.a.pools[IReg]) == 16

  def test_expire_reg_none(self):
    self.a.uops[self.uop1].reg = None
    self.a.free_expired(3)
    assert len(self.a.pools[IReg]) == 14

class TestAllocatorShare(unittest.TestCase):
  def setUp(self):
    self.a = Allocator(16, 0)
    uop1 = UOp(Ops.CONST, arg=1)
    uop2 = UOp(Ops.CONST, arg=2)
    self.uop1, self.uop2 = uop1, uop2
    self.var1, self.var2 = Variable(uop1, 0, 2), Variable(uop2, 0, 10)
    self.a.uops[uop1] = self.var1
    self.a.uops[uop2] = self.var2
    self.a.assign(uop1, IReg, reserve=True)
    self.a.share(uop2, uop1)
    
  def test_share_regs(self):
    assert self.var1.reg == self.var2.reg

  def test_expire_one(self):
    self.a.free_expired(5)
    assert self.a.uops.get(self.uop1) is None
    assert self.var2.reg is not None
    assert IReg(0) not in self.a.pools[IReg]

  def test_expire_both(self):
    self.a.free_expired(11)
    assert IReg(0) in self.a.pools[IReg]

class TestAllocatorSpill(unittest.TestCase):
  def setUp(self):
    self.a = Allocator(2, 0)
    uop1 = UOp(Ops.CONST, arg=1)
    uop2 = UOp(Ops.CONST, arg=2)
    uop3 = UOp(Ops.CONST, arg=3)
    self.uop1, self.uop2, self.uop3 = uop1, uop2, uop3
    self.a.uops[uop1] = Variable(uop1, 0, 9)
    self.a.uops[uop2] = Variable(uop2, 0, 10)
    self.a.uops[uop3] = Variable(uop3, 0, 11)
    self.a.assign(uop1, IReg)
    self.a.assign(uop2, IReg)
  def tearDown(self): del self.uop1, self.uop2, self.uop3, self.a

  def test_spill(self):
    reg = self.a.assign(self.uop3, IReg)
    kernel = self.a.flush_kernel()
    assert reg == IReg(1)
    assert self.a.uops[self.uop1].reg   is not None
    assert self.a.uops[self.uop1].stack is None

    assert self.a.uops[self.uop2].reg   is None
    assert self.a.uops[self.uop2].stack is not None

    assert self.a.uops[self.uop3].reg   is not None
    assert self.a.uops[self.uop3].stack is None
    assert len(kernel) == 1# and kernel[0].startswith("str")

  def test_spill_with_stack_load(self):
    self.a.uops[self.uop2].stack = 0
    self.a.uops[self.uop3].stack = 8
    self.a.stack_size = 16
    reg = self.a.assign(self.uop3, IReg)
    kernel = self.a.flush_kernel()
    assert self.a.uops[self.uop2].stack == 0
    assert self.a.uops[self.uop3].stack == 8
    assert len(kernel) == 2# and kernel[1].startswith("ldr")
    assert self.a.stack_size == 16

  def test_spill_with_stack_str(self):
    assert self.a.stack_size == 0
    self.a.assign(self.uop3, IReg)
    assert self.a.stack_size == 8
    assert self.a.uops[self.uop2].stack == 8

class TestAllocatorSpillFloat(unittest.TestCase):
  def setUp(self):
    self.a = Allocator(num_ireg=2, num_freg=2)
    uop1 = UOp(Ops.CONST, dtype=dtypes.float32, arg=1)
    uop2 = UOp(Ops.CONST, dtype=dtypes.float32, arg=2)
    uop3 = UOp(Ops.CONST, dtype=dtypes.float32, arg=3)
    self.uop1, self.uop2, self.uop3 = uop1, uop2, uop3
    self.a.uops[uop1] = Variable(uop1, 0, 9)
    self.a.uops[uop2] = Variable(uop2, 0, 10)
    self.a.uops[uop3] = Variable(uop3, 0, 11)
    self.a.assign(uop1, reg_type=FReg)
    self.a.assign(uop2, reg_type=FReg)
  def tearDown(self): del self.uop1, self.uop2, self.uop3, self.a

  def test_spill(self):
    reg = self.a.assign(self.uop3, reg_type=FReg)
    kernel = self.a.flush_kernel()
    assert reg == FReg(1)
    assert self.a.uops[self.uop1].reg   is not None
    assert self.a.uops[self.uop1].stack is None

    assert self.a.uops[self.uop2].reg   is None
    assert self.a.uops[self.uop2].stack is not None

    assert self.a.uops[self.uop3].reg   is not None
    assert self.a.uops[self.uop3].stack is None
    assert len(kernel) == 1# and kernel[0].startswith("str")

  def test_spill_with_stack_load(self):
    self.a.uops[self.uop2].stack = 0
    self.a.uops[self.uop3].stack = 8
    self.a.stack_size = 16
    reg = self.a.assign(self.uop3, reg_type=FReg)
    kernel = self.a.flush_kernel()
    assert self.a.uops[self.uop2].stack == 0
    assert self.a.uops[self.uop3].stack == 8
    assert len(kernel) == 2# and kernel[1].startswith("ldr")
    assert self.a.stack_size == 16

  def test_spill_with_stack_str(self):
    assert self.a.stack_size == 0
    self.a.assign(self.uop3, reg_type=FReg)
    assert self.a.stack_size == 16
    assert self.a.uops[self.uop2].stack == 16

class TestAllocatorStackAll(unittest.TestCase):
  """
  Ops.RANGE and Ops.DEFINE_REG's Variable could change, the change need to 
  be saved in stack
  """
  def setUp(self):
    self.a = Allocator(16, 0)
    uop1 = UOp(Ops.RANGE)
    self.uop1 = uop1
    var = Variable(uop1, 0, 10)
    var.stack = 4
    self.a.uops[uop1] = var
    self.a.assign(uop1, IReg)
    self.a.flush_kernel()
  def tearDown(self): del self.a

  def test_update_stack(self):
    stack_all(self.a)
    kernel = self.a.flush_kernel()
    assert len(kernel) == 1

class TestAllocatorExcludeReserve(unittest.TestCase):
  def _setup(self):
    assert self.a
    self.uop1 = UOp(Ops.CONST, arg=1)
    self.var1 = Variable(self.uop1, 0, 10)
    self.uop2 = UOp(Ops.CONST, arg=2)
    self.var2 = Variable(self.uop2, 0, 11)
    self.uop3 = UOp(Ops.CONST, arg=3)
    self.var3 = Variable(self.uop3, 0, 12)
    self.uop4 = UOp(Ops.CONST, arg=4)
    self.var4 = Variable(self.uop4, 0, 12)
    self.a.uops[self.uop1] = self.var1
    self.a.uops[self.uop2] = self.var2
    self.a.uops[self.uop3] = self.var3
    self.a.uops[self.uop4] = self.var4
  def test_exclude(self):
    self.a = Allocator(2, 0)
    self._setup()
    reg1 = self.a.assign(self.uop1, IReg)
    reg2 = self.a.assign(self.uop2, IReg)
    reg3 = self.a.assign(self.uop3, IReg, excludes=[reg2])
    assert self.var1.reg is None and self.var1.stack == 8
    assert self.var2.reg == IReg(1)
    assert self.var3.reg == IReg(0)
  def test_exclude_not_enough_reg(self):
    self.a = Allocator(1, 0)
    self._setup()
    self.a.assign(self.uop2, IReg)
    self.a.assign(self.uop3, IReg)
  def test_exclude_not_enough_reg_raise(self):
    self.a = Allocator(1, 0)
    self._setup()
    reg2 = self.a.assign(self.uop2, IReg)
    with self.assertRaises(Exception):
      self.a.assign(self.uop3, IReg, excludes=[reg2])
  def test_reserve(self):
    self.a = Allocator(2, 0)
    self._setup()
    self.a.assign(self.uop1, IReg)
    self.a.assign(self.uop2, IReg, reserve=True)
    self.a.assign(self.uop3, IReg)
    assert self.var3.reg == IReg(0)
  def test_reserve_not_enough_reg(self):
    self.a = Allocator(2, 0)
    self._setup()
    self.a.assign(self.uop1, IReg, reserve=True)
    self.a.assign(self.uop2, IReg, reserve=True)
    with self.assertRaises(Exception):
      self.a.assign(self.uop3, IReg)
  def test_reserve_release(self):
    self.a = Allocator(2, 0)
    self._setup()
    self.a.assign(self.uop1, IReg, reserve=True)
    reg2 = self.a.assign(self.uop2, IReg, reserve=True)
    self.a.release(reg2)
    self.a.assign(self.uop3, IReg)
  def test_reserve_not_enough_reg_pair(self):
    self.a = Allocator(3, 0)
    self._setup()
    self.a.assign(self.uop1, IReg, reserve=True)
    self.a.assign(self.uop2, IReg, reserve=True)
    with self.assertRaises(Exception):
      reg3 = self.a.assign(self.uop3, IReg)
      self.a.assign(self.uop4, IReg, excludes=[reg3])

class TestAllocatorAluShareReg(unittest.TestCase):
  def test_add_no_share(self):
    self.r = Allocator(3, 3)
    self.uop1 = UOp(Ops.CONST, dtype=dtypes.float, arg=1)
    self.var1 = Variable(self.uop1, 0, 4)
    self.uop2 = UOp(Ops.CONST, dtype=dtypes.float, arg=2)
    self.var2 = Variable(self.uop2, 1, 4)
    self.uop3 = UOp(Ops.ADD, dtype=dtypes.float, src=(self.uop1, self.uop2), arg=3)
    self.var3 = Variable(self.uop3, 2, 4)
    self.r.uops[self.uop1] = self.var1
    self.r.uops[self.uop2] = self.var2
    self.r.uops[self.uop3] = self.var3
    alu = [self.uop1, self.uop2, self.uop3]
    self.r.assign_f32(self.uop1)
    self.r.assign_f32(self.uop2)
    rewriter = arm_rewrite if Arch.arm else x86_rewrite
    l = rewriter.rewrite(self.uop3, self)
    print(l)
    assert self.r.uops[self.uop3].reg != self.r.uops[self.uop1].reg
    #assert len(cast(list[str], l)) == 2

  def test_add_share(self):
    self.r = Allocator(3, 3)
    self.uop1 = UOp(Ops.CONST, dtype=dtypes.float, arg=1)
    self.var1 = Variable(self.uop1, 0, 2)
    self.uop2 = UOp(Ops.CONST, dtype=dtypes.float, arg=2)
    self.var2 = Variable(self.uop2, 1, 2)
    self.uop3 = UOp(Ops.ADD, dtype=dtypes.float, src=(self.uop1, self.uop2), arg=3)
    self.var3 = Variable(self.uop3, 2, 4)
    self.r.uops[self.uop1] = self.var1
    self.r.uops[self.uop2] = self.var2
    self.r.uops[self.uop3] = self.var3
    alu = [self.uop1, self.uop2, self.uop3]
    self.r.assign_f32(self.uop1)
    self.r.assign_f32(self.uop2)
    rewriter = arm_rewrite if Arch.arm else x86_rewrite
    self.r.cur_step = 2
    l = rewriter.rewrite(self.uop3, self)

    assert self.r.uops[self.uop3].reg == self.r.uops[self.uop1].reg
    assert len(cast(list[str], l)) == 1



class TestReg(unittest.TestCase):
  def test_singleton(self):
    assert IReg(3) is IReg(3)
    assert IReg(3) == IReg(3)
    assert IReg(3) != IReg(4)
    assert IReg(3) is not FReg(3)
    assert FReg(3) is not FReg(4)
    assert FReg(3) is FReg(3)

class TestAluOpsStr(unittest.TestCase):
  def test_alu_ops(self):
    assert AluOps.get((Ops.ADD, ArchType.X86, IReg)) == "add"
    assert AluOps.get((Ops.ADD, ArchType.X86, IReg, 32)) == "add"
    assert AluOps.get((Ops.MUL, ArchType.ARM, FReg)) == "fmul"
    assert AluOps.get((Ops.MUL, ArchType.ARM, FReg, 64)) == "fmul"
    assert AluOps.get((Ops.ASSIGN, ArchType.X86, FReg, 64)) == "movsd"
    with self.assertRaises(Exception):
      AluOps.get((ArchType.X86, Ops.ADD))
    with self.assertRaises(Exception):
      AluOps.get((ArchType.ARM, FReg, Ops.MUL, 64))
    with self.assertRaises(Exception):
      AluOps.get((ArchType.X86, FReg))

def arch_decorator(arch: ArchType):
  def decorator(func):
    def wrapper(self, *args, **kwargs):
      original_arch = Arch.arch
      Arch.arch = arch
      try:
        func(self, *args, **kwargs)
      finally:
        Arch.arch = original_arch
    return wrapper
  return decorator
arm = arch_decorator(ArchType.ARM)
x86 = arch_decorator(ArchType.X86)

def linearize(u: UOp):
  visited, queue, result = set(), [u], []
  while queue:
    node = queue.pop(0)
    if node in visited: continue
    visited.add(node)
    result.append(node)
    for child in node.src:
      if child not in visited:
        queue.append(child)
  result.reverse()
  return result

class TestRender(unittest.TestCase):
  def setUp(self):
    self.r = Allocator(16, 16)
    self.mem = []

  def render(self, uop: UOp, rendered: list[str]):
    uops = linearize(uop)
    for u in uops: self.r.uops[u] = Variable(u, 0, 100)
    k = self.r.flush_kernel()
    rewriter = arm_rewrite if Arch.arm else x86_rewrite
    l = rewriter.rewrite(uop, ctx=self)
    l = cast(list[str], l)
    assert l is not None
    assert [*k, *l] == rendered

  def _const(self, dtype: DType, value: Union[int, float], rendered: list[str]):
    a = UOp(Ops.CONST, dtype, arg=value)
    self.render(a, rendered)

  @x86
  def test_x86_const_int32(self): self._const(dtypes.int, 1, ["mov eax, 0x1"])
  @arm
  def test_arm_const_int32(self): self._const(dtypes.int, 1, ["mov w0, #1"])
  @x86
  def test_x86_const_int64(self): self._const(dtypes.int64, 1, ["mov rax, 0x1"])

  @unittest.skip("OUtdated")
  @arm
  def test_arm_const_int64(self): self._const(dtypes.int64, 1, ["mov x0, #1"])

  @x86
  def test_x86_const_float_scalar_32(self): self._const(dtypes.float, 1.0,
    ["movss xmm0, [rip+const_0]"])
  @arm
  def test_arm_const_float_scalar_32(self): self._const(dtypes.float, 1.0,
    ["adrp x0, const_0", "ldr s0, [x0, #:lo12:const_0]"])

  def render_store(self, dtype: DType, rendered: list[str]):
    a = UOp(Ops.STORE, dtypes.void, arg=None, src=(
      UOp(Ops.INDEX, dtypes.int.ptr(16), arg=None, src=(
	UOp(Ops.DEFINE_GLOBAL, dtype.ptr(16), arg=0, src=()),
	UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),
      UOp(Ops.CONST, dtype, arg=20, src=()),))
    self.render(a, rendered)

  @x86
  def test_x86_store_int32(self):
    self.render_store(dtypes.int32, ["mov [rax], ecx"])
  @x86
  def test_x86_store_int64(self):
    self.render_store(dtypes.int64, ["mov [rax], rcx"])
  @x86
  def test_x86_store_float32(self):
    self.render_store(dtypes.float32, ["movss [rax], xmm0"])
  @arm
  def test_arm_store_int32(self):
    self.render_store(dtypes.int32, ["str w0, [x1]"])
  @arm
  def test_arm_store_int64(self):
    self.render_store(dtypes.int64, ["str x0, [x1]"])
  @arm
  def test_arm_store_float64(self):
    self.render_store(dtypes.float64, ["str d0, [x0]"])

  def _load(self, dtype: DType, rendered: list[str]):
    a = UOp(Ops.LOAD, dtype, arg=None, src=(
      UOp(Ops.INDEX, dtype.ptr(12), arg=None, src=(
	UOp(Ops.DEFINE_GLOBAL, dtype.ptr(12), arg=1, src=()),
	UOp(Ops.CONST, dtype, arg=None, src=()),)),))
    self.render(a, rendered)

  @x86
  def test_x86_load_int32(self):
    self._load(dtypes.int32, ["mov eax, [rcx]"])
  @x86
  def test_x86_load_int64(self):
    self._load(dtypes.int64, ["mov rax, [rcx]"])
  @x86
  def test_x86_load_float32(self):
    self._load(dtypes.float32, ["movss xmm0, [rax]"])
  @x86
  def test_x86_load_float64(self):
    self._load(dtypes.float64, ["movsd xmm0, [rax]"])
  @arm
  def test_arm_load_int32(self):
    self._load(dtypes.int32, ["ldr w0, [x1]"])
  @arm
  def test_arm_load_int64(self):
    self._load(dtypes.int64, ["ldr x0, [x1]"])
  @arm
  def test_arm_load_float32(self):
    self._load(dtypes.float32, ["ldr s0, [x0]"])
  @arm
  def test_arm_load_float64(self):
    self._load(dtypes.float64, ["ldr d0, [x0]"])

  def _define_acc(self, dtype: DType, rendered: list[str]):
    a = UOp(Ops.DEFINE_REG, dtype, arg=(0,), src=(
      UOp(Ops.CONST, dtype, arg=0, src=()),
      UOp(Ops.RANGE, dtype, arg=2, src=(
	UOp(Ops.CONST, dtype, arg=4, src=()),)),))
    self.render(a, rendered)
  @x86
  def test_x86_define_acc_int32(self):
    self._define_acc(dtypes.int32, ["mov eax, ecx"])
  @x86
  def test_x86_define_acc_int64(self):
    self._define_acc(dtypes.int64, ["mov rax, rcx"])
  @x86
  def test_x86_define_acc_float32(self):
    self._define_acc(dtypes.float32, ["movss xmm0, xmm1"])
  @x86
  def test_x86_define_acc_float64(self):
    self._define_acc(dtypes.float64, ["movsd xmm0, xmm1"])
  @arm
  def test_arm_define_acc_int32(self):
    self._define_acc(dtypes.int32, ["mov w0, w1"])
  @arm
  def test_arm_define_acc_int64(self):
    self._define_acc(dtypes.int64, ["mov x0, x1"])
  @arm
  def test_arm_define_acc_float32(self):
    self._define_acc(dtypes.float32, ["fmov s0, s1"])
  @arm
  def test_arm_define_acc_float64(self):
    self._define_acc(dtypes.float64, ["fmov d0, d1"])

  def _assign(self, dtype: DType, rendered: list[str]):
    a = UOp(Ops.ASSIGN, dtype, arg=None, src=(
      UOp(Ops.DEFINE_REG, dtype, arg=(0,), src=(
        UOp(Ops.CONST, dtype, arg=0, src=()),
        UOp(Ops.RANGE, dtype, arg=2, src=(
          UOp(Ops.CONST, dtype, arg=4, src=()),
        ))
      )),
      UOp(Ops.CONST, dtype, arg=123, src=()),))
    self.render(a, rendered)

  @unittest.skip("Assign impelmetnation changed")
  @x86
  def test_x86_assign_int32(self):
    self._assign(dtypes.int32, [
      "mov rax, rcx",
    ])
  
  @x86
  @unittest.skip("Assign impelmetnation changed")
  def test_x86_assign_int64(self):
    self._assign(dtypes.int64, [
      "mov rax, rcx",
    ])
  
  @x86
  @unittest.skip("Assign impelmetnation changed")
  def test_x86_assign_float32(self):
    self._assign(dtypes.float32, [
      "movss xmm0, xmm1",
    ])
  
  @x86
  @unittest.skip("Assign impelmetnation changed")
  def test_x86_assign_float64(self):
    self._assign(dtypes.float64, [
      "movsd xmm0, xmm1",
    ])
  
  @arm
  @unittest.skip("Assign impelmetnation changed")
  def test_arm_assign_int32(self):
    self._assign(dtypes.int32, [
      "mov x0, x1",
    ])
  
  @arm
  @unittest.skip("Assign impelmetnation changed")
  def test_arm_assign_int64(self):
    self._assign(dtypes.int64, [
      "mov x0, x1",
    ])
  
  @arm
  @unittest.skip("Assign impelmetnation changed")
  def test_arm_assign_float32(self):
    self._assign(dtypes.float32, [
      "fmov d0, d1",
    ])
  
  @arm
  @unittest.skip("Assign impelmetnation changed")
  def test_arm_assign_float64(self):
    self._assign(dtypes.float64, [
      "fmov d0, d1",
    ])

  @x86
  def test_x86_range(self):
    a = UOp(Ops.RANGE, arg=0, src=(
      UOp(Ops.CONST, arg=4),
    ))
    self.render(a, ["mov rax, #0", ".LOOP_0:"])
    b = UOp(Ops.ENDRANGE, src=(
      a,
    ))
    self.render(b, ["inc rcx", "cmp rcx, 4", "jl .LOOP_0"])
  @arm
  def test_arm_range(self):
    a = UOp(Ops.RANGE, arg=0, src=(
      UOp(Ops.CONST, arg=4),
    ))
    self.render(a, ["mov x0, #0", ".LOOP_0:"])
    b = UOp(Ops.ENDRANGE, src=(
      a,
    ))
    self.render(b, ["add x1, x1, #1", "cmp x1, #4", "b.lt .LOOP_0"])
  @x86
  def test_x86_index(self):
    a = UOp(Ops.INDEX, dtypes.int.ptr(16), arg=None, src=(
      x2:=UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), arg=0, src=()),
      x3:=UOp(Ops.CONST, dtypes.int, arg=None, src=()),))
    self.render(a, ["lea rdx, [rax + rcx * 4]"])
  @arm
  def test_arm_index(self):
    a = UOp(Ops.INDEX, dtypes.int.ptr(16), arg=None, src=(
      x2:=UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), arg=0, src=()),
      x3:=UOp(Ops.CONST, dtypes.int, arg=None, src=()),))
    self.render(a, ["add x2, x0, x1, lsl #2"]);


class TestAllocatorAssignReg(unittest.TestCase):
  def _test_assign_available(self, reg: RegBase):
    self.a = Allocator(3, 3)
    self.uop1 = UOp(Ops.CONST, arg=1)
    self.var1 = Variable(self.uop1, 0, 10)
    self.a.uops[self.uop1] = self.var1
    self.a.assign_reg(reg, self.uop1)
    ret = self.a.flush_kernel()
    assert len(ret) == 0
    assert self.var1.reg == reg
    assert reg not in self.a.pools[type(reg)]

  def test_assign_ireg(self): self._test_assign_available(IReg(0))
  def test_assign_freg(self): self._test_assign_available(FReg(0))

  def _test_assign_occupied(self, dtype: DType, reg: RegBase, reg_old: RegBase, k: list[str]):
    self.a = Allocator(3, 3)
    self.uop1 = UOp(Ops.CONST, arg=1, dtype=dtype)
    self.var1 = Variable(self.uop1, 0, 10)
    self.uop2 = UOp(Ops.CONST, arg=2, dtype=dtype)
    self.var2 = Variable(self.uop2, 0, 10)
    self.a.uops[self.uop1] = self.var1
    self.a.uops[self.uop2] = self.var2
    self.var1.reg = reg_old
    self.a.assign_reg(reg, self.uop1)
    ret = self.a.flush_kernel()
    assert ret == k
    assert self.var1.reg == reg

  def test_assign_occupied_ireg(self):
    ret = [f"mov rax, rcx"] if Arch.x86 else ["mov x0, x1"]
    self._test_assign_occupied(dtypes.int, IReg(0), IReg(1), ret)

  def test_assign_occupied_freg(self):
    ret = [f"movq xmm0, xmm1"] if Arch.x86 else [f"fmov d0, d1"]
    self._test_assign_occupied(dtypes.float, FReg(0), FReg(1), ret)

  def _unassigned_var_spill_reg(self, dtype: DType, reg: RegBase, stack: int, k: list[str]):
    self.a = Allocator(3, 3)
    self.uop1 = UOp(Ops.CONST, arg=1, dtype=dtype)
    self.var1 = Variable(self.uop1, 0, 10)
    self.uop2 = UOp(Ops.CONST, arg=2, dtype=dtype)
    self.var2 = Variable(self.uop2, 0, 10)
    self.a.uops[self.uop1] = self.var1
    self.a.uops[self.uop2] = self.var2
    self.a.assign_reg(reg, self.uop2)
    self.a.flush_kernel()
    self.a.assign_reg(reg, self.uop1)
    ret = self.a.flush_kernel()
    assert ret == k
    assert self.var1.reg == reg
    assert self.var2.reg == None
    assert self.var2.stack == stack

  def test_unassigned_var_spill_reg_i32(self):
    k = ["mov [rbp - 8], rax"] if Arch.x86 else ["str x0, [x29, #-8]"]
    self._unassigned_var_spill_reg(dtypes.int, IReg(0), 8, k)

  def test_unassigned_var_spill_reg_i64(self):
    k = ["mov [rbp - 8], rax"] if Arch.x86 else ["str x0, [x29, #-8]"]
    self._unassigned_var_spill_reg(dtypes.int64, IReg(0), 8, k)

  def test_unassigned_var_spill_reg_f32(self):
    k = ["movss [rbp - 16], xmm0"] if Arch.x86 else ["str d0, [x29, #-16]"]
    self._unassigned_var_spill_reg(dtypes.float32, FReg(0), 16, k)

  def test_unassigned_var_spill_reg_f64(self):
    k = ["movsd [rbp - 16], xmm0"] if Arch.x86 else ["str d0, [x29, #-16]"]
    self._unassigned_var_spill_reg(dtypes.float64, FReg(0), 16, k)

  def _assigned_var_spill_reg(self, dtype: DType, reg: RegBase,
                              reg_old: RegBase,
                              stack: int,
                              k: list[str]):
    self.a = Allocator(3, 3)
    self.uop1 = UOp(Ops.CONST, arg=1, dtype=dtype)
    self.var1 = Variable(self.uop1, 0, 10)
    self.uop2 = UOp(Ops.CONST, arg=2, dtype=dtype)
    self.var2 = Variable(self.uop2, 0, 10)
    self.a.uops[self.uop1] = self.var1
    self.a.uops[self.uop2] = self.var2
    self.a.assign_reg(reg_old, self.uop1)
    self.a.assign_reg(reg, self.uop2)
    self.a.flush_kernel()
    self.a.assign_reg(reg, self.uop1)
    ret = self.a.flush_kernel()
    assert ret == k
    assert self.var1.reg == reg
    assert self.var2.reg == None
    assert self.var2.stack == stack

  def test_assigned_var_spill_reg_i32(self):
    k = ["mov [rbp - 8], rax", "mov rax, rcx"] if Arch.x86 else [
      "str x0, [x29, #-8]", "mov x0, x1"
      ]
    self._assigned_var_spill_reg(dtypes.int, IReg(0), IReg(1), 8, k)

  def test_assigned_var_spill_reg_i64(self):
    k = ["mov [rbp - 8], rax", "mov rax, rcx"] if Arch.x86 else [
      "str x0, [x29, #-8]", "mov x0, x1"
      ]
    self._assigned_var_spill_reg(dtypes.int64, IReg(0), IReg(1), 8, k)

  def test_assigned_var_spill_reg_f32(self):
    k = ["movss [rbp - 16], xmm0", "movq xmm0, xmm1"] if Arch.x86 else [
      "str d0, [x29, #-16]", "fmov d0, d1"
      ]
    self._assigned_var_spill_reg(dtypes.float32, FReg(0), FReg(1), 16, k)

  def test_assigned_var_spill_reg_f64(self):
    k = ["movsd [rbp - 16], xmm0", "movq xmm0, xmm1"] if Arch.x86 else [
      "str d0, [x29, #-16]", "fmov d0, d1"
      ]
    self._assigned_var_spill_reg(dtypes.float64, FReg(0), FReg(1), 16, k)
