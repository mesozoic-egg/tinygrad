from sqlite3 import Binary
from typing import List, Dict, Optional, cast
from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat, UPatAny
from tinygrad.renderer import Renderer
from tinygrad.helpers import DEBUG 
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType
from collections import OrderedDict, defaultdict
from typing import List, Dict, Optional, cast, Literal, Union, Callable
import struct, math, unittest, platform, enum
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
  def render32(self): raise NotImplementedError()
  def render64(self): raise NotImplementedError()
  def render(self, itemsize: int): raise NotImplementedError()

class IReg(RegBase):
  size = 64
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
class Variable:
  def __init__(self, uop: UOp, start: int, end: int):
    """
    Args:
    end:  the index in the linearized uops *after* which the variable is expired
          start and end are both inclusive. 
    size: size in bytes (int32: 4, float64: 8)
    """
    self.uop, self.start, self.end = uop, start, end
    self.reg: Optional[RegBase] = None
    self.stack: Optional[int] = None
    self.mem: Optional[str] = None
  
  @property
  def name(self): return repr(self.uop)[:100]

  def __repr__(self):
    location = f" reg:{self.reg}" if self.reg is not None else f" stack:{self.stack}" if self.stack is not None else ""
    return f"({self.start}-{self.end} reg:{self.reg} stack:{self.stack})"

  def store(self, dst: str) -> list[str]:
    assert self.reg is not None
    to_stack = dst == "stack"
    to_mem = dst == "mem"
    assert to_stack ^ to_mem
    assert getattr(self, dst) is not None
    if to_stack:
      assert self.stack is not None
      note = f""
      if Arch.arm:
        return [f"str {self.reg.render64()}, [x29, #-{self.stack}]"]
      else:
        op = "mov" if dtypes.is_int(self.uop.dtype) or hasattr(self.uop.dtype, "_base") else "movss"
        return [f"{op} [rbp - {self.stack}], {self.reg.render64()}"]
    else:
      raise Exception("not implemented")

  def load(self, reg: RegBase, src: str) -> list[str]:
    assert self.reg is None
    from_stack = src == "stack"
    from_mem = src == "mem"
    assert from_stack ^ from_mem
    self.reg = reg
    if from_stack:
      assert self.stack is not None
      if Arch.arm:
        return [f"ldr {reg.render64()}, [x29, #-{self.stack}]"]
      else:
        op = "mov" if dtypes.is_int(self.uop.dtype) or hasattr(self.uop.dtype, "_base") else "movss"
        return [f"{op} {reg.render64()}, [rbp - {self.stack}]"]
    else: raise Exception("not implemented")

class Allocator:
  def __init__(self, num_ireg: int, num_freg: int = 0):
    self.pool: list[RegBase] = [IReg(i) for i in range(num_ireg-1, -1, -1)]
    self.pools: dict[type[RegBase], list[RegBase]] = {
      IReg: [IReg(i) for i in range(num_ireg)],
      FReg: [FReg(i) for i in range(num_freg)]
    }
    self.variables: list[Variable] = []
    self.stack_size = 0
    self.uops: dict[UOp, Variable] = {}
    self.index = 0
    self.reserved: dict[UOp, int] = {}
    self.x86_params: dict[int, int] = {
      0: 7, #R7 (rdi)
      1: 6, #R6 (rsi)
      2: 2, #R2 (rdx)
      3: 1, #R1 (rcx)
      4: 8, #R8
      5: 9, #R9 
    }
    self.kernel: list[str] = []
    self.i: int = 0

  def __getitem__(self, _key: UOp) -> RegBase:
    return self.assign(_key)

  def flush_kernel(self) -> list[str]:
    ret = self.kernel
    self.kernel = []
    return ret

  def extend_kernel(self, l: list[str]):
    self.kernel.extend(l)

  def alloc(self, excludes: list[UOp]=[], reg_type: Optional[type[RegBase]]=None) -> tuple[RegBase, list[str]]:
    if reg_type is not None:
      pool = self.pools[reg_type]
      if len(pool):
        return pool.pop(0), []
      else:
        if reg_type is FReg:
          raise Exception("Not sure how to spill float register yet")
        vars_in_regs = []
        for u, var in self.uops.items():
          if var.reg is not None and u not in excludes and u not in self.reserved:
            vars_in_regs.append(var)
        if len(vars_in_regs) == 0: raise Exception("No avaialble registers")
        sorted_vars = sorted(vars_in_regs, key=lambda i: i.end, reverse=True)
        last_ending_var, *vars = sorted_vars
        self.move_var_to_stack(last_ending_var)
        reg = self.pools[reg_type].pop(0)
        return reg, []
    else: raise Exception("Dead branch")

  def share(self, dst: UOp, src: UOp):
    dst_var, src_var = self.uops[dst], self.uops[src]
    reg = src_var.reg
    assert reg, f"Source UOp must already been assigned to register {src}"
    dst_var.reg = src_var.reg

  def return_reg(self, reg: RegBase):
    self.pools[type(reg)].insert(0, reg)

  def move_var_to_stack(self, v: Variable):
    reg = v.reg
    assert reg
    self.return_reg(reg)
    assert reg is not None
    ret = self.save_var_to_stack(v)
    v.reg = None

  def save_var_to_stack(self, v: Variable):
    assert v.reg is not None
    if v.stack is None:
      self.stack_size += (v.reg.size // 8)
      v.stack = self.stack_size
    k = v.store("stack")
    self.extend_kernel(k)

  def assign(self, _key: UOp, excludes: list[UOp]=[], reserve: bool=False,
             reg_type: Optional[type[RegBase]]=IReg) -> RegBase:
    if _key not in self.uops:
      raise Exception("Attempting to access a non-existent variable, maybe expired?")
    var = self.uops[_key]
    if var.reg is not None:
      return var.reg
    reg, kernel = self.alloc(excludes=excludes, reg_type=reg_type)
    if var.stack is not None:
      self.extend_kernel(var.load(reg, "stack"))
    if reserve: self.reserved[_key] = 1
    var.reg = reg
    assert var.reg is not None
    return reg
  def assign_i32(self, _key: UOp, excludes: list[UOp]=[], reserve: bool = False):
    return self.assign(_key, excludes, reserve, reg_type=IReg).render32()
  def assign_i64(self, _key: UOp, excludes: list[UOp]=[], reserve: bool = False):
    return self.assign(_key, excludes, reserve, reg_type=IReg).render64()
  def assign_f32(self, _key: UOp, excludes: list[UOp]=[], reserve: bool = False):
    return self.assign(_key, excludes, reserve, reg_type=FReg).render32()
  def assign_f64(self, _key: UOp, excludes: list[UOp]=[], reserve: bool = False):
    return self.assign(_key, excludes, reserve, reg_type=FReg).render64()
  
  def release(self, uop: UOp): del self.reserved[uop] 

  def free_expired(self, i: int):
    expired: list[UOp] = []
    assigned_regs: dict[RegBase, int] = defaultdict(int)
    for uop, var in self.uops.items():
      if var.end < i: expired.append(uop)
      if var.reg: assigned_regs[var.reg] += 1
      if var.reg and var.end < i: assigned_regs[var.reg] -= 1
    for uop in expired:
      del self.uops[uop]
      if self.reserved.get(uop): self.release(uop)
    for reg, count in assigned_regs.items():
      if count == 0:
        pool = self.pools[type(reg)]
        pool.insert(0, reg)
  
def stack_all(a: Allocator):
  for u, var in a.uops.items():
    # Previously was also checking var.stack and missed updated value
    if var.reg is not None and u not in a.reserved:
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
  (Ops.MUL, ArchType.X86, IReg): "imul",
  (Ops.MUL, ArchType.X86, FReg, 32): "mulss",
  (Ops.MUL, ArchType.X86, FReg, 64): "mulsd",
  (Ops.MUL, ArchType.ARM, IReg): "mul",
  (Ops.MUL, ArchType.ARM, FReg): "fmul",
  (Ops.ASSIGN, ArchType.ARM, IReg): "mov",
  (Ops.ASSIGN, ArchType.ARM, FReg): "fmov",
  (Ops.ASSIGN, ArchType.X86, IReg): "mov",
  (Ops.ASSIGN, ArchType.X86, FReg, 32): "movss",
  (Ops.ASSIGN, ArchType.X86, FReg, 64): "movsd",
})

def alu(ctx, x):
  dtype = x.src[0].dtype
  reg_type = IReg if dtypes.is_int(dtype) else FReg
  src0 = ctx.r.assign(x.src[0], reg_type=reg_type)
  src1 = ctx.r.assign(x.src[1], excludes=[x.src[0]], reg_type=reg_type)
  if ctx.r.uops[x.src[0]].end == ctx.r.i:
    ctx.r.share(x, x.src[0])
    dst = src0
  else:
    dst = ctx.r.assign(x, excludes=list(x.src), reg_type=reg_type)
  operator = AluOps.get((x.op, Arch.arch, reg_type, 8*x.dtype.itemsize))
  _src0, _src1, _dst = src0.render(dtype.itemsize), src1.render(dtype.itemsize), dst.render(dtype.itemsize)
  if Arch.arm:
    return [f"{operator} {_dst}, {_src0}, {_src1};"]
  else:
    _mov = "mov" if dtypes.is_int(dtype) else "movss" 
    if _dst == _src0:
      return [f"{operator} {_dst}, {_src1}"]
    else:
      return [f"{_mov} {_dst}, {_src0}",
        f"{operator} {_dst}, {_src1}",]

def acc(ctx, x, acc, src):
  dtype = x.src[0].dtype
  _acc = ctx.r.uops[acc].reg.render(dtype.itemsize)
  _src = ctx.r.uops[src].reg.render(dtype.itemsize)
  ctx.r.share(x, acc)
  reg_type = IReg if dtypes.is_int(dtype) else FReg
  operator = AluOps.get((Ops.ADD, Arch.arch, reg_type, 8*x.dtype.itemsize))
  if Arch.arm:
    return [f"{operator} {_acc}, {_acc}, {_src}"]
  else:
    return [f"{operator} {_acc}, {_src}"]

def const(ctx, x):
  reg = ctx.r.assign(x, reg_type=FReg)
  reg_str = reg.render(x.dtype.itemsize)
  label = f"const_{len(ctx.mem)}"
  if Arch.arm:
    ctx.mem.append((label, f".single {x.arg}"))
    temp_reg, kernel = ctx.r.alloc([reg], IReg)
    ctx.r.return_reg(temp_reg)
    return [f"adrp {temp_reg}, {label}",
      f"ldr {reg_str}, [{temp_reg}, #:lo12:{label}]"]
  else:
    op = "movss" if x.dtype.itemsize == 4 else "movsd"
    ctx.mem.append((label, f".float {x.arg}"))
    return [ f"{op} {reg_str}, [rip+{label}]" ]

def _range(ctx, x):
  stack_all(ctx.r)
  counter = ctx.r.assign(x, reserve=True, reg_type=IReg).render64()
  return [
      f"mov {counter}, #0",
      f"\n.LOOP_{x.arg}:"
  ]

def endrange(ctx, x):
  acc, end = x.src[0], x.src[0].src[0]
  stack_all(ctx.r)
  acc_reg = ctx.r.assign_i64(acc)
  ctx.r.release(x.src[0])
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
  src0_str = ctx.r.assign(src0, reg_type=IReg).render64()
  src1_str = ctx.r.assign(src1, excludes=[src0], reg_type=IReg).render64()
  reg = ctx.r.assign(x, excludes=[src0, src1], reg_type=IReg).render64()
  multiplier = src0.dtype.itemsize
  lsl = int(math.log2(multiplier))
  if Arch.arm:
    return [ f"add {reg}, {src0_str}, {src1_str}, lsl #{lsl}" ]
  else:
    return [
      f"mov {reg}, {src1_str}",
      f"shl {reg}, {lsl}",
      f"add {reg}, {src0_str}",
    ]

def assign(ctx, x):
  reg_type = IReg if dtypes.is_int(x.src[0].dtype) else FReg
  dst = ctx.r.assign(x, reg_type=reg_type)
  src = ctx.r.assign(x.src[1], excludes=[x.src[0]], reg_type=reg_type)
  opcode = AluOps.get((x.op, Arch.arch, reg_type, 8*x.dtype.itemsize))
  ctx.r.uops[x].stack = ctx.r.uops[x.src[0]].stack
  return [f"{opcode} {dst}, {src}"]

complex_rewrites = PatternMatcher([
  (UPat(Ops.ASSIGN, name="x"), assign),
  (UPat(Ops.INDEX, name="x"), _index),
  (UPat(Ops.RANGE, name="x"), _range),
  (UPat(Ops.ENDRANGE, name="x"), endrange),
  (UPat(GroupOp.ALU, name="x"), alu),
  (UPat(Ops.CONST, name="x", dtype=dtypes.floats), const),
])
x86_rewrite = PatternMatcher([
  (UPat(Ops.ADD, name="x", src=(UPat(Ops.DEFINE_REG, name="acc"), UPat(name="src"))), acc),
  (UPat(Ops.CONST, name="x", dtype=dtypes.int32), lambda ctx, x: [f"mov {ctx.r.assign_i32(x)}, {x.arg:#x}"]),
  (UPat(Ops.CONST, name="x", dtype=dtypes.int64), lambda ctx, x: [f"mov {ctx.r.assign_i64(x)}, {x.arg:#x}"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.int))),
      lambda ctx, x, addr, src: [f"mov [{ctx.r.assign_i64(addr)}], {ctx.r.assign_i32(src)}"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.int64))),
      lambda ctx, x, addr, src: [f"mov [{ctx.r.assign_i64(addr)}], {ctx.r.assign_i64(src)}"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.float32))),
      lambda ctx, x, addr, src: [f"movss [{ctx.r.assign_i64(addr)}], {ctx.r.assign_f32(src)}"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.float64))),
      lambda ctx, x, addr, src: [f"movsd [{ctx.r.assign_i64(addr)}], {ctx.r.assign_f64(src)}"]),

  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.int32, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"mov {ctx.r.assign_i32(x, reserve=True)}, {ctx.r.assign_i32(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.int64, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"mov {ctx.r.assign_i64(x, reserve=True)}, {ctx.r.assign_i64(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.float32, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"movss {ctx.r.assign_f32(x, reserve=True)}, {ctx.r.assign_f32(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.float64, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"movsd {ctx.r.assign_f64(x, reserve=True)}, {ctx.r.assign_f64(src)}"]),

  (UPat(Ops.LOAD, name="x", dtype=dtypes.int32, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"mov {ctx.r.assign_i32(x)}, [{ctx.r.assign_i64(src)}]"]),
  (UPat(Ops.LOAD, name="x", dtype=dtypes.int64, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"mov {ctx.r.assign_i64(x)}, [{ctx.r.assign_i64(src)}]"]),
  (UPat(Ops.LOAD, name="x", dtype=dtypes.float32, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"movss {ctx.r.assign_f32(x)}, [{ctx.r.assign_i64(src)}]"]),
  (UPat(Ops.LOAD, name="x", dtype=dtypes.float64, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"movsd {ctx.r.assign_f32(x)}, [{ctx.r.assign_i64(src)}]"]),
]) + complex_rewrites

arm_rewrite = PatternMatcher([
  (UPat(Ops.ADD, name="x", src=(UPat(Ops.DEFINE_REG, name="acc"), UPat(name="src"))), acc),
  (UPat(Ops.CONST, name="x", dtype=dtypes.int32), lambda ctx, x: [f"mov {ctx.r.assign_i32(x)}, #{x.arg}"]),
  (UPat(Ops.CONST, name="x", dtype=dtypes.int64), lambda ctx, x: [f"mov {ctx.r.assign_i64(x)}, #{x.arg}"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.int))),
      lambda ctx, x, addr, src: [f"str {ctx.r.assign_i32(src)}, [{ctx.r.assign_i64(addr)}]"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.int64))),
      lambda ctx, x, addr, src: [f"str {ctx.r.assign_i64(src)}, [{ctx.r.assign_i64(addr)}]"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.float32))),
      lambda ctx, x, addr, src: [f"str {ctx.r.assign_f32(src)}, [{ctx.r.assign_i64(addr)}]"]),
  (UPat(Ops.STORE, name="x", src=(UPat(name="addr"), UPat(name="src", dtype=dtypes.float64))),
      lambda ctx, x, addr, src: [f"str {ctx.r.assign_f64(src)}, [{ctx.r.assign_i64(addr)}]"]),

  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.int32, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"mov {ctx.r.assign_i32(x, reserve=True)}, {ctx.r.assign_i32(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.int64, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"mov {ctx.r.assign_i64(x, reserve=True)}, {ctx.r.assign_i64(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.float32, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"fmov {ctx.r.assign_f32(x, reserve=True)}, {ctx.r.assign_f32(src)}"]),
  (UPat(Ops.DEFINE_REG, name="x", dtype=dtypes.float64, src=(UPat(name="src"),), allow_any_len=True),
      lambda ctx, x, src: [f"fmov {ctx.r.assign_f64(x, reserve=True)}, {ctx.r.assign_f64(src)}"]),

  (UPat(Ops.LOAD, name="x", dtype=dtypes.int32, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"ldr {ctx.r.assign_i32(x)}, [{ctx.r.assign_i64(src)}]"]),
  (UPat(Ops.LOAD, name="x", dtype=dtypes.int64, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"ldr {ctx.r.assign_i64(x)}, [{ctx.r.assign_i64(src)}]"]),
  (UPat(Ops.LOAD, name="x", dtype=dtypes.float32, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"ldr {ctx.r.assign_f32(x)}, [{ctx.r.assign_i64(src)}]"]),
  (UPat(Ops.LOAD, name="x", dtype=dtypes.float64, src=(UPat(name="src",),)),
     lambda ctx, x, src: [f"ldr {ctx.r.assign_f64(x)}, [{ctx.r.assign_i64(src)}]"]),
]) + complex_rewrites

extra_matcher = PatternMatcher([
  (UPat(Ops.ASSIGN, name="assign", src=(UPat(Ops.DEFINE_REG, name="acc"), UPat(Ops.ADD, name="add"))), lambda ctx, assign, acc, add: add),
])

class AsmRenderer(Renderer):
  supports_float4 = False
  has_local = False
  has_shared = False
  global_max = None
  extra_matcher = extra_matcher

  def __init__(self) -> None:
    super().__init__()
    arch = platform.machine()
    self.arm = arch == "aarch64"
    self.x86 = arch == "x86_64"
    assert self.arm ^ self.x86

  def __getitem__(self, key: UOp):
    return self.r[key]

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
    if DEBUG >= 6:
      print(uops[-1])

    name = "test"
    uop_order = {} 
    var_intervals: dict[UOp, Variable] = OrderedDict()
    for i, u in enumerate(uops):
      #if u.dtype is not dtypes.void:
      var = Variable(u, i, -1)
      if u.op is Ops.DEFINE_GLOBAL:
        if Arch.arm:
          var.reg = r.pools[IReg].pop(0)
        else:
          reg_num = r.x86_params[u.arg]
          reg_idx = r.pools[IReg].index(IReg(reg_num))
          assert reg_idx > -1
          var.reg = r.pools[IReg].pop(reg_idx)
      var_intervals[u] = var
    for i, u in enumerate(uops):
      for src in u.src:
        if src.dtype is not dtypes.void:
          prev = var_intervals[src].end
          var_intervals[src].end = max(prev, i)
    for v in var_intervals.values():
      if v.end == -1: v.end = len(uops)
    self.r.uops = var_intervals
    if Arch.x86:
      r.pools[IReg].pop(r.pools[IReg].index(IReg(5)))

    if DEBUG.value >= 6:
      for _u, v in r.uops.items(): print(v, oneline_uop(_u))
    for i,u in enumerate(uops):
      self.r.i = i
      if DEBUG.value >= 6:
        print("=================================")
        print(i, r.uops[u], u)
      r.free_expired(i)
      if u.op is Ops.DEFINE_GLOBAL:
        self.r.move_var_to_stack(r.uops[u])
        kernel.extend(self.r.flush_kernel())
      elif u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
      else:
        rewriter = arm_rewrite if Arch.arm else x86_rewrite
        if (l:=rewriter.rewrite(u, ctx=self)) is None:
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        l = cast(list[str], l)
        l = [*r.flush_kernel(), *l]
        if DEBUG.value >= 6:
          print("\n".join(kernel)[-100:])
          print("\033[32m", "\n".join(l), "\033[0m", sep="")
        kernel.extend(l)
    prologue = [
      "stp x29, x30, [sp, #-16]!",
      "mov x29, sp",
      "mov x30, sp",
      f"sub sp, sp, #{r.stack_size}",
    ] if self.arm else [
      "push rbp",
      "mov rbp, rsp",
      f"sub rsp, {r.stack_size}",
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
    mem_data = [f"{a}: {b}" for a,b in mem]
    data_section = [
      ".section .data",
      ".p2align 2",
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
    with open("../tg-dev/matmul6/kernel.s", "wt") as f: f.write(ret)
    return ret

class Tests(unittest.TestCase):
  def test_to_hex(self):
    assert float32_to_hex(20.0) == "0x41a00000"
    assert float32_to_hex(49.193) == "0x4244c5a2"

class TestAllocatorExpire(unittest.TestCase):
  def setUp(self):
    self.a = Allocator(16)
    uop1 = UOp(Ops.CONST, arg=1)
    uop2 = UOp(Ops.CONST, arg=2)
    self.uop1, self.uop2 = uop1, uop2
    self.a.uops[uop1] = Variable(uop1, 0, 2)
    self.a.uops[uop2] = Variable(uop2, 0, 10)
    self.a.assign(uop1, reserve=True)
    self.a.assign(uop2, reserve=True)
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
    self.a = Allocator(16)
    uop1 = UOp(Ops.CONST, arg=1)
    uop2 = UOp(Ops.CONST, arg=2)
    self.uop1, self.uop2 = uop1, uop2
    self.var1, self.var2 = Variable(uop1, 0, 2), Variable(uop2, 0, 10)
    self.a.uops[uop1] = self.var1
    self.a.uops[uop2] = self.var2
    self.a.assign(uop1, reserve=True)
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
    self.a = Allocator(2)
    uop1 = UOp(Ops.CONST, arg=1)
    uop2 = UOp(Ops.CONST, arg=2)
    uop3 = UOp(Ops.CONST, arg=3)
    self.uop1, self.uop2, self.uop3 = uop1, uop2, uop3
    self.a.uops[uop1] = Variable(uop1, 0, 9)
    self.a.uops[uop2] = Variable(uop2, 0, 10)
    self.a.uops[uop3] = Variable(uop3, 0, 11)
    self.a.assign(uop1)
    self.a.assign(uop2)
  def tearDown(self): del self.uop1, self.uop2, self.uop3, self.a

  def test_spill(self):
    reg = self.a.assign(self.uop3)
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
    reg = self.a.assign(self.uop3)
    kernel = self.a.flush_kernel()
    assert self.a.uops[self.uop2].stack == 0
    assert self.a.uops[self.uop3].stack == 8
    assert len(kernel) == 2# and kernel[1].startswith("ldr")
    assert self.a.stack_size == 16

  def test_spill_with_stack_str(self):
    assert self.a.stack_size == 0
    self.a.assign(self.uop3)
    assert self.a.stack_size == 8
    assert self.a.uops[self.uop2].stack == 8

class TestAllocatorStackAll(unittest.TestCase):
  """
  Ops.RANGE and Ops.DEFINE_REG's Variable could change, the change need to 
  be saved in stack
  """
  def setUp(self):
    self.a = Allocator(16)
    uop1 = UOp(Ops.RANGE)
    self.uop1 = uop1
    var = Variable(uop1, 0, 10)
    var.stack = 4
    self.a.uops[uop1] = var
    self.a.assign(uop1)
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
    self.a = Allocator(2)
    self._setup()
    self.a.assign(self.uop1)
    self.a.assign(self.uop2)
    self.a.assign(self.uop3, excludes=[self.uop2])
    assert self.var1.reg is None and self.var1.stack == 8
    assert self.var2.reg == IReg(1)
    assert self.var3.reg == IReg(0)
  def test_exclude_not_enough_reg(self):
    self.a = Allocator(1)
    self._setup()
    self.a.assign(self.uop2)
    self.a.assign(self.uop3)
  def test_exclude_not_enough_reg_raise(self):
    self.a = Allocator(1)
    self._setup()
    self.a.assign(self.uop2)
    with self.assertRaises(Exception):
      self.a.assign(self.uop3, excludes=[self.uop2])
  def test_reserve(self):
    self.a = Allocator(2)
    self._setup()
    self.a.assign(self.uop1)
    self.a.assign(self.uop2, reserve=True)
    self.a.assign(self.uop3)
    assert self.var3.reg == IReg(0)
  def test_reserve_not_enough_reg(self):
    self.a = Allocator(2)
    self._setup()
    self.a.assign(self.uop1, reserve=True)
    self.a.assign(self.uop2, reserve=True)
    with self.assertRaises(Exception):
      self.a.assign(self.uop3)
  def test_reserve_release(self):
    self.a = Allocator(2)
    self._setup()
    self.a.assign(self.uop1, reserve=True)
    self.a.assign(self.uop2, reserve=True)
    self.a.release(self.uop2)
    self.a.assign(self.uop3)
  def test_reserve_not_enough_reg_pair(self):
    self.a = Allocator(3)
    self._setup()
    self.a.assign(self.uop1, reserve=True)
    self.a.assign(self.uop2, reserve=True)
    with self.assertRaises(Exception):
      self.a.assign(self.uop3)
      self.a.assign(self.uop4, excludes=[self.uop3])

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
    assert len(cast(list[str], l)) == 2

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
    self.r.i = 2
    l = rewriter.rewrite(self.uop3, self)
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

  @x86
  def test_x86_assign_int32(self):
    self._assign(dtypes.int32, [
      "mov rax, rcx",
    ])
  
  @x86
  def test_x86_assign_int64(self):
    self._assign(dtypes.int64, [
      "mov rax, rcx",
    ])
  
  @x86
  def test_x86_assign_float32(self):
    self._assign(dtypes.float32, [
      "movss xmm0, xmm1",
    ])
  
  @x86
  def test_x86_assign_float64(self):
    self._assign(dtypes.float64, [
      "movsd xmm0, xmm1",
    ])
  
  @arm
  def test_arm_assign_int32(self):
    self._assign(dtypes.int32, [
      "mov x0, x1",
    ])
  
  @arm
  def test_arm_assign_int64(self):
    self._assign(dtypes.int64, [
      "mov x0, x1",
    ])
  
  @arm
  def test_arm_assign_float32(self):
    self._assign(dtypes.float32, [
      "fmov d0, d1",
    ])
  
  @arm
  def test_arm_assign_float64(self):
    self._assign(dtypes.float64, [
      "fmov d0, d1",
    ])

  @x86
  def test_x86_range(self):
    a = UOp(Ops.RANGE, arg=0, src=(
      UOp(Ops.CONST, arg=4),
    ))
    self.render(a, ["mov rax, #0", "\n.LOOP_0:"])
    b = UOp(Ops.ENDRANGE, src=(
      a,
    ))
    self.render(b, ["inc rcx", "cmp rcx, 4", "jl .LOOP_0"])
  @arm
  def test_arm_range(self):
    a = UOp(Ops.RANGE, arg=0, src=(
      UOp(Ops.CONST, arg=4),
    ))
    self.render(a, ["mov x0, #0", "\n.LOOP_0:"])
    b = UOp(Ops.ENDRANGE, src=(
      a,
    ))
    self.render(b, ["add x1, x1, #1", "cmp x1, #4", "b.lt .LOOP_0"])
  @x86
  def test_x86_index(self):
    a = UOp(Ops.INDEX, dtypes.int.ptr(16), arg=None, src=(
      x2:=UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), arg=0, src=()),
      x3:=UOp(Ops.CONST, dtypes.int, arg=None, src=()),))
    self.render(a, ["mov rdx, rcx",
                    "shl rdx, 2",
                    "add rdx, rax"])
  @arm
  def test_arm_index(self):
    a = UOp(Ops.INDEX, dtypes.int.ptr(16), arg=None, src=(
      x2:=UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), arg=0, src=()),
      x3:=UOp(Ops.CONST, dtypes.int, arg=None, src=()),))
    self.render(a, ["add x2, x0, x1, lsl #2"]);

