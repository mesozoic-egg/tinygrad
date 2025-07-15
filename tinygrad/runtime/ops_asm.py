from tinygrad.device import Compiled, MallocAllocator, Compiler
from tinygrad.runtime.ops_cpu import ClangJITCompiler, CPUProgram, jit_loader
from tinygrad.renderer.asm import AsmRenderer
import subprocess

class AsmJITCompiler(Compiler):
  def __init__(self, cachekey=None): super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    obj = subprocess.check_output(['clang', '-x', 'assembler', '-c', '-', '-o', '-'], input=src.encode('utf-8'))
    #disassembled = subprocess.check_output(["objdump", "-d", "/dev/stdin"], input=obj)
    #print(disassembled.decode())
    return jit_loader(obj)

  def disassemble(self, lib:bytes): pass

class AsmDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, AsmRenderer(),
                      AsmJITCompiler(cachekey=None), CPUProgram)
