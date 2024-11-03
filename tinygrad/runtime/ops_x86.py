"""
clang '-S' '-O2' '-Wall' '-Werror' '-x' 'c' '-fPIC' '-ffreestanding' '-nostdlib' -o temp/add2.s temp/add2.c
clang '-O2' '-Wall' '-Werror' '-fPIC' '-nostdlib' -o temp/add2.o temp/add2.s 
"""
from typing import Optional, List
import ctypes, subprocess, pathlib, tempfile
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import cpu_time_execution, DEBUG, cpu_objdump
from tinygrad.renderer.cstyle import ClangRenderer

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None):
    self.args = ['-march=native'] if args is None else args
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    print("hi", self.args)
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    # with tempfile.NamedTemporaryFile(delete=True) as output_file:
    #   subprocess.check_output(['clang', '-S', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
    #                            '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
    #   asm = pathlib.Path(output_file.name).read_bytes()
    #   print("asm code")
    #   print(asm.decode())

    # with tempfile.NamedTemporaryFile(delete=True) as output_file:
    #   subprocess.check_output(['clang', '-shared', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
    #                            '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
    #   return pathlib.Path(output_file.name).read_bytes()
    # with tempfile.NamedTemporaryFile(delete=True) as output_file:
    #   subprocess.check_output(['as', '-shared', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
    #                            '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
    #   return pathlib.Path(output_file.name).read_bytes()
    
    with open("temp/add2.o", "rb") as output_file:
      return output_file.read()

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    if DEBUG >= 6: cpu_objdump(lib)
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class X86Device(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler(), ClangProgram, ClangGraph)
