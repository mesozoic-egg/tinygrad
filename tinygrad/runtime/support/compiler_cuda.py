import subprocess, hashlib, tempfile, ctypes, ctypes.util, re, pathlib, os
from tempfile import NamedTemporaryFile
from typing import Callable
from tinygrad.helpers import to_char_p_p, colored, init_c_var, getenv
import tinygrad.runtime.autogen.nvrtc as nvrtc
from tinygrad.device import Compiler, CompileError

PTX = getenv("PTX")  # this shouldn't be here, in fact, it shouldn't exist

def _get_bytes(arg, get_str, get_sz, check) -> bytes:
  sz = init_c_var(ctypes.c_size_t(), lambda x: check(get_sz(arg, ctypes.byref(x))))
  return ctypes.string_at(init_c_var(ctypes.create_string_buffer(sz.value), lambda x: check(get_str(arg, x))), size=sz.value)

def nvrtc_check(status, ctx=None):
  if status != 0:
    err_log = _get_bytes(ctx, nvrtc.nvrtcGetProgramLog, nvrtc.nvrtcGetProgramLogSize, lambda _: None).decode() if ctx else ""
    raise CompileError(f"Nvrtc Error {status}, {ctypes.string_at(nvrtc.nvrtcGetErrorString(status)).decode()}\n{err_log}")

def jitlink_check(status, ctx=None):
  if status != 0:
    err_log = _get_bytes(ctx, nvrtc.nvJitLinkGetErrorLog, nvrtc.nvJitLinkGetErrorLogSize, lambda _: None).decode() if ctx else ""
    raise CompileError(f"NvJitLink Error {status}, {nvrtc.nvJitLinkResult__enumvalues.get(status, 'Unknown')}\n{err_log}")

def pretty_ptx(s):
  # all expressions match `<valid_before><expr><valid_after>` and replace it with `<valid_before>color(<expr>)<valid_after>`
  s = re.sub(r'([!@<\[\s,\+\-;\n])((?:[_%$][\w%\$_]+(?:\.[xyz])?\:?)|(?:buf\d+))([<>\]\s,\+\-;\n\)])', lambda m:m[1]+colored(m[2], "blue")+m[3], s, flags=re.M) # identifiers  # noqa: E501
  s = re.sub(r'(.)((?:b|s|u|f)(?:8|16|32|64)|pred)([\.\s])', lambda m:m[1]+colored(m[2], "green")+m[3], s, flags=re.M) # types
  s = re.sub(r'^(\s*)([\w]+)(.*?;$)', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # instructions
  s = re.sub(r'([<>\[\]\s,\+\-;])((?:0[fF][0-9a-fA-F]{8})|(?:[0-9]+)|(?:0[xX][0-9a-fA-F]+))([<>\[\]\s,\+\-;])', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # numbers  # noqa: E501
  s = re.sub(r'(\.)(param|reg|global)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # space
  s = re.sub(r'(\.)(version|target|address_size|visible|entry)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # derivatives
  return s

def cuda_disassemble(lib, arch):
  try:
    fn = (pathlib.Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
    with open(fn + ".ptx", "wb") as f: f.write(lib)
    subprocess.run(["ptxas", f"-arch={arch}", "-o", fn, fn+".ptx"], check=True)
    print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
  except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains ptxas/nvdisasm binary of compatible version.")

class CUDACompiler(Compiler):
  def __init__(self, arch:str, cache_key:str="cuda"):
    self.arch, self.compile_options = arch, [f'--gpu-architecture={arch}', "-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include/"]
    nvrtc_check(nvrtc.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_{cache_key}_{self.arch}")
  def _compile_program(self, src:str, nvrtc_get_content:Callable, nvrtc_get_size:Callable) -> bytes:
    nvrtc_check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog := nvrtc.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    nvrtc_check(nvrtc.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options])), prog)
    data = _get_bytes(prog, nvrtc_get_content, nvrtc_get_size, nvrtc_check)
    nvrtc_check(nvrtc.nvrtcDestroyProgram(ctypes.byref(prog)))
    return data
  def compile(self, src:str) -> bytes: return self._compile_program(src, nvrtc.nvrtcGetPTX, nvrtc.nvrtcGetPTXSize)
  def disassemble(self, lib:bytes):
    try:
      fn = (pathlib.Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
      with open(fn + ".cubin", "wb") as f: f.write(lib)
      print(subprocess.check_output(["nvdisasm", fn+".cubin"]).decode('utf-8'))
    except Exception as e: print("Failed to disasm cubin:", str(e), "Make sure your PATH contains nvdisasm binary of compatible version.")

class NVCompiler(CUDACompiler):
  def __init__(self, arch:str): super().__init__(arch, cache_key="nv")
  def compile(self, src:str) -> bytes: return self._compile_program(src, nvrtc.nvrtcGetCUBIN, nvrtc.nvrtcGetCUBINSize)

class PTXCompiler(Compiler):
  def __init__(self, arch:str, cache_key="ptx"):
    self.arch = arch
    super().__init__(f"compile_{cache_key}_{self.arch}")
  def compile(self, src:str) -> bytes: return src.replace("TARGET", self.arch).replace("VERSION", "7.8" if self.arch >= "sm_89" else "7.5").encode()

class NVPTXCompiler(PTXCompiler):
  def __init__(self, arch:str): super().__init__(arch, cache_key="nv_ptx")
  def compile(self, src:str) -> bytes:
    jitlink_check(nvrtc.nvJitLinkCreate(handle := nvrtc.nvJitLinkHandle(), 1, to_char_p_p([f'-arch={self.arch}'.encode()])), handle)
    jitlink_check(nvrtc.nvJitLinkAddData(handle, nvrtc.NVJITLINK_INPUT_PTX, ptxsrc:=super().compile(src), len(ptxsrc), "<null>".encode()), handle)
    jitlink_check(nvrtc.nvJitLinkComplete(handle), handle)
    data = _get_bytes(handle, nvrtc.nvJitLinkGetLinkedCubin, nvrtc.nvJitLinkGetLinkedCubinSize, jitlink_check)
    jitlink_check(nvrtc.nvJitLinkDestroy(handle))
    return data

class SASSCompilerIntercept(Compiler):
  def __init__(self, arch:str, cache_key="sass"):
    self.arch = arch
    super().__init__(f"compile_{cache_key}_{self.arch}")
  
  def compile(self, src: str):
    from extra.sass.assembler.CubinFile import CubinFile
    from extra.sass.assembler.CuAsmParser import CuAsmParser
    with NamedTemporaryFile() as cuda_file, NamedTemporaryFile() as ptx_file, NamedTemporaryFile() as ptxas_cubin_file, \
      NamedTemporaryFile() as cuasm_file, NamedTemporaryFile() as cuasm_cubin_file: 
      with open(cuda_file.name, "w") as f: f.write(src)
      subprocess.run(["nvcc", "-arch", self.arch, "--ptx", "-x", "cu", "-o", ptx_file.name, cuda_file.name], check=True)
      subprocess.run(["ptxas", "-arch", self.arch, "-m64", "-o", ptxas_cubin_file.name, ptx_file.name], check=True)
      cf = CubinFile(ptxas_cubin_file.name)
      cf.saveAsCuAsm(cuasm_file.name)
      parser = CuAsmParser()
      parser.parse(cuasm_file.name)
      parser.saveAsCubin(cuasm_cubin_file.name)
      cubin = cuasm_cubin_file.read()
      return cubin

class SASSCompiler(Compiler):
  def __init__(self, arch:str, cache_key="sass_render"):
    self.arch = arch
    super().__init__(f"compile_{cache_key}_{self.arch}")
    
  def compile(self, src: str):
    from extra.sass.assembler.CubinFile import CubinFile
    from extra.sass.assembler.CuAsmParser import CuAsmParser
    ptx_code = ""
    sass_code = ""
    sass = 0
    for line in src.splitlines(keepends=True):
      if line.strip() == "<sass>":
        sass = 1
        continue
      elif line.strip() == "</sass>":
        sass = 0
        continue
      if sass:
        sass_code += line
      else:
        ptx_code += line
    with NamedTemporaryFile(mode="w+b", delete_on_close=True) as ptx, NamedTemporaryFile(delete_on_close=True) as cubin, \
      NamedTemporaryFile(delete_on_close=True) as cuasm, NamedTemporaryFile("w+b", delete_on_close=True) as cuasm_swapped, \
      NamedTemporaryFile(delete_on_close=True) as cubin2:
      print(ptx_code)
      ptx.write(ptx_code.encode())
      ptx.flush()
      ptx.seek(0)
      subprocess.run(["ptxas", f"-arch={self.arch}", "-m64", "-o", cubin.name, ptx.name], check=True)
      cf = CubinFile(cubin.name)
      cf.saveAsCuAsm(cuasm.name)
      state = 'seek_start'
      for line in cuasm:
        if state == 'seek_start':
          cuasm_swapped.write(line)
          if line.decode().strip() == '.text.E_32:':
            cuasm_swapped.write(sass_code.encode())
            cuasm_swapped.write(b"\n")
            state = 'skip_until_empty'
        elif state == 'skip_until_empty':
          if line.decode().strip() == '':
            cuasm_swapped.write(line)
            state = 'seek_start'
        else:
          cuasm_swapped.write(line)     
      cuasm_swapped.flush()
      cuasm_swapped.seek(0)
      parser = CuAsmParser()
      parser.parse(cuasm_swapped.name)
      parser.saveAsCubin(cubin2.name)
      ret = cubin2.read()
      return ret

      