import argparse, io, sys, struct, dataclasses, re, os
from tinygrad.runtime.support.assembler.CuAsmParser import CuAsmParser, CuAsmSection, CuAsmSegment
from tinygrad.runtime.support.assembler.CuKernelAssembler import CuKernelAssembler
from tinygrad.runtime.support.assembler.CuInsAssemblerRepos import CuInsAssemblerRepos
from elftools.elf.structs import ELFStructs
from typing import Optional, Union
from tinygrad.runtime.support.assembler.CuSMVersion import CuSMVersion

def lookup(ins:str):
  arch = CuSMVersion(86)
  repo = CuInsAssemblerRepos(InsAsmDict="/home/alvy/gbin/tinygrad2/tinygrad/runtime/support/assembler/InsAsmRepos/DefaultInsAsmRepos.sm_86.txt",
                             arch=arch)
  if ins in repo.m_InsAsmDict:
    return repr(repo[ins])
  else:
    return [k for k in repo.m_InsAsmDict.keys() if k.startswith(ins)]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('ins', type=str)
  _ins = parser.parse_args().ins
  print(lookup(_ins))
