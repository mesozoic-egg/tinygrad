from typing import cast, Callable, Union, Mapping
import struct, dataclasses
from collections import defaultdict
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import flatten, get_single_element, prod

def dict_to_str(d: Mapping[str, Union[str, int]]):
  ret = ""
  for k, v in d.items():
    ret += f"{k}  {v}\n"
  return ret

@dataclasses.dataclass
class SectionHeader:
  name: int; _type: str;
  flags: int; addr: int; offset: int; size: int; link: int; info: int;
  entsize: int;

@dataclasses.dataclass
class Section:
  name_str: str; num: int; align: int; data: bytes
  header: SectionHeader
  def to_asm(self):
    ret = [f".section \"{self.name_str}\", {self.num}, {self.header._type}"]
    for k, v in dataclasses.asdict(self.header).items():
      if k == "_type": k = "type"
      if type(v) == int:
        v = f"{v:#x}"
      ret.append(f".__section_{k}    {v}")
    ret.append(f".align    {self.align}")
    ret = "\n".join(ret)
    return ret


  
class SASSRenderer(Renderer):
  device = "CUDA"
  suffix = "SASS"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  tc_sm80 = [x for x in tc.cuda_sm80 if x.dtype_in in [dtypes.half, dtypes.float]]
  def __init__(self, arch:str, device="CUDA"):
    self.device, self.arch = device, arch

  def render(self, uops:list[UOp]) -> str:
    elf_header: dict[str, str] = {
      ".__elf_ident_osabi": "51",
      ".__elf_ident_abiversion": "7",
      ".__elf_type": "ET_EXEC",
      ".__elf_machine": "EM_CUDA",
      ".__elf_version": "129",
      ".__elf_entry": "0",
      ".__elf_phoff": "0xa00",
      ".__elf_shoff": "0x700",
      ".__elf_flags": "0x560556",
      ".__elf_ehsize": "64",
      ".__elf_phentsize": "56",
      ".__elf_phnum": "3",
      ".__elf_shentsize": "64",
      ".__elf_shnum": "12",
      ".__elf_shstrndx": "1",
    }
    ret = ""
    ret += dict_to_str(elf_header)
    sections = {
      "empty": Section('', 0, 0, b'', SectionHeader(0, 'SHT_NULL', 0, 0, 0, 0, 0, 0, 0,))
    }
    ret += sections['empty'].to_asm()
    ret += """

// --------------------- .shstrtab                        --------------------------
	.section  ".shstrtab", 0, SHT_STRTAB
	// all strings in .shstrtab section will be kept as is.
	.__section_name         0x1 	// offset in .shstrtab
	.__section_type         SHT_STRTAB
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x40 	// maybe updated by assembler
	.__section_size         0xdb 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0x0
	.__section_entsize      0
	.align                1 	// equivalent to set sh_addralign
    // .shstrtab[0] = b'\x00' 
    /*0000*/ .byte 0x00

    // .shstrtab[1] = b'.shstrtab\x00' 
    /*0001*/ .byte 0x2e, 0x73, 0x68, 0x73, 0x74, 0x72, 0x74, 0x61
    /*0009*/ .byte 0x62, 0x00

    // .shstrtab[2] = b'.strtab\x00' 
    /*000b*/ .byte 0x2e, 0x73, 0x74, 0x72, 0x74, 0x61, 0x62, 0x00

    // .shstrtab[3] = b'.symtab\x00' 
    /*0013*/ .byte 0x2e, 0x73, 0x79, 0x6d, 0x74, 0x61, 0x62, 0x00

    // .shstrtab[4] = b'.symtab_shndx\x00' 
    /*001b*/ .byte 0x2e, 0x73, 0x79, 0x6d, 0x74, 0x61, 0x62, 0x5f
    /*0023*/ .byte 0x73, 0x68, 0x6e, 0x64, 0x78, 0x00

    // .shstrtab[5] = b'.nv.info\x00' 
    /*0029*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x69, 0x6e, 0x66, 0x6f
    /*0031*/ .byte 0x00

    // .shstrtab[6] = b'.text.E_3\x00' 
    /*0032*/ .byte 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x45, 0x5f
    /*003a*/ .byte 0x33, 0x00

    // .shstrtab[7] = b'.nv.info.E_3\x00' 
    /*003c*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x69, 0x6e, 0x66, 0x6f
    /*0044*/ .byte 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .shstrtab[8] = b'.nv.shared.E_3\x00' 
    /*0049*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x73, 0x68, 0x61, 0x72
    /*0051*/ .byte 0x65, 0x64, 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .shstrtab[9] = b'.nv.constant0.E_3\x00' 
    /*0058*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x63, 0x6f, 0x6e, 0x73
    /*0060*/ .byte 0x74, 0x61, 0x6e, 0x74, 0x30, 0x2e, 0x45, 0x5f
    /*0068*/ .byte 0x33, 0x00

    // .shstrtab[10] = b'.rel.nv.constant0.E_3\x00' 
    /*006a*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x2e, 0x6e, 0x76, 0x2e
    /*0072*/ .byte 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74
    /*007a*/ .byte 0x30, 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .shstrtab[11] = b'.debug_frame\x00' 
    /*0080*/ .byte 0x2e, 0x64, 0x65, 0x62, 0x75, 0x67, 0x5f, 0x66
    /*0088*/ .byte 0x72, 0x61, 0x6d, 0x65, 0x00

    // .shstrtab[12] = b'.rel.debug_frame\x00' 
    /*008d*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x2e, 0x64, 0x65, 0x62
    /*0095*/ .byte 0x75, 0x67, 0x5f, 0x66, 0x72, 0x61, 0x6d, 0x65
    /*009d*/ .byte 0x00

    // .shstrtab[13] = b'.rela.debug_frame\x00' 
    /*009e*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x61, 0x2e, 0x64, 0x65
    /*00a6*/ .byte 0x62, 0x75, 0x67, 0x5f, 0x66, 0x72, 0x61, 0x6d
    /*00ae*/ .byte 0x65, 0x00

    // .shstrtab[14] = b'.nv.callgraph\x00' 
    /*00b0*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x63, 0x61, 0x6c, 0x6c
    /*00b8*/ .byte 0x67, 0x72, 0x61, 0x70, 0x68, 0x00

    // .shstrtab[15] = b'.nv.prototype\x00' 
    /*00be*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x70, 0x72, 0x6f, 0x74
    /*00c6*/ .byte 0x6f, 0x74, 0x79, 0x70, 0x65, 0x00

    // .shstrtab[16] = b'.nv.rel.action\x00' 
    /*00cc*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x72, 0x65, 0x6c, 0x2e
    /*00d4*/ .byte 0x61, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x00


	.section  ".strtab", 0, SHT_STRTAB
	.__section_name         0xb 	// offset in .shstrtab
	.__section_type         SHT_STRTAB
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x11b 	// maybe updated by assembler
	.__section_size         0xdf 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0x0
	.__section_entsize      0
	.align                1 	// equivalent to set sh_addralign
    // .strtab[0] = b'\x00' 
    /*0000*/ .byte 0x00

    // .strtab[1] = b'.shstrtab\x00' 
    /*0001*/ .byte 0x2e, 0x73, 0x68, 0x73, 0x74, 0x72, 0x74, 0x61
    /*0009*/ .byte 0x62, 0x00

    // .strtab[2] = b'.strtab\x00' 
    /*000b*/ .byte 0x2e, 0x73, 0x74, 0x72, 0x74, 0x61, 0x62, 0x00

    // .strtab[3] = b'.symtab\x00' 
    /*0013*/ .byte 0x2e, 0x73, 0x79, 0x6d, 0x74, 0x61, 0x62, 0x00

    // .strtab[4] = b'.symtab_shndx\x00' 
    /*001b*/ .byte 0x2e, 0x73, 0x79, 0x6d, 0x74, 0x61, 0x62, 0x5f
    /*0023*/ .byte 0x73, 0x68, 0x6e, 0x64, 0x78, 0x00

    // .strtab[5] = b'.nv.info\x00' 
    /*0029*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x69, 0x6e, 0x66, 0x6f
    /*0031*/ .byte 0x00

    // .strtab[6] = b'.text.E_3\x00' 
    /*0032*/ .byte 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x45, 0x5f
    /*003a*/ .byte 0x33, 0x00

    // .strtab[7] = b'.nv.info.E_3\x00' 
    /*003c*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x69, 0x6e, 0x66, 0x6f
    /*0044*/ .byte 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .strtab[8] = b'.nv.shared.E_3\x00' 
    /*0049*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x73, 0x68, 0x61, 0x72
    /*0051*/ .byte 0x65, 0x64, 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .strtab[9] = b'.rel.nv.constant0.E_3\x00' 
    /*0058*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x2e, 0x6e, 0x76, 0x2e
    /*0060*/ .byte 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74
    /*0068*/ .byte 0x30, 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .strtab[10] = b'.nv.constant0.E_3\x00' 
    /*006e*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x63, 0x6f, 0x6e, 0x73
    /*0076*/ .byte 0x74, 0x61, 0x6e, 0x74, 0x30, 0x2e, 0x45, 0x5f
    /*007e*/ .byte 0x33, 0x00

    // .strtab[11] = b'.debug_frame\x00' 
    /*0080*/ .byte 0x2e, 0x64, 0x65, 0x62, 0x75, 0x67, 0x5f, 0x66
    /*0088*/ .byte 0x72, 0x61, 0x6d, 0x65, 0x00

    // .strtab[12] = b'.rel.debug_frame\x00' 
    /*008d*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x2e, 0x64, 0x65, 0x62
    /*0095*/ .byte 0x75, 0x67, 0x5f, 0x66, 0x72, 0x61, 0x6d, 0x65
    /*009d*/ .byte 0x00

    // .strtab[13] = b'.rela.debug_frame\x00' 
    /*009e*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x61, 0x2e, 0x64, 0x65
    /*00a6*/ .byte 0x62, 0x75, 0x67, 0x5f, 0x66, 0x72, 0x61, 0x6d
    /*00ae*/ .byte 0x65, 0x00

    // .strtab[14] = b'.nv.callgraph\x00' 
    /*00b0*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x63, 0x61, 0x6c, 0x6c
    /*00b8*/ .byte 0x67, 0x72, 0x61, 0x70, 0x68, 0x00

    // .strtab[15] = b'.nv.prototype\x00' 
    /*00be*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x70, 0x72, 0x6f, 0x74
    /*00c6*/ .byte 0x6f, 0x74, 0x79, 0x70, 0x65, 0x00

    // .strtab[16] = b'.nv.rel.action\x00' 
    /*00cc*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x72, 0x65, 0x6c, 0x2e
    /*00d4*/ .byte 0x61, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x00

    // .strtab[17] = b'E_3\x00' 
    /*00db*/ .byte 0x45, 0x5f, 0x33, 0x00


// --------------------- .symtab                          --------------------------
	.section  ".symtab", 0, SHT_SYMTAB
	// all symbols in .symtab sections will be kept
	// but the symbol size may be changed accordingly
	.__section_name         0x13 	// offset in .shstrtab
	.__section_type         SHT_SYMTAB
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x200 	// maybe updated by assembler
	.__section_size         0xa8 	// maybe updated by assembler
	.__section_link         2
	.__section_info         0x6
	.__section_entsize      24
	.align                8 	// equivalent to set sh_addralign
    // Symbol[0] "": Container({'st_name': 0, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_NOTYPE'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 'SHN_UNDEF', 'st_value': 0, 'st_size': 0})
    /*0000*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0008*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0010*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[1] ".text.E_3": Container({'st_name': 50, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_SECTION'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 11, 'st_value': 0, 'st_size': 0})
    /*0018*/ .byte 0x32, 0x00, 0x00, 0x00, 0x03, 0x00, 0x0b, 0x00
    /*0020*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0028*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[2] ".nv.constant0.E_3": Container({'st_name': 110, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_SECTION'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 10, 'st_value': 0, 'st_size': 0})
    /*0030*/ .byte 0x6e, 0x00, 0x00, 0x00, 0x03, 0x00, 0x0a, 0x00
    /*0038*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0040*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[3] ".debug_frame": Container({'st_name': 128, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_SECTION'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 4, 'st_value': 0, 'st_size': 0})
    /*0048*/ .byte 0x80, 0x00, 0x00, 0x00, 0x03, 0x00, 0x04, 0x00
    /*0050*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0058*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[4] ".nv.callgraph": Container({'st_name': 176, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_SECTION'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 7, 'st_value': 0, 'st_size': 0})
    /*0060*/ .byte 0xb0, 0x00, 0x00, 0x00, 0x03, 0x00, 0x07, 0x00
    /*0068*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0070*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[5] ".nv.rel.action": Container({'st_name': 204, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_SECTION'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 8, 'st_value': 0, 'st_size': 0})
    /*0078*/ .byte 0xcc, 0x00, 0x00, 0x00, 0x03, 0x00, 0x08, 0x00
    /*0080*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0088*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[6] "E_3": Container({'st_name': 219, 'st_info': Container({'bind': 'STB_GLOBAL', 'type': 'STT_FUNC'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 11, 'st_value': 0, 'st_size': 384})
    /*0090*/ .byte 0xdb, 0x00, 0x00, 0x00, 0x12, 0x10, 0x0b, 0x00
    /*0098*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*00a0*/ .byte 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00


// --------------------- .debug_frame                     --------------------------
	.section	.debug_frame,"",@progbits
	.__section_name         0x80 	// offset in .shstrtab
	.__section_type         SHT_PROGBITS
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x2a8 	// maybe updated by assembler
	.__section_size         0x70 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0x0
	.__section_entsize      0
	.align                1 	// equivalent to set sh_addralign
  .debug_frame:
          /*0000*/ 	.byte	0xff, 0xff, 0xff, 0xff, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
          /*0010*/ 	.byte	0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x04, 0x7c, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x0c, 0x81, 0x80
          /*0020*/ 	.byte	0x80, 0x28, 0x00, 0x08, 0xff, 0x81, 0x80, 0x28, 0x08, 0x81, 0x80, 0x80, 0x28, 0x00, 0x00, 0x00
          /*0030*/ 	.byte	0xff, 0xff, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
          /*0040*/ 	.byte	0x00, 0x00, 0x00, 0x00
          /*0044*/ 	.dword	E_3
          /*004c*/ 	.byte	0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x00, 0x00, 0x00, 0x04, 0x2c, 0x00
          /*005c*/ 	.byte	0x00, 0x00, 0x0c, 0x81, 0x80, 0x80, 0x28, 0x00, 0x04, 0xfc, 0xff, 0xff, 0x3f, 0x00, 0x00, 0x00
          /*006c*/ 	.byte	0x00, 0x00, 0x00, 0x00
  
  
// --------------------- .nv.info                         --------------------------
	.section	.nv.info,"",@"SHT_CUDA_INFO"
	.__section_name         0x29 	// offset in .shstrtab
	.__section_type         1879048192
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x318 	// maybe updated by assembler
	.__section_size         0x24 	// maybe updated by assembler
	.__section_link         3
	.__section_info         0x0
	.__section_entsize      0
	.align                4 	// equivalent to set sh_addralign
  	.align	4
  
  
  	//----- nvinfo : EIATTR_REGCOUNT
  	.align		4
          /*0000*/ 	.byte	0x04, 0x2f
          /*0002*/ 	.short	(.L_1 - .L_0)
  	.align		4
  .L_0:
          /*0004*/ 	.word	index@(E_3)
          /*0008*/ 	.word	0x0000000c
  
  
  	//----- nvinfo : EIATTR_FRAME_SIZE
  	.align		4
  .L_1:
          /*000c*/ 	.byte	0x04, 0x11
          /*000e*/ 	.short	(.L_3 - .L_2)
  	.align		4
  .L_2:
          /*0010*/ 	.word	index@(E_3)
          /*0014*/ 	.word	0x00000000
  
  
  	//----- nvinfo : EIATTR_MIN_STACK_SIZE
  	.align		4
  .L_3:
          /*0018*/ 	.byte	0x04, 0x12
          /*001a*/ 	.short	(.L_5 - .L_4)
  	.align		4
  .L_4:
          /*001c*/ 	.word	index@(E_3)
          /*0020*/ 	.word	0x00000000
  .L_5:
  
  
// --------------------- .nv.info.E_3                     --------------------------
	.section	.nv.info.E_3,"",@"SHT_CUDA_INFO"
	.__section_name         0x3c 	// offset in .shstrtab
	.__section_type         1879048192
	.__section_flags        0x40
	.__section_addr         0x0
	.__section_offset       0x33c 	// maybe updated by assembler
	.__section_size         0x68 	// maybe updated by assembler
	.__section_link         3
	.__section_info         0xb
	.__section_entsize      0
	.align                4 	// equivalent to set sh_addralign
  	.sectionflags	@""
  	.align	4
  
  
  	//----- nvinfo : EIATTR_CUDA_API_VERSION
  	.align		4
          /*0000*/ 	.byte	0x04, 0x37
          /*0002*/ 	.short	(.L_7 - .L_6)
  .L_6:
          /*0004*/ 	.word	0x00000081
  
  
  	//----- nvinfo : EIATTR_SW2861232_WAR
  	.align		4
  .L_7:
          /*0008*/ 	.byte	0x01, 0x35
  	.zero		2
  
  
  	//----- nvinfo : EIATTR_PARAM_CBANK
  	.align		4
          /*000c*/ 	.byte	0x04, 0x0a
          /*000e*/ 	.short	(.L_9 - .L_8)
  	.align		4
  .L_8:
          /*0010*/ 	.word	index@(.nv.constant0.E_3)
          /*0014*/ 	.short	0x0160
          /*0016*/ 	.short	0x0018
  
  
  	//----- nvinfo : EIATTR_CBANK_PARAM_SIZE
  	.align		4
  .L_9:
          /*0018*/ 	.byte	0x03, 0x19
          /*001a*/ 	.short	0x0018
  
  
  	//----- nvinfo : EIATTR_KPARAM_INFO
  	.align		4
          /*001c*/ 	.byte	0x04, 0x17
          /*001e*/ 	.short	(.L_11 - .L_10)
  .L_10:
          /*0020*/ 	.word	0x00000000
          /*0024*/ 	.short	0x0002
          /*0026*/ 	.short	0x0010
          /*0028*/ 	.byte	0x00, 0xf0, 0x21, 0x00
  
  
  	//----- nvinfo : EIATTR_KPARAM_INFO
  	.align		4
  .L_11:
          /*002c*/ 	.byte	0x04, 0x17
          /*002e*/ 	.short	(.L_13 - .L_12)
  .L_12:
          /*0030*/ 	.word	0x00000000
          /*0034*/ 	.short	0x0001
          /*0036*/ 	.short	0x0008
          /*0038*/ 	.byte	0x00, 0xf0, 0x21, 0x00
  
  
  	//----- nvinfo : EIATTR_KPARAM_INFO
  	.align		4
  .L_13:
          /*003c*/ 	.byte	0x04, 0x17
          /*003e*/ 	.short	(.L_15 - .L_14)
  .L_14:
          /*0040*/ 	.word	0x00000000
          /*0044*/ 	.short	0x0000
          /*0046*/ 	.short	0x0000
          /*0048*/ 	.byte	0x00, 0xf0, 0x21, 0x00
  
  
  	//----- nvinfo : EIATTR_MAXREG_COUNT
  	.align		4
  .L_15:
          /*004c*/ 	.byte	0x03, 0x1b
          /*004e*/ 	.short	0x00ff
  
  
  	//----- nvinfo : EIATTR_EXIT_INSTR_OFFSETS
  	.align		4
          /*0050*/ 	.byte	0x04, 0x1c
          /*0052*/ 	.short	(.L_17 - .L_16)
  
  
  	//   ....[0]....
  .L_16:
          /*0054*/ 	.word	0x000000b0
  
  
  	//----- nvinfo : EIATTR_MAX_THREADS
  	.align		4
  .L_17:
          /*0058*/ 	.byte	0x04, 0x05
          /*005a*/ 	.short	(.L_19 - .L_18)
  .L_18:
          /*005c*/ 	.word	0x00000003
          /*0060*/ 	.word	0x00000001
          /*0064*/ 	.word	0x00000001
  .L_19:
  
  
// --------------------- .nv.callgraph                    --------------------------
	.section	.nv.callgraph,"",@"SHT_CUDA_CALLGRAPH"
	.__section_name         0xb0 	// offset in .shstrtab
	.__section_type         1879048193
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x3a4 	// maybe updated by assembler
	.__section_size         0x20 	// maybe updated by assembler
	.__section_link         3
	.__section_info         0x0
	.__section_entsize      8
	.align                4 	// equivalent to set sh_addralign
  	.align	4
  	.sectionentsize	8
  	.align		4
          /*0000*/ 	.word	0x00000000
  	.align		4
          /*0004*/ 	.word	0xffffffff
  	.align		4
          /*0008*/ 	.word	0x00000000
  	.align		4
          /*000c*/ 	.word	0xfffffffe
  	.align		4
          /*0010*/ 	.word	0x00000000
  	.align		4
          /*0014*/ 	.word	0xfffffffd
  	.align		4
          /*0018*/ 	.word	0x00000000
  	.align		4
          /*001c*/ 	.word	0xfffffffc
  
  
// --------------------- .nv.rel.action                   --------------------------
	.section	.nv.rel.action,"",@"SHT_CUDA_RELOCINFO"
	.__section_name         0xcc 	// offset in .shstrtab
	.__section_type         1879048203
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x3c8 	// maybe updated by assembler
	.__section_size         0x10 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0x0
	.__section_entsize      8
	.align                8 	// equivalent to set sh_addralign
  	.align	8
  	.sectionentsize	8
          /*0000*/ 	.byte	0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x25, 0x00, 0x05, 0x36
  
  
// --------------------- .rel.debug_frame                 --------------------------
	.section  ".rel.debug_frame", 64, SHT_REL
	// all relocation sections will be dynamically generated by assembler 
	// but most of the section header will be kept as is.
	.__section_name         0x8d 	// offset in .shstrtab
	.__section_type         SHT_REL
	.__section_flags        0x40
	.__section_addr         0x0
	.__section_offset       0x3d8 	// maybe updated by assembler
	.__section_size         0x10 	// maybe updated by assembler
	.__section_link         3
	.__section_info         0x4
	.__section_entsize      16
	.align                8 	// equivalent to set sh_addralign
    // Relocation[0] : E_3, Container({'r_offset': 68, 'r_info': 25769803778, 'r_info_sym': 6, 'r_info_type': 2})

// --------------------- .nv.constant0.E_3                --------------------------
	.section	.nv.constant0.E_3,"a",@progbits
	.__section_name         0x58 	// offset in .shstrtab
	.__section_type         SHT_PROGBITS
	.__section_flags        0x42
	.__section_addr         0x0
	.__section_offset       0x3e8 	// maybe updated by assembler
	.__section_size         0x178 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0xb
	.__section_entsize      0
	.align                4 	// equivalent to set sh_addralign
  	.sectionflags	@""
  	.align	4
  .nv.constant0.E_3:
  	.zero		376
  
  
// --------------------- .text.E_3                        --------------------------
	.section	.text.E_3,"ax",@progbits
	.__section_name         0x32 	// offset in .shstrtab
	.__section_type         SHT_PROGBITS
	.__section_flags        0x6
	.__section_addr         0x0
	.__section_offset       0x580 	// maybe updated by assembler
	.__section_size         0x180 	// maybe updated by assembler
	.__section_link         3
	.__section_info         0xc000006
	.__section_entsize      0
	.align                128 	// equivalent to set sh_addralign
  	.sectioninfo	@"SHI_REGISTERS=12"
  	.align	128
          .global         E_3
          .type           E_3,@function
          .size           E_3,(.L_x_1 - E_3)
          .other          E_3,@"STO_CUDA_ENTRY STV_DEFAULT"
  E_3:
  .text.E_3:
      [B------:R-:W-:-:S02]         /*0000*/                   MOV R1, c[0x0][0x28] ;
      [B------:R-:W0:-:S01]         /*0010*/                   S2R R6, SR_TID.X ;
      [B------:R-:W-:-:S01]         /*0020*/                   MOV R7, 0x4 ;
      [B------:R-:W-:Y:S04]         /*0030*/                   ULDC.64 UR4, c[0x0][0x118] ;
      [B0-----:R-:W-:Y:S04]         /*0040*/                   IMAD.WIDE R2, R6, R7, c[0x0][0x168] ;
      [B------:R-:W-:-:S02]         /*0050*/                   IMAD.WIDE R4, R6.reuse, R7.reuse, c[0x0][0x170] ;
      [B------:R-:W2:-:S04]         /*0060*/                   LDG.E R2, desc[UR4][R2.64] ;
      [B------:R-:W2:-:S01]         /*0070*/                   LDG.E R5, desc[UR4][R4.64] ;
      [B------:R-:W-:-:S01]         /*0080*/                   IMAD.WIDE R6, R6, R7, c[0x0][0x160] ;
      [B--2---:R-:W-:Y:S05]         /*0090*/                   FMUL R9, R2, R5 ;
      [B------:R-:W-:-:S01]         /*00a0*/                   STG.E desc[UR4][R6.64], R9 ;
      [B------:R-:W-:-:S05]         /*00b0*/                   EXIT ;
  .L_x_0:
      [B------:R-:W-:Y:S00]         /*00c0*/                   BRA `(.L_x_0);
      [B------:R-:W-:Y:S00]         /*00d0*/                   NOP;
      [B------:R-:W-:Y:S00]         /*00e0*/                   NOP;
      [B------:R-:W-:Y:S00]         /*00f0*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0100*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0110*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0120*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0130*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0140*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0150*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0160*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0170*/                   NOP;
  .L_x_1:

  //-------------------------------------------------
  //---------------- END of sections ----------------
  //-------------------------------------------------


// Program segment PT_PHDR, 5 
  .__segment  "PT_PHDR", 5 
  .__segment_offset  0xa00   		// maybe updated by assembler 
  .__segment_vaddr   0x0   		// Seems always 0? 
  .__segment_paddr   0x0   		// ??? 
  .__segment_filesz  0xa8   		// file size, maybe updated by assembler 
  .__segment_memsz   0xa8   		// file size + nobits sections, maybe updated by assembler 
  .__segment_align     8   		//  

// Program segment PT_LOAD, 5 
  .__segment  "PT_LOAD", 5 
  .__segment_offset  0x3e8   		// maybe updated by assembler 
  .__segment_vaddr   0x0   		// Seems always 0? 
  .__segment_paddr   0x0   		// ??? 
  .__segment_filesz  0x318   		// file size, maybe updated by assembler 
  .__segment_memsz   0x318   		// file size + nobits sections, maybe updated by assembler 
  .__segment_align     8   		//  
  .__segment_startsection    ".nv.constant0.E_3"  		// first section in this segment 
  .__segment_endsection      ".text.E_3"  		// last  section in this segment 

// Program segment PT_LOAD, 5 
  .__segment  "PT_LOAD", 5 
  .__segment_offset  0xa00   		// maybe updated by assembler 
  .__segment_vaddr   0x0   		// Seems always 0? 
  .__segment_paddr   0x0   		// ??? 
  .__segment_filesz  0xa8   		// file size, maybe updated by assembler 
  .__segment_memsz   0xa8   		// file size + nobits sections, maybe updated by assembler 
  .__segment_align     8   		//  
  .__segment_startsection    "@PROGRAM_HEADER"  		// first section in this segment 
  .__segment_endsection      "@PROGRAM_HEADER"  		// last  section in this segment 


  //-------------------------------------------------
  //---------------- END of segments ----------------
  //-------------------------------------------------


    """
    return ret

  def render2(self, uops:list[UOp]) -> str:
    elf_header: dict[str, str] = {
      ".__elf_ident_osabi": "51",
      ".__elf_ident_abiversion": "7",
      ".__elf_type": "ET_EXEC",
      ".__elf_machine": "EM_CUDA",
      ".__elf_version": "129",
      ".__elf_entry": "0",
      ".__elf_phoff": "0xa00",
      ".__elf_shoff": "0x700",
      ".__elf_flags": "0x560556",
      ".__elf_ehsize": "64",
      ".__elf_phentsize": "56",
      ".__elf_phnum": "3",
      ".__elf_shentsize": "64",
      ".__elf_shnum": "12",
      ".__elf_shstrndx": "1",
    }
    ret = ""
    ret += dict_to_str(elf_header)
    ret += """
// --------------------- FileHeader --------------------------
	// All file header info is kept as is (unless offset/size attributes)
	// The original header flags is not complete, thus discarded. 
	// 	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM86 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM86)"
	// 	.elftype	@"ET_EXEC"
	// 
	// 
	//.__elf_ident_osabi      51
	//.__elf_ident_abiversion 7
	//.__elf_type             ET_EXEC
	//.__elf_machine          EM_CUDA
	//.__elf_version          129 		// CUDA toolkit version 
	//.__elf_entry            0 		// entry point address 
	//.__elf_phoff            0xa00 		// program header offset, maybe updated by assembler
	//.__elf_shoff            0x700 		// section header offset, maybe updated by assembler
	//.__elf_flags            0x560556 		// Flags, SM_86(0x56), COMPUTE_86(0x56) 
	//.__elf_ehsize           64 		// elf header size 
	//.__elf_phentsize        56 		// program entry size
	//.__elf_phnum            3 		// number of program entries
	//.__elf_shentsize        64 		// section entry size
	//.__elf_shnum            12 		// number of sections, currently no sections can be appended/removed
	//.__elf_shstrndx         1 		// Section name string table index 


  //-------------------------------------------------
  //------------ END of FileHeader ------------------
  //-------------------------------------------------



// ---------------------                                  --------------------------
	// there will always be an empty section at index 0
	.section  "", 0, SHT_NULL
	.__section_name         0x0 	// offset in .shstrtab
	.__section_type         SHT_NULL
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x0 	// maybe updated by assembler
	.__section_size         0x0 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0x0
	.__section_entsize      0
	.align                0 	// equivalent to set sh_addralign

// --------------------- .shstrtab                        --------------------------
	.section  ".shstrtab", 0, SHT_STRTAB
	// all strings in .shstrtab section will be kept as is.
	.__section_name         0x1 	// offset in .shstrtab
	.__section_type         SHT_STRTAB
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x40 	// maybe updated by assembler
	.__section_size         0xdb 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0x0
	.__section_entsize      0
	.align                1 	// equivalent to set sh_addralign
    // .shstrtab[0] = b'\x00' 
    /*0000*/ .byte 0x00

    // .shstrtab[1] = b'.shstrtab\x00' 
    /*0001*/ .byte 0x2e, 0x73, 0x68, 0x73, 0x74, 0x72, 0x74, 0x61
    /*0009*/ .byte 0x62, 0x00

    // .shstrtab[2] = b'.strtab\x00' 
    /*000b*/ .byte 0x2e, 0x73, 0x74, 0x72, 0x74, 0x61, 0x62, 0x00

    // .shstrtab[3] = b'.symtab\x00' 
    /*0013*/ .byte 0x2e, 0x73, 0x79, 0x6d, 0x74, 0x61, 0x62, 0x00

    // .shstrtab[4] = b'.symtab_shndx\x00' 
    /*001b*/ .byte 0x2e, 0x73, 0x79, 0x6d, 0x74, 0x61, 0x62, 0x5f
    /*0023*/ .byte 0x73, 0x68, 0x6e, 0x64, 0x78, 0x00

    // .shstrtab[5] = b'.nv.info\x00' 
    /*0029*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x69, 0x6e, 0x66, 0x6f
    /*0031*/ .byte 0x00

    // .shstrtab[6] = b'.text.E_3\x00' 
    /*0032*/ .byte 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x45, 0x5f
    /*003a*/ .byte 0x33, 0x00

    // .shstrtab[7] = b'.nv.info.E_3\x00' 
    /*003c*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x69, 0x6e, 0x66, 0x6f
    /*0044*/ .byte 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .shstrtab[8] = b'.nv.shared.E_3\x00' 
    /*0049*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x73, 0x68, 0x61, 0x72
    /*0051*/ .byte 0x65, 0x64, 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .shstrtab[9] = b'.nv.constant0.E_3\x00' 
    /*0058*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x63, 0x6f, 0x6e, 0x73
    /*0060*/ .byte 0x74, 0x61, 0x6e, 0x74, 0x30, 0x2e, 0x45, 0x5f
    /*0068*/ .byte 0x33, 0x00

    // .shstrtab[10] = b'.rel.nv.constant0.E_3\x00' 
    /*006a*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x2e, 0x6e, 0x76, 0x2e
    /*0072*/ .byte 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74
    /*007a*/ .byte 0x30, 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .shstrtab[11] = b'.debug_frame\x00' 
    /*0080*/ .byte 0x2e, 0x64, 0x65, 0x62, 0x75, 0x67, 0x5f, 0x66
    /*0088*/ .byte 0x72, 0x61, 0x6d, 0x65, 0x00

    // .shstrtab[12] = b'.rel.debug_frame\x00' 
    /*008d*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x2e, 0x64, 0x65, 0x62
    /*0095*/ .byte 0x75, 0x67, 0x5f, 0x66, 0x72, 0x61, 0x6d, 0x65
    /*009d*/ .byte 0x00

    // .shstrtab[13] = b'.rela.debug_frame\x00' 
    /*009e*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x61, 0x2e, 0x64, 0x65
    /*00a6*/ .byte 0x62, 0x75, 0x67, 0x5f, 0x66, 0x72, 0x61, 0x6d
    /*00ae*/ .byte 0x65, 0x00

    // .shstrtab[14] = b'.nv.callgraph\x00' 
    /*00b0*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x63, 0x61, 0x6c, 0x6c
    /*00b8*/ .byte 0x67, 0x72, 0x61, 0x70, 0x68, 0x00

    // .shstrtab[15] = b'.nv.prototype\x00' 
    /*00be*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x70, 0x72, 0x6f, 0x74
    /*00c6*/ .byte 0x6f, 0x74, 0x79, 0x70, 0x65, 0x00

    // .shstrtab[16] = b'.nv.rel.action\x00' 
    /*00cc*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x72, 0x65, 0x6c, 0x2e
    /*00d4*/ .byte 0x61, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x00


// --------------------- .strtab                          --------------------------
	.section  ".strtab", 0, SHT_STRTAB
	// all strings in .strtab section will be kept as is.
	.__section_name         0xb 	// offset in .shstrtab
	.__section_type         SHT_STRTAB
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x11b 	// maybe updated by assembler
	.__section_size         0xdf 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0x0
	.__section_entsize      0
	.align                1 	// equivalent to set sh_addralign
    // .strtab[0] = b'\x00' 
    /*0000*/ .byte 0x00

    // .strtab[1] = b'.shstrtab\x00' 
    /*0001*/ .byte 0x2e, 0x73, 0x68, 0x73, 0x74, 0x72, 0x74, 0x61
    /*0009*/ .byte 0x62, 0x00

    // .strtab[2] = b'.strtab\x00' 
    /*000b*/ .byte 0x2e, 0x73, 0x74, 0x72, 0x74, 0x61, 0x62, 0x00

    // .strtab[3] = b'.symtab\x00' 
    /*0013*/ .byte 0x2e, 0x73, 0x79, 0x6d, 0x74, 0x61, 0x62, 0x00

    // .strtab[4] = b'.symtab_shndx\x00' 
    /*001b*/ .byte 0x2e, 0x73, 0x79, 0x6d, 0x74, 0x61, 0x62, 0x5f
    /*0023*/ .byte 0x73, 0x68, 0x6e, 0x64, 0x78, 0x00

    // .strtab[5] = b'.nv.info\x00' 
    /*0029*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x69, 0x6e, 0x66, 0x6f
    /*0031*/ .byte 0x00

    // .strtab[6] = b'.text.E_3\x00' 
    /*0032*/ .byte 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x45, 0x5f
    /*003a*/ .byte 0x33, 0x00

    // .strtab[7] = b'.nv.info.E_3\x00' 
    /*003c*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x69, 0x6e, 0x66, 0x6f
    /*0044*/ .byte 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .strtab[8] = b'.nv.shared.E_3\x00' 
    /*0049*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x73, 0x68, 0x61, 0x72
    /*0051*/ .byte 0x65, 0x64, 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .strtab[9] = b'.rel.nv.constant0.E_3\x00' 
    /*0058*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x2e, 0x6e, 0x76, 0x2e
    /*0060*/ .byte 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74
    /*0068*/ .byte 0x30, 0x2e, 0x45, 0x5f, 0x33, 0x00

    // .strtab[10] = b'.nv.constant0.E_3\x00' 
    /*006e*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x63, 0x6f, 0x6e, 0x73
    /*0076*/ .byte 0x74, 0x61, 0x6e, 0x74, 0x30, 0x2e, 0x45, 0x5f
    /*007e*/ .byte 0x33, 0x00

    // .strtab[11] = b'.debug_frame\x00' 
    /*0080*/ .byte 0x2e, 0x64, 0x65, 0x62, 0x75, 0x67, 0x5f, 0x66
    /*0088*/ .byte 0x72, 0x61, 0x6d, 0x65, 0x00

    // .strtab[12] = b'.rel.debug_frame\x00' 
    /*008d*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x2e, 0x64, 0x65, 0x62
    /*0095*/ .byte 0x75, 0x67, 0x5f, 0x66, 0x72, 0x61, 0x6d, 0x65
    /*009d*/ .byte 0x00

    // .strtab[13] = b'.rela.debug_frame\x00' 
    /*009e*/ .byte 0x2e, 0x72, 0x65, 0x6c, 0x61, 0x2e, 0x64, 0x65
    /*00a6*/ .byte 0x62, 0x75, 0x67, 0x5f, 0x66, 0x72, 0x61, 0x6d
    /*00ae*/ .byte 0x65, 0x00

    // .strtab[14] = b'.nv.callgraph\x00' 
    /*00b0*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x63, 0x61, 0x6c, 0x6c
    /*00b8*/ .byte 0x67, 0x72, 0x61, 0x70, 0x68, 0x00

    // .strtab[15] = b'.nv.prototype\x00' 
    /*00be*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x70, 0x72, 0x6f, 0x74
    /*00c6*/ .byte 0x6f, 0x74, 0x79, 0x70, 0x65, 0x00

    // .strtab[16] = b'.nv.rel.action\x00' 
    /*00cc*/ .byte 0x2e, 0x6e, 0x76, 0x2e, 0x72, 0x65, 0x6c, 0x2e
    /*00d4*/ .byte 0x61, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x00

    // .strtab[17] = b'E_3\x00' 
    /*00db*/ .byte 0x45, 0x5f, 0x33, 0x00


// --------------------- .symtab                          --------------------------
	.section  ".symtab", 0, SHT_SYMTAB
	// all symbols in .symtab sections will be kept
	// but the symbol size may be changed accordingly
	.__section_name         0x13 	// offset in .shstrtab
	.__section_type         SHT_SYMTAB
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x200 	// maybe updated by assembler
	.__section_size         0xa8 	// maybe updated by assembler
	.__section_link         2
	.__section_info         0x6
	.__section_entsize      24
	.align                8 	// equivalent to set sh_addralign
    // Symbol[0] "": Container({'st_name': 0, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_NOTYPE'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 'SHN_UNDEF', 'st_value': 0, 'st_size': 0})
    /*0000*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0008*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0010*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[1] ".text.E_3": Container({'st_name': 50, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_SECTION'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 11, 'st_value': 0, 'st_size': 0})
    /*0018*/ .byte 0x32, 0x00, 0x00, 0x00, 0x03, 0x00, 0x0b, 0x00
    /*0020*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0028*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[2] ".nv.constant0.E_3": Container({'st_name': 110, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_SECTION'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 10, 'st_value': 0, 'st_size': 0})
    /*0030*/ .byte 0x6e, 0x00, 0x00, 0x00, 0x03, 0x00, 0x0a, 0x00
    /*0038*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0040*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[3] ".debug_frame": Container({'st_name': 128, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_SECTION'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 4, 'st_value': 0, 'st_size': 0})
    /*0048*/ .byte 0x80, 0x00, 0x00, 0x00, 0x03, 0x00, 0x04, 0x00
    /*0050*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0058*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[4] ".nv.callgraph": Container({'st_name': 176, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_SECTION'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 7, 'st_value': 0, 'st_size': 0})
    /*0060*/ .byte 0xb0, 0x00, 0x00, 0x00, 0x03, 0x00, 0x07, 0x00
    /*0068*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0070*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[5] ".nv.rel.action": Container({'st_name': 204, 'st_info': Container({'bind': 'STB_LOCAL', 'type': 'STT_SECTION'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 8, 'st_value': 0, 'st_size': 0})
    /*0078*/ .byte 0xcc, 0x00, 0x00, 0x00, 0x03, 0x00, 0x08, 0x00
    /*0080*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*0088*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    // Symbol[6] "E_3": Container({'st_name': 219, 'st_info': Container({'bind': 'STB_GLOBAL', 'type': 'STT_FUNC'}), 'st_other': Container({'local': 0, 'visibility': 'STV_DEFAULT'}), 'st_shndx': 11, 'st_value': 0, 'st_size': 384})
    /*0090*/ .byte 0xdb, 0x00, 0x00, 0x00, 0x12, 0x10, 0x0b, 0x00
    /*0098*/ .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    /*00a0*/ .byte 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00


// --------------------- .debug_frame                     --------------------------
	.section	.debug_frame,"",@progbits
	.__section_name         0x80 	// offset in .shstrtab
	.__section_type         SHT_PROGBITS
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x2a8 	// maybe updated by assembler
	.__section_size         0x70 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0x0
	.__section_entsize      0
	.align                1 	// equivalent to set sh_addralign
  .debug_frame:
          /*0000*/ 	.byte	0xff, 0xff, 0xff, 0xff, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
          /*0010*/ 	.byte	0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x04, 0x7c, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x0c, 0x81, 0x80
          /*0020*/ 	.byte	0x80, 0x28, 0x00, 0x08, 0xff, 0x81, 0x80, 0x28, 0x08, 0x81, 0x80, 0x80, 0x28, 0x00, 0x00, 0x00
          /*0030*/ 	.byte	0xff, 0xff, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
          /*0040*/ 	.byte	0x00, 0x00, 0x00, 0x00
          /*0044*/ 	.dword	E_3
          /*004c*/ 	.byte	0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x00, 0x00, 0x00, 0x04, 0x2c, 0x00
          /*005c*/ 	.byte	0x00, 0x00, 0x0c, 0x81, 0x80, 0x80, 0x28, 0x00, 0x04, 0xfc, 0xff, 0xff, 0x3f, 0x00, 0x00, 0x00
          /*006c*/ 	.byte	0x00, 0x00, 0x00, 0x00
  
  
// --------------------- .nv.info                         --------------------------
	.section	.nv.info,"",@"SHT_CUDA_INFO"
	.__section_name         0x29 	// offset in .shstrtab
	.__section_type         1879048192
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x318 	// maybe updated by assembler
	.__section_size         0x24 	// maybe updated by assembler
	.__section_link         3
	.__section_info         0x0
	.__section_entsize      0
	.align                4 	// equivalent to set sh_addralign
  	.align	4
  
  
  	//----- nvinfo : EIATTR_REGCOUNT
  	.align		4
          /*0000*/ 	.byte	0x04, 0x2f
          /*0002*/ 	.short	(.L_1 - .L_0)
  	.align		4
  .L_0:
          /*0004*/ 	.word	index@(E_3)
          /*0008*/ 	.word	0x0000000c
  
  
  	//----- nvinfo : EIATTR_FRAME_SIZE
  	.align		4
  .L_1:
          /*000c*/ 	.byte	0x04, 0x11
          /*000e*/ 	.short	(.L_3 - .L_2)
  	.align		4
  .L_2:
          /*0010*/ 	.word	index@(E_3)
          /*0014*/ 	.word	0x00000000
  
  
  	//----- nvinfo : EIATTR_MIN_STACK_SIZE
  	.align		4
  .L_3:
          /*0018*/ 	.byte	0x04, 0x12
          /*001a*/ 	.short	(.L_5 - .L_4)
  	.align		4
  .L_4:
          /*001c*/ 	.word	index@(E_3)
          /*0020*/ 	.word	0x00000000
  .L_5:
  
  
// --------------------- .nv.info.E_3                     --------------------------
	.section	.nv.info.E_3,"",@"SHT_CUDA_INFO"
	.__section_name         0x3c 	// offset in .shstrtab
	.__section_type         1879048192
	.__section_flags        0x40
	.__section_addr         0x0
	.__section_offset       0x33c 	// maybe updated by assembler
	.__section_size         0x68 	// maybe updated by assembler
	.__section_link         3
	.__section_info         0xb
	.__section_entsize      0
	.align                4 	// equivalent to set sh_addralign
  	.sectionflags	@""
  	.align	4
  
  
  	//----- nvinfo : EIATTR_CUDA_API_VERSION
  	.align		4
          /*0000*/ 	.byte	0x04, 0x37
          /*0002*/ 	.short	(.L_7 - .L_6)
  .L_6:
          /*0004*/ 	.word	0x00000081
  
  
  	//----- nvinfo : EIATTR_SW2861232_WAR
  	.align		4
  .L_7:
          /*0008*/ 	.byte	0x01, 0x35
  	.zero		2
  
  
  	//----- nvinfo : EIATTR_PARAM_CBANK
  	.align		4
          /*000c*/ 	.byte	0x04, 0x0a
          /*000e*/ 	.short	(.L_9 - .L_8)
  	.align		4
  .L_8:
          /*0010*/ 	.word	index@(.nv.constant0.E_3)
          /*0014*/ 	.short	0x0160
          /*0016*/ 	.short	0x0018
  
  
  	//----- nvinfo : EIATTR_CBANK_PARAM_SIZE
  	.align		4
  .L_9:
          /*0018*/ 	.byte	0x03, 0x19
          /*001a*/ 	.short	0x0018
  
  
  	//----- nvinfo : EIATTR_KPARAM_INFO
  	.align		4
          /*001c*/ 	.byte	0x04, 0x17
          /*001e*/ 	.short	(.L_11 - .L_10)
  .L_10:
          /*0020*/ 	.word	0x00000000
          /*0024*/ 	.short	0x0002
          /*0026*/ 	.short	0x0010
          /*0028*/ 	.byte	0x00, 0xf0, 0x21, 0x00
  
  
  	//----- nvinfo : EIATTR_KPARAM_INFO
  	.align		4
  .L_11:
          /*002c*/ 	.byte	0x04, 0x17
          /*002e*/ 	.short	(.L_13 - .L_12)
  .L_12:
          /*0030*/ 	.word	0x00000000
          /*0034*/ 	.short	0x0001
          /*0036*/ 	.short	0x0008
          /*0038*/ 	.byte	0x00, 0xf0, 0x21, 0x00
  
  
  	//----- nvinfo : EIATTR_KPARAM_INFO
  	.align		4
  .L_13:
          /*003c*/ 	.byte	0x04, 0x17
          /*003e*/ 	.short	(.L_15 - .L_14)
  .L_14:
          /*0040*/ 	.word	0x00000000
          /*0044*/ 	.short	0x0000
          /*0046*/ 	.short	0x0000
          /*0048*/ 	.byte	0x00, 0xf0, 0x21, 0x00
  
  
  	//----- nvinfo : EIATTR_MAXREG_COUNT
  	.align		4
  .L_15:
          /*004c*/ 	.byte	0x03, 0x1b
          /*004e*/ 	.short	0x00ff
  
  
  	//----- nvinfo : EIATTR_EXIT_INSTR_OFFSETS
  	.align		4
          /*0050*/ 	.byte	0x04, 0x1c
          /*0052*/ 	.short	(.L_17 - .L_16)
  
  
  	//   ....[0]....
  .L_16:
          /*0054*/ 	.word	0x000000b0
  
  
  	//----- nvinfo : EIATTR_MAX_THREADS
  	.align		4
  .L_17:
          /*0058*/ 	.byte	0x04, 0x05
          /*005a*/ 	.short	(.L_19 - .L_18)
  .L_18:
          /*005c*/ 	.word	0x00000003
          /*0060*/ 	.word	0x00000001
          /*0064*/ 	.word	0x00000001
  .L_19:
  
  
// --------------------- .nv.callgraph                    --------------------------
	.section	.nv.callgraph,"",@"SHT_CUDA_CALLGRAPH"
	.__section_name         0xb0 	// offset in .shstrtab
	.__section_type         1879048193
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x3a4 	// maybe updated by assembler
	.__section_size         0x20 	// maybe updated by assembler
	.__section_link         3
	.__section_info         0x0
	.__section_entsize      8
	.align                4 	// equivalent to set sh_addralign
  	.align	4
  	.sectionentsize	8
  	.align		4
          /*0000*/ 	.word	0x00000000
  	.align		4
          /*0004*/ 	.word	0xffffffff
  	.align		4
          /*0008*/ 	.word	0x00000000
  	.align		4
          /*000c*/ 	.word	0xfffffffe
  	.align		4
          /*0010*/ 	.word	0x00000000
  	.align		4
          /*0014*/ 	.word	0xfffffffd
  	.align		4
          /*0018*/ 	.word	0x00000000
  	.align		4
          /*001c*/ 	.word	0xfffffffc
  
  
// --------------------- .nv.rel.action                   --------------------------
	.section	.nv.rel.action,"",@"SHT_CUDA_RELOCINFO"
	.__section_name         0xcc 	// offset in .shstrtab
	.__section_type         1879048203
	.__section_flags        0x0
	.__section_addr         0x0
	.__section_offset       0x3c8 	// maybe updated by assembler
	.__section_size         0x10 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0x0
	.__section_entsize      8
	.align                8 	// equivalent to set sh_addralign
  	.align	8
  	.sectionentsize	8
          /*0000*/ 	.byte	0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x25, 0x00, 0x05, 0x36
  
  
// --------------------- .rel.debug_frame                 --------------------------
	.section  ".rel.debug_frame", 64, SHT_REL
	// all relocation sections will be dynamically generated by assembler 
	// but most of the section header will be kept as is.
	.__section_name         0x8d 	// offset in .shstrtab
	.__section_type         SHT_REL
	.__section_flags        0x40
	.__section_addr         0x0
	.__section_offset       0x3d8 	// maybe updated by assembler
	.__section_size         0x10 	// maybe updated by assembler
	.__section_link         3
	.__section_info         0x4
	.__section_entsize      16
	.align                8 	// equivalent to set sh_addralign
    // Relocation[0] : E_3, Container({'r_offset': 68, 'r_info': 25769803778, 'r_info_sym': 6, 'r_info_type': 2})

// --------------------- .nv.constant0.E_3                --------------------------
	.section	.nv.constant0.E_3,"a",@progbits
	.__section_name         0x58 	// offset in .shstrtab
	.__section_type         SHT_PROGBITS
	.__section_flags        0x42
	.__section_addr         0x0
	.__section_offset       0x3e8 	// maybe updated by assembler
	.__section_size         0x178 	// maybe updated by assembler
	.__section_link         0
	.__section_info         0xb
	.__section_entsize      0
	.align                4 	// equivalent to set sh_addralign
  	.sectionflags	@""
  	.align	4
  .nv.constant0.E_3:
  	.zero		376
  
  
// --------------------- .text.E_3                        --------------------------
	.section	.text.E_3,"ax",@progbits
	.__section_name         0x32 	// offset in .shstrtab
	.__section_type         SHT_PROGBITS
	.__section_flags        0x6
	.__section_addr         0x0
	.__section_offset       0x580 	// maybe updated by assembler
	.__section_size         0x180 	// maybe updated by assembler
	.__section_link         3
	.__section_info         0xc000006
	.__section_entsize      0
	.align                128 	// equivalent to set sh_addralign
  	.sectioninfo	@"SHI_REGISTERS=12"
  	.align	128
          .global         E_3
          .type           E_3,@function
          .size           E_3,(.L_x_1 - E_3)
          .other          E_3,@"STO_CUDA_ENTRY STV_DEFAULT"
  E_3:
  .text.E_3:
      [B------:R-:W-:-:S02]         /*0000*/                   MOV R1, c[0x0][0x28] ;
      [B------:R-:W0:-:S01]         /*0010*/                   S2R R6, SR_TID.X ;
      [B------:R-:W-:-:S01]         /*0020*/                   MOV R7, 0x4 ;
      [B------:R-:W-:Y:S04]         /*0030*/                   ULDC.64 UR4, c[0x0][0x118] ;
      [B0-----:R-:W-:Y:S04]         /*0040*/                   IMAD.WIDE R2, R6, R7, c[0x0][0x168] ;
      [B------:R-:W-:-:S02]         /*0050*/                   IMAD.WIDE R4, R6.reuse, R7.reuse, c[0x0][0x170] ;
      [B------:R-:W2:-:S04]         /*0060*/                   LDG.E R2, desc[UR4][R2.64] ;
      [B------:R-:W2:-:S01]         /*0070*/                   LDG.E R5, desc[UR4][R4.64] ;
      [B------:R-:W-:-:S01]         /*0080*/                   IMAD.WIDE R6, R6, R7, c[0x0][0x160] ;
      [B--2---:R-:W-:Y:S05]         /*0090*/                   FMUL R9, R2, R5 ;
      [B------:R-:W-:-:S01]         /*00a0*/                   STG.E desc[UR4][R6.64], R9 ;
      [B------:R-:W-:-:S05]         /*00b0*/                   EXIT ;
  .L_x_0:
      [B------:R-:W-:Y:S00]         /*00c0*/                   BRA `(.L_x_0);
      [B------:R-:W-:Y:S00]         /*00d0*/                   NOP;
      [B------:R-:W-:Y:S00]         /*00e0*/                   NOP;
      [B------:R-:W-:Y:S00]         /*00f0*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0100*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0110*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0120*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0130*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0140*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0150*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0160*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0170*/                   NOP;
  .L_x_1:

  //-------------------------------------------------
  //---------------- END of sections ----------------
  //-------------------------------------------------


// Program segment PT_PHDR, 5 
  .__segment  "PT_PHDR", 5 
  .__segment_offset  0xa00   		// maybe updated by assembler 
  .__segment_vaddr   0x0   		// Seems always 0? 
  .__segment_paddr   0x0   		// ??? 
  .__segment_filesz  0xa8   		// file size, maybe updated by assembler 
  .__segment_memsz   0xa8   		// file size + nobits sections, maybe updated by assembler 
  .__segment_align     8   		//  

// Program segment PT_LOAD, 5 
  .__segment  "PT_LOAD", 5 
  .__segment_offset  0x3e8   		// maybe updated by assembler 
  .__segment_vaddr   0x0   		// Seems always 0? 
  .__segment_paddr   0x0   		// ??? 
  .__segment_filesz  0x318   		// file size, maybe updated by assembler 
  .__segment_memsz   0x318   		// file size + nobits sections, maybe updated by assembler 
  .__segment_align     8   		//  
  .__segment_startsection    ".nv.constant0.E_3"  		// first section in this segment 
  .__segment_endsection      ".text.E_3"  		// last  section in this segment 

// Program segment PT_LOAD, 5 
  .__segment  "PT_LOAD", 5 
  .__segment_offset  0xa00   		// maybe updated by assembler 
  .__segment_vaddr   0x0   		// Seems always 0? 
  .__segment_paddr   0x0   		// ??? 
  .__segment_filesz  0xa8   		// file size, maybe updated by assembler 
  .__segment_memsz   0xa8   		// file size + nobits sections, maybe updated by assembler 
  .__segment_align     8   		//  
  .__segment_startsection    "@PROGRAM_HEADER"  		// first section in this segment 
  .__segment_endsection      "@PROGRAM_HEADER"  		// last  section in this segment 


  //-------------------------------------------------
  //---------------- END of segments ----------------
  //-------------------------------------------------


    """
    return ret
