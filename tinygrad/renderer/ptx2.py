from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.ptx import ptx_matcher

class PTXRenderer(CStyleLanguage):
  extra_matcher = ptx_matcher
  pass