import time, math, unittest, functools, os
import numpy as np
from typing import List, Callable
import warnings
from tinygrad.helpers import DISABLE_COMPILER_CACHE, getenv, IMAGE, DEBUG, CI, Context, TRANSCENDENTAL, DEVECTORIZE, OSX, Context
from tinygrad import Tensor, Device, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.device import is_dtype_supported

class TestOps(unittest.TestCase):
  def test_full(self):
    a = Tensor.full((4, 4), 20, dtype=dtypes.int).contiguous().realize()
    np.testing.assert_equal(a.numpy(), np.full((4,4), 20))
  def test_full_int64(self):
    a = Tensor.full((4, 4), 20, dtype=dtypes.int64).contiguous().realize()
    np.testing.assert_equal(a.numpy(), np.full((4,4), 20, dtype=np.int64))
  def test_zeros(self):
    a = Tensor.zeros(4, 4, dtype=dtypes.int32).contiguous().realize()
    np.testing.assert_equal(a.numpy(), np.zeros((4,4), dtype=np.int32))
  def test_full_float32(self):
    a = Tensor.full((4,4), 20.0, dtype=dtypes.float32).contiguous().numpy()
    np.testing.assert_equal(a, np.full((4,4), 20.0, dtype=np.float32))

  @unittest.skip("")
  def test_eye(self):
    print(Tensor.eye(10).numpy())

  @unittest.skip("")
  def test_split(self):
    tensors = Tensor.arange(16).reshape((4,4)).split((2,2))
    print(tensors[1].numpy())

  @unittest.skip("")
  def test_chunk(self):
    t = Tensor.arange(13).repeat((8, 1))
    print(f"{t.shape=}")
    ts = t.chunk(6, 1)
    for _t in ts: print(f"{_t.shape=}")
    print(ts[0].numpy())

  def test_meshgrid(self):
    x, y = Tensor([1, 2, 3]), Tensor([4, 5, 6])
    grid_x, grid_y = x.meshgrid(y)
    grid_x, grid_y = x.meshgrid(y, indexing="ij")
    print(grid_x.numpy())
    #print(grid_y.numpy())

  @unittest.skip("")
  def test_arange(self):
    print(Tensor.arange(100).numpy())

  @unittest.skip("")
  def test_linespace(self):
    print(Tensor.linspace(5, 10, 3).numpy())

  @unittest.skip("")
  def test_sum(self):
    print(Tensor.ones(16, 16, dtype=dtypes.int).sum(axis=1).numpy())

  @unittest.skip("")
  def test_where(self):
    a = Tensor([1, 2, 3])
    b = (a > 2).where(8, 9)
    print(b.numpy())

  def test_matmul_int64(self):
    with Context(DEBUG=0):
      a = Tensor.arange(12, device="PYTHON", dtype=dtypes.int64).reshape((3,4)).realize()
      b = Tensor.arange(8, device="PYTHON", dtype=dtypes.int64).reshape((4,2)).realize()
      a = a.to(Device.DEFAULT)
      b = b.to(Device.DEFAULT)
    c = a.dot(b)
    print(c.numpy())
    np.testing.assert_equal(c.numpy(), np.array([
      [28, 34],
      [76, 98],
      [124, 162]
      ], dtype=np.int64))
  def test_matmul_int64_noopt(self):
    with Context(DEBUG=0):
      a = Tensor.arange(12, device="PYTHON", dtype=dtypes.int64).reshape((3,4)).realize()
      b = Tensor.arange(8, device="PYTHON", dtype=dtypes.int64).reshape((4,2)).realize()
      a = a.to(Device.DEFAULT)
      b = b.to(Device.DEFAULT)
    c = a.dot(b)
    with Context(NOOPT=1):
      np.testing.assert_equal(c.numpy(), np.array([
        [28, 34],
        [76, 98],
        [124, 162]
        ], dtype=np.int64))
  def test_matmul_int32(self):
    with Context(DEBUG=0):
      a = Tensor.arange(12, device="PYTHON", dtype=dtypes.int32).reshape((3,4)).realize()
      b = Tensor.arange(8, device="PYTHON", dtype=dtypes.int32).reshape((4,2)).realize()
      a = a.to(Device.DEFAULT)
      b = b.to(Device.DEFAULT)
    c = a.dot(b)
    np.testing.assert_equal(c.numpy(), np.array([
      [28, 34],
      [76, 98],
      [124, 162]
      ], dtype=np.int32))
  def test_matmul_int32_noopt(self):
    with Context(DEBUG=0):
      a = Tensor.arange(12, device="PYTHON", dtype=dtypes.int32).reshape((3,4)).realize()
      b = Tensor.arange(8, device="PYTHON", dtype=dtypes.int32).reshape((4,2)).realize()
      a = a.to(Device.DEFAULT)
      print(f"{a.dtype=} {a.dtype.itemsize=}")
      b = b.to(Device.DEFAULT)
    c = a.dot(b)
    with Context(NOOPT=1):
      np.testing.assert_equal(c.numpy(), np.array([
        [28, 34],
        [76, 98],
        [124, 162]
        ], dtype=np.int32))

  def test_matmul_f32_noopt(self):
    with Context(DEBUG=0):
      a = Tensor.arange(12, device="PYTHON", dtype=dtypes.float32).reshape((3,4)).realize()
      b = Tensor.arange(8, device="PYTHON", dtype=dtypes.float32).reshape((4,2)).realize()
      a = a.to(Device.DEFAULT)
      b = b.to(Device.DEFAULT)
    c = a.dot(b)
    with Context(NOOPT=1):
      np.testing.assert_equal(c.numpy(), np.array([
        [28, 34],
        [76, 98],
        [124, 162]
        ], dtype=np.float32))

  def test_matmul_f32(self):
    with Context(DEBUG=0):
      a = Tensor.arange(12, device="PYTHON", dtype=dtypes.float32).reshape((3,4)).realize()
      b = Tensor.arange(8, device="PYTHON", dtype=dtypes.float32).reshape((4,2)).realize()
      a = a.to(Device.DEFAULT)
      b = b.to(Device.DEFAULT)
    c = a.dot(b)
    with Context(NOOPT=1):
      np.testing.assert_equal(c.numpy(), np.array([
        [28, 34],
        [76, 98],
        [124, 162]
        ], dtype=np.float32))
  def test_matmul_f32_rand(self):
    np.random.seed(0)
    a = np.random.rand(3, 4).astype(np.float32)
    b = np.random.rand(4, 2).astype(np.float32)
    a_t = Tensor(a)
    b_t = Tensor(b)
    np_res = np.matmul(a, b)
    with Context(NOOPT=1, DEBUG=0):
      clang_res = a_t.to("cpu").dot(b_t.to("cpu")).numpy()
    with Context(NOOPT=1):
      #np.testing.assert_equal(clang_res, np_res)
      #np.testing.assert_equal(np_res, a_t.to("asm").dot(b_t.to("asm")).numpy())
      np.testing.assert_allclose(np_res, a_t.to("asm").dot(b_t.to("asm")).numpy())

  @unittest.skipUnless(os.environ.get("MANUAL"), "")
  def test_matmul_f32_rand_small(self):
    np.random.seed(0)
    a = np.random.rand(2, 2).astype(np.float32)
    np.save("../tg-dev/matmul6/a.npy", a)
    b = np.random.rand(2, 2).astype(np.float32)
    np.save("../tg-dev/matmul6/b.npy", b)
    print(f"{a=}")
    print(f"{b=}")
    a_t = Tensor(a)
    b_t = Tensor(b)
    np_res = np.matmul(a, b)
    np.save("../tg-dev/matmul6/np_res.npy", np_res)
    with Context(NOOPT=1, DEBUG=5, SHOW_DISASM=1,  CLANG_O_LEVEL=0, FFP=0):
        clang_res = a_t.to("cpu").dot(b_t.to("cpu")).numpy()
        np.save("../tg-dev/matmul6/clang_res.npy", clang_res)
    with Context(NOOPT=1):
      asm_res = a_t.to("asm").dot(b_t.to("asm")).numpy()
      np.save("../tg-dev/matmul6/asm_res.npy", asm_res)
      np.testing.assert_equal(clang_res, asm_res)

  @unittest.skipUnless(os.environ.get("MANUAL"), "speed test")
  def test_matmul_f32_speed(self):
    """
    (1024,512) @ (512, 256)
      atol=1e-04 x86 asm:2.893 clang0:2.606 clang1:0.988 clang2:0.999
                 arm asm:6.19  clang0:5.08  clang1:0.935 clang2:1.04
    """
    with Context(DEBUG=0):
      np.random.seed(0)
      a = np.random.rand(1024, 512).astype(np.float32)
      b = np.random.rand(512, 256).astype(np.float32)

    with Context(NOOPT=1, BEAM=0, FFP=0):
      with Context(CLANG_O_LEVEL=1, SHOW_DISASM=0):
        repeats = 10
        a = Tensor(a)
        b = Tensor(b)
        with Context(DEBUG=2):
          c_cpu = speedrun("clang", a.to("cpu").dot(b.to("cpu")), repeats)
        with Context(DEBUG=6):
          c_asm = speedrun("asm", a.to("asm").dot(b.to("asm")), repeats)
        np.testing.assert_equal(c_asm, c_cpu, )

def speedrun(name: str, c: Tensor, repeat: int,) -> np.ndarray:
  res = c.clone().numpy()
  t0 = time.time()
  for i in range(repeat):
    c.clone().realize()
  t1 = time.time()
  print(f"Took {name} {(t1-t0)}s")
  return res
    
@unittest.skipUnless(os.environ.get("MANUAL"), "")
class TestMatmul(unittest.TestCase):
  def _setup_data(self, shapeA, shapeB):
    with Context(DEBUG=0):
      np.random.seed(0)
      a = np.random.rand(*shapeA).astype(np.float32)
      b = np.random.rand(*shapeB).astype(np.float32)
      return Tensor(a, device="cpu"), Tensor(b, device="cpu")

  def _clang(self, a: Tensor, b: Tensor):
    with Context(DEBUG=4):
      c_cpu = a.to("cpu").dot(b.to("cpu")).numpy()
      return c_cpu
  def _asm(self, a: Tensor, b: Tensor):
      with Context(DEBUG=6):
        c_asm = a.to("asm").dot(b.to("asm")).numpy()
        return c_asm
  def test_(self):
    with Context():
      K = 32
      a, b = self._setup_data((3, K), (K, 4))
      if os.environ.get("MANUAL_CLANG"):
        clang_res = self._clang(a, b)
      if os.environ.get("MANUAL_ASM"):
        asm_res = self._asm(a, b) 
      #np.testing.assert_equal(clang_res, asm_res)

