import time, math, unittest, functools, os, torch
import numpy as np
from typing import List, Callable
import warnings
from tinygrad.helpers import DISABLE_COMPILER_CACHE, getenv, IMAGE, DEBUG, CI, Context, TRANSCENDENTAL, DEVECTORIZE, OSX, Context
from tinygrad.helpers import getenv, IMAGE, DEBUG, CI, Context, TRANSCENDENTAL, OSX, AMD_LLVM
from tinygrad import Tensor, Device, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.device import is_dtype_supported

FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
PRINT_TENSORS = getenv("PRINT_TENSORS", 0)
def helper_test_op(shps, torch_fxn, tinygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3,
                   forward_only=False, vals=None, low=-2, high=2):
  if tinygrad_fxn is None: tinygrad_fxn = torch_fxn
  ts, tst = prepare_test_op2(low, high, shps, vals, forward_only)

  st = time.monotonic()
  out = torch_fxn(*ts)
  torch_fp = time.monotonic() - st

  # move inputs to a different device, test the device of intermediate tensors are correct
  #if mt:=getenv("MOVE_TENSOR", ""): for t in tst: t.to_(mt)

  st = time.monotonic()
  ret = tinygrad_fxn(*tst).realize()
  tinygrad_fp = time.monotonic() - st

  def compare(s, tinygrad_output, torch_output, atol, rtol):
    if PRINT_TENSORS: print(s, tinygrad_output, torch_output)
    try:
      assert tinygrad_output.shape == torch_output.shape, f"shape mismatch: tinygrad={tinygrad_output.shape} | torch={torch_output.shape}"
      assert tinygrad_output.dtype == torch_output.dtype, f"dtype mismatch: tinygrad={tinygrad_output.dtype} | torch={torch_output.dtype}"
      if np.issubdtype(tinygrad_output.dtype, np.floating):
        np.testing.assert_allclose(tinygrad_output, torch_output, atol=atol, rtol=rtol)
      else:
        np.testing.assert_equal(tinygrad_output, torch_output)
    except Exception as e:
      raise Exception(f"{s} failed shape {tinygrad_output.shape}: {e}")

  if DEBUG >= 6:
    np.set_printoptions(linewidth=200, suppress=True)
    print(ret.numpy())
    print(out.detach().cpu().numpy())
  compare("forward pass", ret.numpy(), out.detach().cpu().numpy(), atol=atol, rtol=rtol)

  torch_fbp, tinygrad_fbp = np.nan, np.nan
  if not forward_only and not FORWARD_ONLY and ts and tst:
    st = time.monotonic()
    torch_grads = torch.autograd.grad(torch_fxn(*ts).sum(), ts)
    torch_fbp = time.monotonic() - st

    st = time.monotonic()
    # NOTE: we now have to recompute the forward pass since we realized it
    tiny_grads = tinygrad_fxn(*tst).sum().gradient(*tst)
    Tensor.realize(*tiny_grads)
    tinygrad_fbp = time.monotonic() - st

    for i, (t, torch_grad) in enumerate(zip(tiny_grads, torch_grads)):
      compare(f"backward pass tensor {i}", t.numpy(), torch_grad.detach().cpu().numpy(), atol=grad_atol, rtol=grad_rtol)

  if not CI:
    print("\ntesting %40r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % \
          (shps, torch_fp*1000, tinygrad_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")

def prepare_test_op(low, high, shps, vals, forward_only=False):
  if shps is None:
    ts = [torch.tensor(x, requires_grad=(not forward_only)) for x in vals]
  else:
    np.random.seed(0)
    np_data = [np.random.uniform(low=low, high=high, size=size).astype(_to_np_dtype(dtypes.default_float)) for size in shps]
    ts = [torch.tensor(data, requires_grad=(not forward_only)) for data in np_data]
  for i in range(len(ts)):
    # NOTE: torch default int64 for python ints input
    if ts[i].dtype == torch.int64: ts[i] = ts[i].type(torch.int32)
  tst = [Tensor(x.detach().cpu().numpy(), requires_grad=(not forward_only and not FORWARD_ONLY)) for x in ts]
  return ts, tst


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

  def test_arange(self):
    print(Tensor.arange(100).numpy())

  def test_linespace(self):
    print(Tensor.linspace(5, 10, 3).numpy())

  def test_abs(self):
    with Context(NOOPT=1): helper_test_op([(2,2)], torch.abs, Tensor.abs)
    #with Context(NOOPT=0): helper_test_op([(8,8)], torch.abs, Tensor.abs)
    #helper_test_op([(45,65)], torch.abs, Tensor.abs)

  def _test_abs(self, data, dtype):
    a = Tensor(data,  dtype=dtype, device="asm") 
    np.testing.assert_equal(a.abs().numpy(), np.abs(np.array(data).astype(_to_np_dtype(dtype))))

  def test_abs_f32(self):
    self._test_abs([-1, 0, 2, -4], dtypes.float32)
    with Context(NOOPT=1): self._test_abs([-1, 0, 2, -4], dtypes.float32)
  def test_abs_f64(self):
    self._test_abs([-1, 0, 2, -4], dtypes.float64)
    with Context(NOOPT=1): self._test_abs([-1, 0, 2, -4], dtypes.float64)
  def test_abs_i32(self):
    self._test_abs([-1, 0, 2, -4], dtypes.int32)
    with Context(NOOPT=1): self._test_abs([-1, 0, 2, -4], dtypes.int32)
  def test_abs_i64(self):
    self._test_abs([-1, 0, 2, -4], dtypes.int64)
    with Context(NOOPT=1): self._test_abs([-1, 0, 2, -4], dtypes.int64)

  def test_acos_noopt(self):
    with Context(NOOPT=1):
      helper_test_op([(2,2)], lambda x: x.acos(), low=-1, high=1)
  def test_acos(self):
    helper_test_op([(4,)], lambda x: x.acos(), low=-1, high=1)
    helper_test_op([(45,65)], lambda x: x.acos(), low=-1, high=1)
    helper_test_op([(45,65)], lambda x: x.acos(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.acos(), low=300, high=303)

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

  def _test_matmul_f32_rand(self, shape_a, shape_b):
    np.random.seed(0)
    a = np.random.rand(*shape_a).astype(np.float32)
    b = np.random.rand(*shape_b).astype(np.float32)
    a_t = Tensor(a)
    b_t = Tensor(b)
    np_res = np.matmul(a, b)
    with Context(NOOPT=1, DEBUG=0):
      clang_res = a_t.to("cpu").dot(b_t.to("cpu")).numpy()
    with Context(NOOPT=1):
      asm_res = a_t.to("asm").dot(b_t.to("asm")).numpy()
      np.testing.assert_allclose(clang_res, asm_res)
    with Context(NOOPT=0):
      asm_res_2 = a_t.to("asm").dot(b_t.to("asm")).numpy()
      np.testing.assert_allclose(clang_res, asm_res_2)

  def test_matmul_f32_rand_3_2_4(self):
    self._test_matmul_f32_rand((3,4), (4,2))

  def test_matmul_f32_rand_3_2_8(self):
    self._test_matmul_f32_rand((3,8), (8,2))

  def test_matmul_f32_rand_3_2_16(self):
    self._test_matmul_f32_rand((3,16), (16,2))

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
      K = 2
      a, b = self._setup_data((3, K), (K, 4))
      if os.environ.get("MANUAL_CLANG"):
        clang_res = self._clang(a, b)
      if os.environ.get("MANUAL_ASM"):
        asm_res = self._asm(a, b) 
      #np.testing.assert_equal(clang_res, asm_res)

