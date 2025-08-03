import time, math, unittest, functools, os, torch
import numpy as np
from typing import List, Callable
import warnings
from tinygrad.helpers import DISABLE_COMPILER_CACHE, getenv, IMAGE, DEBUG, CI, Context, TRANSCENDENTAL, DEVECTORIZE, OSX, Context 
from tinygrad.helpers import getenv, IMAGE, DEBUG, CI, Context, TRANSCENDENTAL, OSX, AMD_LLVM
from tinygrad import Tensor, Device, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.device import is_dtype_supported
from tinygrad.renderer.asm import Arch

def skipU(flag: str):
  if os.environ.get(flag):
    return lambda func: func
  return unittest.skip("")

if getenv("TINY_BACKEND"):
  import tinygrad.frontend.torch # noqa: F401 # pylint: disable=unused-import
  torch.set_default_device("tiny")

FORWARD_ONLY = getenv("FORWARD_ONLY", 1)
PRINT_TENSORS = getenv("PRINT_TENSORS", 0)
def helper_test_op(shps, torch_fxn, tinygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3,
                   forward_only=False, vals=None, low=-2, high=2):
  if tinygrad_fxn is None: tinygrad_fxn = torch_fxn
  ts, tst = prepare_test_op(low, high, shps, vals, forward_only)

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
    if os.environ.get("INPUT_BYTES"):
      print(f"{np_data=}")
      b = np_data[0].tobytes()
      for _b in b: print(f"{_b:#x}", end=", ")
      print()
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

  @unittest.skip("need to handle MOD")
  def test_eye(self):
    print(Tensor.eye(10).numpy())

  def test_split(self):
    tensors = Tensor.arange(16).reshape((4,4)).split((2,2))
    ret = tensors[1].tolist()
    assert ret == [
      [8, 9, 10, 11],
      [12, 13, 14, 15]
    ]

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
    helper_test_op([], lambda: torch.arange(10, dtype=torch.int32), lambda: Tensor.arange(10), forward_only=True)
    helper_test_op([], lambda: torch.arange(36, dtype=torch.int32), lambda: Tensor.arange(36), forward_only=True)
    helper_test_op([], lambda: torch.arange(5, 10, 3, dtype=torch.int32), lambda: Tensor.arange(5, 10, 3), forward_only=True)
    helper_test_op([], lambda: torch.arange(10, 5, -3, dtype=torch.int32), lambda: Tensor.arange(10, 5, -3), forward_only=True)
    helper_test_op([], lambda: torch.arange(11, 5, -3, dtype=torch.int32), lambda: Tensor.arange(11, 5, -3), forward_only=True)
    helper_test_op([], lambda: torch.arange(1, 78, 2, dtype=torch.int32), lambda: Tensor.arange(1, 78, 2), forward_only=True)

  def test_arange_float(self):
    helper_test_op([], lambda: torch.arange(5.5, 175.5, 2.5), lambda: Tensor.arange(5.5, 175.5, 2.5), forward_only=True)
    end = 164 #164 would fail, 163 passes
    helper_test_op([], lambda: torch.arange(5.5, end, 2.5),
                   lambda: Tensor.arange(5.5, end, 2.5), forward_only=True)

  def test_argmax(self):
    # check if it returns the first index for multiple occurences
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[2, 2]])
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[1, 2, 2]])
    np.testing.assert_equal(Tensor([2,2]).argmax().numpy(), 0)
    np.testing.assert_equal(Tensor([1,2,2]).argmax().numpy(), 1)
    helper_test_op([(10,20)], lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(0, False).type(torch.int32), lambda x: x.argmax(0, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(1, False).type(torch.int32), lambda x: x.argmax(1, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(1, True).type(torch.int32), lambda x: x.argmax(1, True), forward_only=True)
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[0, -2**31]])
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[-2**31, 0]])
    helper_test_op(None, lambda x: x.type(torch.int32).argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[False, True]])
    helper_test_op(None, lambda x: x.type(torch.int32).argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[True, False]])
    helper_test_op(None, lambda x: (~x).argmax().type(torch.int32), lambda x: (~x).argmax(), forward_only=True, vals=[[2, 2]])

  def test_max(self):
    helper_test_op([(45,3)], lambda x: x.max())
    helper_test_op([(45,3)], lambda x: x.max().mul(0.5))
    helper_test_op(None, lambda x: x.max().mul(0.5), vals=[[[1.0,1.0,0.0,1.0]],])
    helper_test_op(None, lambda x: x.max().mul(0.5), vals=[[[1.0,1.0,0.0,1.0]],])
    helper_test_op([(3,4,5,6)], lambda x: x.max(axis=1)[0], lambda x: x.max(axis=1))
    helper_test_op([()], lambda x: x.max())
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[False, True]])
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[True, False]])
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[0, -2**31]])
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[-2**31, 0]])

  def test_argsort(self):
    for dim in [-1, 0, 1]:
      for descending in [True, False]:
        helper_test_op([(8,8,6)], lambda x: torch.argsort(x, dim=dim, descending=descending, stable=True).type(torch.int32),
                                  lambda x: x.argsort(dim, descending), forward_only=True)


  def test_linespace(self):
    print(Tensor.linspace(5, 10, 3).numpy())

  def test_abs(self):
    with Context(NOOPT=1): helper_test_op([(2,2)], torch.abs, Tensor.abs)
    with Context(NOOPT=0): helper_test_op([(8,8)], torch.abs, Tensor.abs)
    helper_test_op([(45,65)], torch.abs, Tensor.abs)

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
  def test_acos_large(self):
    helper_test_op([(20,)], lambda x: x.acos(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.acos(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.acos(), low=300, high=303)

  def test_sum(self):
    np_ret = np.ones((16, 16)).sum(axis=1)
    tg_ret = Tensor.ones(16, 16, dtype=dtypes.int).sum(axis=1).numpy()
    np.testing.assert_equal(tg_ret, np_ret)

  def test_where(self):
    a = Tensor([1, 2, 3])
    b = (a > 2).where(8, 9)
    assert b.tolist() == [9, 9, 8]

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
  def test_idiv(self):
    helper_test_op(None, functools.partial(torch.div, rounding_mode="trunc"), Tensor.idiv, forward_only=True,
                   vals=[[-4, 7, 5, 4, -7, 8], [2, -3, 8, -2, 3, 5]])

  def test_acosh(self):
    helper_test_op([(2,)], lambda x: x.acosh(), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-3, grad_rtol=1e-2, low=-300, high=-297)

  def test_acosh_high(self):
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-6, low=300, high=303)

  def test_log(self):
    helper_test_op([(45,65)], lambda x: x.log(), grad_atol=1e-6, low=300, high=303)

  def test_log2(self):
    with Context(NOOPT=0):
      helper_test_op([(1,)], lambda x: x.log2(), grad_atol=1e-6)
    with Context(NOOPT=1):
      helper_test_op([(45,65)], lambda x: x.log2(), grad_atol=1e-6)

  def test_log2_unroll(self):
    with Context(NOOPT=0):
      helper_test_op([(4,)], lambda x: x.log2(), grad_atol=1e-6)

  def test_recip(self):
    with Context(NOOPT=1, TRANSCENDENTAL=1):
      helper_test_op([(4,)], lambda x: x.reciprocal(), grad_atol=1e-6)
    with Context(NOOPT=0, TRANSCENDENTAL=1):
      helper_test_op([(4,)], lambda x: x.reciprocal(), grad_atol=1e-6)

  def test_and(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor&tor, lambda: ten&ten, forward_only=True)
    helper_test_op([], lambda: tor&0x1337, lambda: ten&0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337&tor, lambda: 0x1337&ten, forward_only=True)

    data = [[True, True, False, False], [True, False, True, False]]
    tor0, tor1 = torch.tensor(data[0], dtype=torch.bool),  torch.tensor(data[1], dtype=torch.bool)
    ten0, ten1 = Tensor(data[0], dtype=dtypes.bool), Tensor(data[1], dtype=dtypes.bool)
    helper_test_op([], lambda: tor0&tor1, lambda: ten0&ten1, forward_only=True)

    helper_test_op(None, lambda x: (1 < x) & (x < 2), forward_only=True, vals=[[1.2, 1.2, 1.2, 3.2]])
  def test_all(self):
    helper_test_op([(3,4,5,6)], lambda x: x.all(), forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[True, True]], forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[True, False]], forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[False, False]], forward_only=True)
    helper_test_op([()], lambda x: x.all(), forward_only=True)

  def test_avg_pool2d(self):
    shape = (32,2,111,28)
    for ksz in [(2,2), (3,3), (3,2), (5,5), (5,1)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz), rtol=1e-5)

    # regression test for https://github.com/tinygrad/tinygrad/pull/7581
    helper_test_op([(1,1,8,8)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(1,2), padding=(0,1), stride=(5,1)),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(1,2), padding=(0,1), stride=(5,1)), rtol=1e-5)

  def test_avg_pool2d_ceil_mode(self):
    shape = (1,1,6,6)
    for ksz in [(3,3), 3, (3,2), 4]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True), rtol=1e-5)

  def test_avg_pool2d_padding(self):
    shape = (32,2,111,28)
    for ksz in [(2,2), (3,3), 2, 3, (3,2)]:
      for p in [1, (1,0), (0,1)]:
        with self.subTest(kernel_size=ksz, padding=p):
          helper_test_op([shape],
            lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=p),
            lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=p), rtol=1e-5)
    with self.assertRaises(ValueError):
      Tensor.avg_pool2d(Tensor.randn((32,2,111,28)), kernel_size=(2,2), padding=(1,1,1))

  #no candidates left
  def test_pool_sum(self):
    shape = [(1,1,16,16,16)]
    x, x2 = prepare_test_op(-2, 2, shape, True)
    x2 = x2[0]
    padding = [1,1,1,1,1,1]
    axis = (-3, -2, -1)
    kernel_size = (8,8,8)
    stride = 5
    dilation = 1
    x2.ones_like().pad(padding)._pool(kernel_size, stride, dilation).sum(axis).realize()
    #y = Tensor.avg_pool2d(x2, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False)
    #y.realize()

  def test_avg_pool3d_failure(self):
    with Context(NOOPT=0):
      helper_test_op([(1,1,16,16,16)],
        lambda x: torch.nn.functional.avg_pool3d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False),
        lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False), rtol=1e-5, forward_only=True)

  def test_sigmoid(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid)
  def test_sigmoid_extreme(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=300, high=400)
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=-400, high=-300)
    x = Tensor([300.0])
    self.assertAlmostEqual(x.sigmoid()[0].gradient(x)[0].item(), 0.0)
    x = Tensor([-300.0])
    self.assertAlmostEqual(x.sigmoid()[0].gradient(x)[0].item(), 0.0)

  def test_sigmoid_alt_extreme(self):
    def sigmoid(x:Tensor): return x.exp() / (1 + x.exp())
    x = Tensor([300.0])
    self.assertAlmostEqual(sigmoid(x)[0].gradient(x)[0].item(), 0.0)
    x = Tensor([-300.0])
    self.assertAlmostEqual(sigmoid(x)[0].gradient(x)[0].item(), 0.0)

  def test_logsigmoid(self):
    helper_test_op([(45,65)], torch.nn.functional.logsigmoid, Tensor.logsigmoid)
    helper_test_op([()], torch.nn.functional.logsigmoid, Tensor.logsigmoid)

  def test_hardsigmoid(self):
    helper_test_op([(45,65)], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid)
    helper_test_op([()], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid)
  def test_hardsigmoid_extreme(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=300, high=400)
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=-400, high=-300)
  def test_softplus(self):
    helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6)
    helper_test_op([(45,65)], lambda t: torch.nn.functional.softplus(t, beta=3), lambda t: Tensor.softplus(t, beta=3), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda t: torch.nn.functional.softplus(t, beta=1/3), lambda t: Tensor.softplus(t, beta=1/3), grad_atol=1e-6)
    # # TODO: support threshold and enable this
    # helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6, low=300, high=400)
    helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6, low=-400, high=-300)
    helper_test_op([()], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6)

  def test_cross_entropy_1(self):
    r = "none"
    #shape = [(32, 10), (32, 10)]
    shape = [(5,4), (5,4)]
    x1, x2 = prepare_test_op(-2, 2, shape, True)
    x, y = x2
    with Context(NOOPT=0):
      x.sigmoid().binary_crossentropy(y.clip(0,1), reduction=r).realize()
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(),y.clip(0,1)),
                                       lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1)))

  def test_binary_crossentropy(self):
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(),y.clip(0,1)),
                                       lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,y.clip(0,1)),
                                       lambda x,y: x.binary_crossentropy_logits(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,y.clip(0,1)),
                                       lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(),y.clip(0,1)),
                                       lambda x,y: x.binary_crossentropy_logits(y.clip(0,1)))
  def test_binary_crossentropy_reductions(self):
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(), y.clip(0,1), reduction=r),
                                         lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1), reduction=r))
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x, y.clip(0,1), reduction=r),
                                         lambda x,y: x.binary_crossentropy_logits(y.clip(0,1), reduction=r))
  def test_binary_crossentropy_logits_pos_weights(self):
    pos_weight = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,y.clip(0,1),
                                                                                                        pos_weight=torch.tensor(pos_weight)),
                                       lambda x,y: x.binary_crossentropy_logits(y.clip(0,1),pos_weight=Tensor(pos_weight)))
  def test_cross_entropy_class_probabilities(self):
    helper_test_op([(32,), (32,)], lambda x,y: torch.nn.functional.cross_entropy(x, y), lambda x,y: x.cross_entropy(y))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y), lambda x,y: x.cross_entropy(y))
    helper_test_op([(32,4,4,4), (32,4,4,4)], lambda x,y: torch.nn.functional.cross_entropy(x, y), lambda x,y: x.cross_entropy(y))

  def test_cross_entropy_2(self):
    r = "none"
    shape = [(32, 10), (32, 10)]
    shape = [(5, 4), (5, 4)]
    x1, x2 = prepare_test_op(-2, 2, shape, True)
    x, y = x2
    with Context(NOOPT=0):
      x.sigmoid().binary_crossentropy(y.clip(0,1)).realize()

  def test_pad_reflect_mode(self):
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,2,3,2), mode="reflect"), lambda x: x.pad((0,2,3,2), mode="reflect"))
    helper_test_op([(5,5,5)], lambda x: torch.nn.functional.pad(x, (0,2), mode="reflect"), lambda x: x.pad((0,2), mode="reflect"))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,2,-1), mode="reflect"), lambda x: x.pad((-1,2,2,-1), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-3,0,-3), mode="reflect"), lambda x: x.pad((3,-3,0,-3), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-5,1,-5), mode="reflect"), lambda x: x.pad((3,-5,1,-5), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,0,0,-5), mode="reflect"), lambda x: x.pad((0,0,0,-5), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (4,4,0,4), mode="reflect"), lambda x:x.pad((4,4,0,4),mode="reflect"))
    self.helper_test_exception([(1,1,5,5)],
                                lambda x: torch.nn.functional.pad(x, (3,5,0,0),mode="reflect"), lambda x: x.pad((3,5,0,0),mode="reflect"),
                                expected=(RuntimeError, ValueError))

  def helper_test_exception(self, shps, torch_fxn, tinygrad_fxn, expected, forward_only=False, exact=False, vals=None, low=-1.5, high=1.5):
    if getenv("MOCKGPU") and Device.DEFAULT == "NV": self.skipTest('helper_test_exception fails in CI CUDA')
    ts, tst = prepare_test_op(low, high, shps, vals, forward_only)
    with self.assertRaises(expected) as torch_cm:
      torch_fxn(*ts)
    with self.assertRaises(expected) as tinygrad_cm:
      tinygrad_fxn(*tst)
    if exact: self.assertEqual(str(torch_cm.exception), str(tinygrad_cm.exception))
    if not CI: print("\ntesting %40r   torch/tinygrad exception: %s / %s" % (shps, torch_cm.exception, tinygrad_cm.exception), end="")

  def test_pad_reflect_mode(self):
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,2,3,2), mode="reflect"), lambda x: x.pad((0,2,3,2), mode="reflect"))
    helper_test_op([(5,5,5)], lambda x: torch.nn.functional.pad(x, (0,2), mode="reflect"), lambda x: x.pad((0,2), mode="reflect"))
    helper_test_op([(1,1,5,5,5)], lambda x: torch.nn.functional.pad(x, (1,2,3,4,1,2), mode="reflect"),
                                  lambda x: x.pad((1,2,3,4,1,2), mode="reflect"))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,2,-1), mode="reflect"), lambda x: x.pad((-1,2,2,-1), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-3,0,-3), mode="reflect"), lambda x: x.pad((3,-3,0,-3), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-5,1,-5), mode="reflect"), lambda x: x.pad((3,-5,1,-5), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,0,0,-5), mode="reflect"), lambda x: x.pad((0,0,0,-5), mode="reflect"))

    # max pad size for reflect is exactly once: pad < input size
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (4,4,0,4), mode="reflect"), lambda x:x.pad((4,4,0,4),mode="reflect"))
    # raise error for relfection padding when: pad >= input size
    self.helper_test_exception([(1,1,5,5)],
                                lambda x: torch.nn.functional.pad(x, (3,5,0,0),mode="reflect"), lambda x: x.pad((3,5,0,0),mode="reflect"),
                                expected=(RuntimeError, ValueError))
  @skipU("MANUAL")
  def test_manual(self):
    #shape = (1,1,5,5,5)
    #np_data = np.random.uniform(low=-2, high=2, size=shape).astype(_to_np_dtype(dtypes.default_float))
    #x = Tensor(np_data).reshape((1,1,5,5,5)).pad((1,2,3,4,1,2), mode="reflect")
    #print(x.numpy())
    #return
    helper_test_op([(1,1,5,5,5)], lambda x: torch.nn.functional.pad(x, (1,2,3,4,1,2), mode="reflect"),
                                  lambda x: x.pad((1,2,3,4,1,2), mode="reflect"))

  def test_all_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.all(axis=(1,2)), forward_only=True)


  def test_broadcast_full(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div), (torch.pow, Tensor.pow)]:
      for shapes in [((5,13,24,16), (5,1,24,1)), ((1,3,1,7,1), (2,1,5,1,8))]:
        print(f"{tinygrad_op=} {shapes=}")
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          if tinygrad_op != Tensor.pow:
            helper_test_op(shapes, torch_op, tinygrad_op)
          else:
            helper_test_op(shapes, torch_op, tinygrad_op, low=0, high=3)

  def test_broadcast_pow(self):
    tinygrad_op = Tensor.pow
    torch_op = torch.pow
    shapes = ((5, 13, 24, 16), (5, 1, 24, 1))
    s = 20
    shapes = ((5, 13, s, 16),  (5, 1, s, 1))
    helper_test_op(shapes, torch_op, tinygrad_op, low=0, high=3)

  def test_cast(self):
    helper_test_op([(3, 3)], lambda x: x.float())
    helper_test_op(None, lambda x: x.float(), vals=[[0, 1, 2, 3]], forward_only=True)
    helper_test_op(None, lambda x: x.float(), vals=[[True, False]], forward_only=True)
    helper_test_op([(3, 3)], lambda x: x.int(), forward_only=True)
    helper_test_op([(3, 3)], lambda x: x.bool(), forward_only=True)
  def test_all(self):
    helper_test_op([(3,4,5,6)], lambda x: x.all(), forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[True, True]], forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[True, False]], forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[False, False]], forward_only=True)
    helper_test_op([()], lambda x: x.all(), forward_only=True)

  def test_cmp_lt_backwards(self):
    Tensor.manual_seed(0)
    tt = Tensor.randn(4, requires_grad=True)
    (tt*(tt < 0)).sum().backward()
    print(f"tinygrad: {tt.grad.numpy()=}")
    t = torch.tensor(tt.numpy(), requires_grad=True)
    (t*(t < 0)).sum().backward()
    print(f"torch: {t.grad.cpu().numpy()=}")
    np.testing.assert_allclose(tt.grad.numpy(), t.grad.cpu().numpy(), rtol=1e-5)
  def test_cmp_ne_backwards(self):
    # new grad zeroes these out
    """
    t1 = torch.ones(4, requires_grad=True)
    t2 = torch.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (t1 != t2).sum().backward)
    tt1 = Tensor.ones(4, requires_grad=True)
    tt2 = Tensor.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (tt1 != tt2).sum().backward)
    """
    Tensor.manual_seed(0)
    tt = Tensor.randn(4, requires_grad=True)
    (tt*(tt != 0)).sum().backward()
    t = torch.tensor(tt.numpy(), requires_grad=True)
    (t*(t != 0)).sum().backward()
    np.testing.assert_allclose(tt.grad.numpy(), t.grad.cpu().numpy(), rtol=1e-5)

  def test_logical_not(self):
    helper_test_op(None, torch.logical_not, Tensor.logical_not, vals=[[True, False, True]], forward_only=True)
    helper_test_op(None, torch.logical_not, Tensor.logical_not,
                   vals=[[1.,2.,0.,0.5]], forward_only=True)

  @skipU("MANUAL")
  def test_manual(self):
    pass

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

