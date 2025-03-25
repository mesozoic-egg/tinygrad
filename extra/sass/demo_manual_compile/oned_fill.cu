extern "C" __global__ void oned_fill(float* data0) {
  int gidx0 = blockIdx.x; /* 32 */
  *(data0+gidx0) = 1.0f;
}