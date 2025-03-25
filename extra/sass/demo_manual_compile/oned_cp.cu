extern "C" __global__ void oned_cp(float* data0, float* data1) {
  int gidx0 = blockIdx.x; /* 32 */
  *(data0+gidx0) = *(data1+gidx0);
}