extern "C" __global__ void loop(float* A, float* B, int K) {
  for (int i = 0; i < K; i++) {
    A[i] = B[i];
  }
}