extern "C" __global__ void normalize(const unsigned char* input, float* output, int pixels, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels * batch * 3) return;
    int n = i / (pixels * 3), c = (i / pixels) % 3, p = i % pixels;
    output[i] = float(input[(n * pixels + p) * 3 + c]) * (1.0f / 255.0f);
}
extern "C" __global__ void sigmoid(float* logits, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) logits[i] = 1.0f / (1.0f + expf(-logits[i]));
}
