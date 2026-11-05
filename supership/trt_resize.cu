extern "C" __global__ void normalize(const unsigned char* input, float* output, int side, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = side * side;
    if (i >= pixels * batch * 3) return;
    int n = i / (pixels * 3), c = (i / pixels) % 3, p = i % pixels;
    const int* info = reinterpret_cast<const int*>(input) + n * 4;
    int width = info[0], height = info[1], offset = info[2];
    if (width <= 0 || height <= 0) { output[i] = 0; return; }
    const unsigned char* src = input + offset;
    if (width == side && height == side) {
        output[i] = float(src[p * 3 + c]) * (1.0f / 255.0f);
        return;
    }
    double fx = (double(p % side) + 0.5) * double(width) / side - 0.5;
    double fy = (double(p / side) + 0.5) * double(height) / side - 0.5;
    int x = (int)floor(fx), y = (int)floor(fy);
    fx -= x; fy -= y;
    if (x < 0) { x = 0; fx = 0; }
    if (x >= width - 1) { x = width - 1; fx = 0; }
    int y0 = max(0, min(y, height - 1)), y1 = max(0, min(y + 1, height - 1));
    int x1 = min(x + 1, width - 1);
    int a1 = __double2int_rn(fx * 2048), a0 = __double2int_rn((1-fx) * 2048);
    int b1 = __double2int_rn(fy * 2048), b0 = __double2int_rn((1-fy) * 2048);
    int top = src[(y0 * width + x) * 3 + c] * a0 + src[(y0 * width + x1) * 3 + c] * a1;
    int bottom = src[(y1 * width + x) * 3 + c] * a0 + src[(y1 * width + x1) * 3 + c] * a1;
    int value = (top * b0 + bottom * b1 + (1 << 21)) >> 22;
    output[i] = float(value) * (1.0f / 255.0f);
}
extern "C" __global__ void sigmoid(float* logits, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) logits[i] = 1.0f / (1.0f + expf(-logits[i]));
}
