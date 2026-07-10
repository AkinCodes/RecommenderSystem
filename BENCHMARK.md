# Inference Benchmark: PyTorch vs ONNX Runtime

## Model

- **Architecture:** DLRM (num_features=8, cardinalities=[943, 1682], embedding_dim=64, mlp=[128, 64, 32])
- **Parameters:** 211,841
- **ONNX file size:** 3.9 KB

## Setup

- **Warmup runs:** 50
- **Benchmark runs:** 500
- **Batch sizes:** [1, 8, 32, 64, 128, 256]
- **Device:** CPU

## Results

| Batch Size | Runtime | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Throughput (samples/s) |
|------------|---------|-----------|-------------|----------|----------|----------------------|
|          1 | PyTorch |     0.059 |       0.058 |    0.066 |    0.074 |               16,827 |
|          1 | ONNX    |     0.010 |       0.010 |    0.011 |    0.018 |               95,558 |
|          8 | PyTorch |     0.070 |       0.067 |    0.077 |    0.103 |              114,505 |
|          8 | ONNX    |     0.020 |       0.019 |    0.020 |    0.022 |              409,299 |
|         32 | PyTorch |     0.076 |       0.076 |    0.077 |    0.085 |              421,932 |
|         32 | ONNX    |     0.036 |       0.036 |    0.040 |    0.044 |              877,085 |
|         64 | PyTorch |     0.085 |       0.083 |    0.089 |    0.099 |              757,247 |
|         64 | ONNX    |     0.055 |       0.054 |    0.065 |    0.086 |            1,157,815 |
|        128 | PyTorch |     0.100 |       0.098 |    0.107 |    0.147 |            1,276,479 |
|        128 | ONNX    |     0.088 |       0.088 |    0.101 |    0.124 |            1,451,022 |
|        256 | PyTorch |     0.132 |       0.130 |    0.142 |    0.159 |            1,935,853 |
|        256 | ONNX    |     0.128 |       0.123 |    0.140 |    0.165 |            1,997,350 |

## Speedup Summary

| Batch Size | Speedup (ONNX vs PyTorch) |
|------------|--------------------------|
| 1 | 5.68x |
| 8 | 3.57x |
| 32 | 2.08x |
| 64 | 1.53x |
| 128 | 1.14x |
| 256 | 1.03x |
