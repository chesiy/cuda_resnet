
### Test conv_winograd_cpu.cpp
```g++ conv_winograd_cpu.cpp -std=c++11 -o cpu_out```

### Test conv_winograd.gpu.cu
```nvcc conv_winograd_gpu.cu -std=c++11 -lcudnn -o out```