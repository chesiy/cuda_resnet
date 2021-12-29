
### Test conv_winograd_cpu.cpp
```g++ conv_winograd_cpu.cpp -std=c++11 -o cpu_out```

### Test conv_winograd.gpu.cu
```nvcc conv_winograd_gpu.cu -std=c++11 -lcudnn -o out```

### Test conv_winograd_4x4_3x3.cu
```nvcc conv_winograd_4x4_3x3.cu -std=c++11 -lcudnn -o out_4x4```

### Test conv_winograd_4x4_3x3_transpose.cu
```nvcc conv_winograd_4x4_3x3_transpose.cu -std=c++11 -lcudnn -o out_t_4x4```

### Test conv_im2col.cu
```nvcc conv_im2col.cu -std=c++11 -lcudnn -o out_im2col```

### Test conv_im2col_transpose.cu
```nvcc conv_im2col_transpose.cu -std=c++11 -lcudnn -o out_t_im2col```