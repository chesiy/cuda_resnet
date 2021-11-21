

//Dtype is float or float3 or float4
template <typename Dtype>
void convolution(const Dtype* A, const Dtype* B, const Dtype* K,
                 const int channels_A, const int height_A, const int width_A, const int channels_B,
                 const int pad_h, const int pad_w,
                 const int stride_v, const int stride_h,
                 const int dilation_h, const int dilation_w,
                 const int Hk, const int Wk){
    cudnnHandle_t h_cudnn;
    cudnnCreate(&h_cudnn);

    cudnnTensorDescriptor_t ts_in;

    checkCUDNN(cudnnCreateTensorDescriptor(&ts_in));
    checkCUDNN(cudnnSetTensor4dDescriptor(ts_in, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, channels_A, height_A, width_A));

    // =================================================计算输出大小
    cudnnTensorDescriptor_t ts_out;
    int height_B = (height_A+2*pad_h-dilation_h*(Hk-1)-1)/stride_v + 1;
    int width_B = (width_A+2*pad_w-dilation_w*(Wk-1)-1)/stride_h + 1;
    checkCUDNN(cudnnCreateTensorDescriptor(&ts_out));
    checkCUDNN(cudnnSetTensor4dDescriptor(ts_out, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, channels_B, height_B, width_B));

    cudnnFilterDescriptor_t kernel;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, channels_B, channels_A, Hk, Wk));

    cudnnConvolutionDescriptor_t conv;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_v, stride_h, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t algo;
    cudnnConvolutionFwdAlgoPerf_t *pref_algos = new cudnnConvolutionFwdAlgoPerf_t[1];
    int returnedAlgoCount;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(h_cudnn, ts_in, kernel, conv, ts_out, 1, &returnedAlgoCount, pref_algos));
    algo = pref_algos[0].algo;

    size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(h_cudnn, ts_in, kernel, conv, ts_out, algo, &workspace_size));
    // 分配工作区空间
    void * workspace;
    cudaMalloc(&workspace, workspace_size);
    // =================================================线性因子
    float alpha = 1.0f;
    float beta  = -100.0f;
    // =================================================数据准备
    const Dtype* d_A, d_B, d_K;
    cudaMalloc((void**)&d_A, width_A * height_A * sizeof(Dtype));
    cudaMalloc((void**)&d_B, width_B * height_B * sizeof(Dtype));
    cudaMalloc((void**)&d_K, Wk * Hk * sizeof (Dtype) * sizeof(Dtype));

    cudaMemcpy((void*)d_A, (void*)A, width_A * height_A * sizeof(Dtype), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_K, (void*)K, Wk * Hk * sizeof(Dtype) * sizeof(Dtype), cudaMemcpyHostToDevice);
    // =================================================执行
    checkCUDNN(cudnnConvolutionForward(
            h_cudnn,
            &alpha,
            ts_in,
            d_A,                         // 输入
            kernel,
            d_K,                      // 核
            conv,
            algo,
            workspace,
            workspace_size,
            &beta,
            ts_out,
            d_B                         // 输出
    ));
    cudaMemcpy((void*)B, (void*)d_B, width_B * height_B * sizeof(Dtype), cudaMemcpyDeviceToHost);

    // 释放cuDNN
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(ts_in);
    cudnnDestroyTensorDescriptor(ts_out);
    cudnnDestroyConvolutionDescriptor(conv);
    cudnnDestroyFilterDescriptor(kernel);
    cudnnDestroy(h_cudnn);
}

// **mode** can be "max"->0 or "avg_without_padding"->1
template <typename Dtype>
void Pooling(const Dtype* A, const Dtype* B,
                const int channels, const int height, const int width,
                int windowHeight,int windowWidth,
                int verticalPadding, int horizontalPadding,
                int verticalStride, int horizontalStride, int mode){
    cudnnHandle_t h_cudnn;
    checkCUDNN(cudnnCreate(&h_cudnn));

    cudnnPoolingDescriptor_t pooling_desc;

    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
    if(mode==0){
        checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,CUDNN_POOLING_MAX,CUDNN_NOT_PROPAGATE_NAN,windowHeight,
                                               windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride));
    }else
        if(mode==1){
            checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,CUDNN_NOT_PROPAGATE_NAN,windowHeight,
                                                   windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride));
        }

    cudnnTensorDescriptor_t in_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc,CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 1, channels, height, width));

    // =================================================计算输出大小
    cudnnTensorDescriptor_t out_desc;
    int height_B = (height-windowHeight+2*verticalPadding)/verticalStride+1;
    int width_B = (width-windowWidth+2*horizontalPadding)/horizontalStride+1;

    checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&out_desc,CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 1, channels, height_B, width_B));

    // =================================================线性因子
    float alpha = 1.0f;
    float beta  = -100.0f;
    // =================================================数据准备
    const Dtype* d_A, d_B;
    cudaMalloc((void**)&d_A, width * height * sizeof(Dtype));
    cudaMalloc((void**)&d_B, width_B * height_B * sizeof(Dtype));

    cudaMemcpy((void*)d_A, (void*)A, width * height * sizeof(Dtype), cudaMemcpyHostToDevice);
    // =================================================执行
    checkCUDNN(cudnnPoolingForward(h_cudnn,
                                   pooling_desc,
                                   &alpha,
                                   in_desc,
                                   A,
                                   &beta,
                                   out_desc));

    cudaMemcpy((void*)B, (void*)d_B, width_B * height_B * sizeof(Dtype), cudaMemcpyDeviceToHost);

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroy(h_cudnn);
}


//1*1卷积代替FC
template <typename Dtype>
void FullyConnect(const Dtype* A, const Dtype* B, const Dtype* K,
                  const int channels_A, const int height_A, const int width_A,
                  const int channels_B, const int height_B, const int width_B
                  ){
    cudnnHandle_t h_cudnn;
    checkCUDNN(cudnnCreate(&h_cudnn));

    cudnnTensorDescriptor_t ts_in;

    checkCUDNN(cudnnCreateTensorDescriptor(&ts_in));
    checkCUDNN(cudnnSetTensor4dDescriptor(ts_in, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, channels_A, height_A, width_A));

    cudnnTensorDescriptor_t ts_out;
    checkCUDNN(cudnnCreateTensorDescriptor(&ts_out));
    checkCUDNN(cudnnSetTensor4dDescriptor(ts_out, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, channels_B, height_B, width_B));

    cudnnFilterDescriptor_t kernel;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, channels_B, channels_A, 1, 1));

    cudnnConvolutionDescriptor_t conv;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t algo;
    cudnnConvolutionFwdAlgoPerf_t *pref_algos = new cudnnConvolutionFwdAlgoPerf_t[1];
    int returnedAlgoCount;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(h_cudnn, ts_in, kernel, conv, ts_out, 1, &returnedAlgoCount, pref_algos));
    algo = pref_algos[0].algo;

    size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(h_cudnn, ts_in, kernel, conv, ts_out, algo, &workspace_size));
    // 分配工作区空间
    void * workspace;
    cudaMalloc(&workspace, workspace_size);
    // =================================================线性因子
    float alpha = 1.0f;
    float beta  = -100.0f;
    // =================================================数据准备
    const Dtype* d_A, d_B, d_K;
    cudaMalloc((void**)&d_A, width_A * height_A * sizeof(Dtype));
    cudaMalloc((void**)&d_B, width_B * height_B * sizeof(Dtype));
    cudaMalloc((void**)&d_K, 1 * 1 * sizeof (Dtype) * sizeof(Dtype));

    cudaMemcpy((void*)d_A, (void*)A, width_A * height_A * sizeof(Dtype), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_K, (void*)K, 1 * 1 * sizeof(Dtype) * sizeof(Dtype), cudaMemcpyHostToDevice);
    // =================================================执行
    checkCUDNN(cudnnConvolutionForward(
            h_cudnn,
            &alpha,
            ts_in,
            d_A,                         // 输入
            kernel,
            d_K,                      // 核
            conv,
            algo,
            workspace,
            workspace_size,
            &beta,
            ts_out,
            d_B                         // 输出
    ));
    cudaMemcpy((void*)B, (void*)d_B, width_B * height_B * sizeof(Dtype), cudaMemcpyDeviceToHost);

    // 释放cuDNN
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(ts_in);
    cudnnDestroyTensorDescriptor(ts_out);
    cudnnDestroyConvolutionDescriptor(conv);
    cudnnDestroyFilterDescriptor(kernel);
    cudnnDestroy(h_cudnn);
}


template <typename Dtype>
void Relu(const Dtype* A, const Dtype* B,
          const int channels, const int height, const int width){
    cudnnHandle_t h_cudnn;
    checkCUDNN(cudnnCreate(&h_cudnn));

    cudnnActivationDescriptor_t acti_dec;
    checkCUDNN(cudnnCreateActivationDescriptor(&acti_dec));
    checkCUDNN(cudnnSetActivationDescriptor(&acti_dec,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN));

    cudnnTensorDescriptor_t ts_in;

    checkCUDNN(cudnnCreateTensorDescriptor(&ts_in));
    checkCUDNN(cudnnSetTensor4dDescriptor(ts_in, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, channels, height, width));

    cudnnTensorDescriptor_t ts_out;
    checkCUDNN(cudnnCreateTensorDescriptor(&ts_out));
    checkCUDNN(cudnnSetTensor4dDescriptor(ts_out, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, channels, height, width));

    // =================================================线性因子
    float alpha = 1.0f;
    float beta  = -100.0f;
    // =================================================数据准备
    const Dtype* d_A, d_B, d_K;
    cudaMalloc((void**)&d_A, width * height * sizeof(Dtype));
    cudaMalloc((void**)&d_B, width * height * sizeof(Dtype));

    cudaMemcpy((void*)d_A, (void*)A, width * height * sizeof(Dtype), cudaMemcpyHostToDevice);

    // =================================================执行
    checkCUDNN(cudnnActivationForward(h_cudnn,acti_dec,&alpha,ts_in,d_A,&beta,ts_out,d_B));

    cudaMemcpy((void*)B, (void*)d_B, width * height * sizeof(Dtype), cudaMemcpyDeviceToHost);

    // 释放cuDNN
    cudnnDestroyTensorDescriptor(ts_in);
    cudnnDestroyTensorDescriptor(ts_out);
    cudnnDestroy(h_cudnn);
}