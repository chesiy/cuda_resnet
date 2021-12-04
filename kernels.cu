template<typename T>
__global__ void ConvolutionForward(const T* bottom_data,  T* top_data,const T* kernel,const T* bias,
                                   const int nthreads, const int channels, const int height, const int width,
                                   const int pooled_height, const int pooled_width,const int kernel_h,const int kernel_w,
                                   const int stride_h, const int stride_w, const int pad_h, const int pad_w)
/*
bottom_data-input
top_data-output feature
nthreads- the number of top_data
*/
{
    CUDA_KERNEL_LOOP(index,nthreads){
        //(n,c,ph,pw)
        const int pw=index%pooled_width;
        const int ph=(index/pooled_width)%pooled_height;
        const int c=(index/pooled_width/pooled_height)%channels;
        const int n=index/pooled_width/pooled_height/channels;

        const int k_c = c;

        int hstart=ph*stride_h-pad_h;
        int wstart=pw*stride_w-pad_w;
        const int hend=min(hstart+kernel_h,height);
        const int wend=min(wstart+kernel_w,width);
        hstart=max(hstart,0);
        wstart=max(wstart,0);

        T tmp= 0;

        const T* bottom_slice=bottom_data + n*channels * height * width;
        const T* kernel_slice = kernel + k_c * channels * kernel_h * kernel_w;
        for(int channel = 0; channel < channels; channel++){
            for(int h =hstart; h<hend; h++){
                for(int w=wstart;w<wend;w++){
                    tmp += bottom_slice[channel*height*width+h*width+w]*kernel_slice[channel*height*width+h*width+w];
                }
            }
        }
        tmp += bias[k_c];

        top_data[index]=tmp;
    }
}


template<typename T>
__global__ void MaxPoolForward(const T* bottom_data, const T* top_data,
const int nthreads, const int channels, const int height, const int width,
const int pooled_height, const int pooled_width,const int kernel_h,const int kernel_w,
const int stride_h, const int stride_w, const int pad_h, const int pad_w)
/*
bottom_data-input
top_data-output feature
nthreads- the number of top_data
*/
{
	CUDA_KERNEL_LOOP(index,nthreads){
		const int pw=index%pooled_width;
		const int ph=(index/pooled_width)%pooled_height;
		const int c=(index/pooled_width/pooled_height)%channels;
		const int n=index/pooled_width/pooled_height/channels;
		
		int hstart=ph*stride_h-pad_h;
		int wstart=pw*stride_w-pad_w;
		const int hend=min(hstart+kernel_h,height);
		const int wend=min(wstart+kernel_w,width);
		hstart=max(hstart,0);
		wstart=max(wstart,0);
		T maxval= -FLT_MAX;
		int maxidx = -1;
		const T* bottom_slice=bottom_data+(n*channels+c)*height*width;
		for(int h =hstart; h<hend; h++){
			for(int w=wstart;w<wend;w++){
				if(bottom_slice[h*width+w]>maxval){
					maxidx=h*width+w;
					maxval=bottom_slice[maxidx];
				}
			}
		}
		top_data[index]=maxval;
	}
}

template<typename T>
__global__ void relu(const T* A, T* B,const int nthreads)
//A-input B-output
{
	CUDA_KERNEL_LOOP(index,nthreads){
		if(A[index]>0){
			B[index]=A[index];
		}else{
			B[index]=0;
		}
		
	}
}

template<typename T>
__global__ void add(const T* A,const T* B, T* C,const int nthreads)
//A-input B-input C-output
{
	CUDA_KERNEL_LOOP(index,nthreads){
		C[index]=A[index]+B[index];
	}
}


template<typename T>
__global__ void AvgPoolForward(const T* bottom_data, const T* top_data,
const int nthreads, const int channels, const int height, const int width,
const int pooled_height, const int pooled_widht,const int kernel_h,const int kernel_w,
const int stride_h, const int stride_w, const int pad_h, const int pad_w)
/*
bottom_data-input
top_data-output feature
nthreads- the number of top_data
*/
{
	CUDA_KERNEL_LOOP(index,nthreads){
		const int pw=index%pooled_widht;
		const int ph=(index/pooled_widht)%pooled_height;
		const int c=(index/pooled_widht/pooled_height)%channels;
		const int n=index/pooled_widht/pooled_height/channels;
		
		int hstart=ph*stride_h-pad_h;
		int wstart=pw*stride_w-pad_w;
		const int hend=min(hstart+kernel_h,height);
		const int wend=min(wstart+kernel_w,width);
		hstart=max(hstart,0);
		wstart=max(wstart,0);
		
		T tmp= 0;
		int maxidx = -1;
		
		const T* bottom_slice=bottom_data+(n*channels+c)*height*width;
		for(int h =hstart; h<hend; h++){
			for(int w=wstart;w<wend;w++){
				tmp += bottom_slice[h*width+w];
				/*if(bottom_slice[h*width+w]>maxval){
					maxidx=h*width+w;
					maxval=bottom_slice[maxidx];
				}*/
			}
		}
		
		top_data[index]=tmp/((hend-hstart)*(wend-wstart));
		
	}
}