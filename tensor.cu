//
// Created by admin on 2021/12/19.
//

template<class Dtype> struct tensor{
    Dtype* data;
    int width,height,channels,batch;
    //tensor shape: (batch, channels, width, height)
    tensor(Dtype* d, int w, int h, int c, int batch):data(d),width(w),height(h),channels(c),batch(batch){}
    tensor(const tensor<Dtype> &d){
        data = d.data;
        width = d.width;
        height = d.height;
        batch = d.batch;
        channels=d.channels;
    }
    tensor(){}
};
