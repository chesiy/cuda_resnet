//
// Created by admin on 2021/11/22.
//
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <assert.h>
#include <vector>

const char* model_path = "resnet18.onnx";

int main(){
    //设置为VERBOSE，方便控制台输出时看到是使用了cpu还是gpu执行
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);

    session_options.SetIntraOpNumThreads(5);
    // 第二个参数代表GPU device_id = 0，注释这行就是cpu执行
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    // ORT_ENABLE_ALL: To Enable All possible opitmizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::AllocatorWithDefaultOptions allocator;
    // 获得模型又多少个输入和输出，一般是指对应网络层的数目
    // 一般输入只有图像的话input_nodes为1
    size_t num_input_nodes = session.GetInputCount();
    // 如果是多输出网络，就会是对应输出的数目
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
    printf("Number of inputs = %zu\n", num_input_nodes);
    printf("Number of output = %zu\n", num_output_nodes);
    // 迭代所有输出层信息
    for (int i = 0; i < num_output_nodes; i++) {
        char* output_name = session.GetOutputName(i, allocator);
        printf("Output %d : name=%s\n", i, output_name);
        // 将输出层的名称添加到output_node_names这个vector
        output_node_names[i] = output_name;

        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);

        output_node_dims = tensor_info.GetShape();
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < output_node_dims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }
    // 获取所有输入层信息
    for (int i = 0; i < num_input_nodes; i++) {
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }
    // 输入的总大小(所有像素)
    size_t input_tensor_size = 256 * 256 * 3;
    // 生成假的输入
    std::vector<float> input_tensor_values(input_tensor_size);
    for (unsigned int i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    clock_t startTime, endTime;
    // 创建输入tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());
    startTime = clock();
    // 第二个参数代表输入节点的名称集合
    // 第四个参数1代表输入层的数目
    // 第五个参数代表输出节点的名称集合
    // 最后一个参数代表输出节点的数目
    // 除了第一个节点外，其他参数与原网络对应不上程序就会无法执行
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), num_output_nodes);
    endTime = clock();
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
    // 获取输出输出
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    // TODO 因为这里我的输出是个heat map，暂时还没完成这部分后处理功能
    // 计算运行时间
    std::cout << "The run time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
    printf("Done!\n");
    system("pause");

    return 0;
}

