#include <assert.h>
#include <vector>
#include "onnxruntime_cxx_api.h"

int main(int argc, char* argv[]) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const char* model_path = "/home/group20/resnet18.onnx";

    printf("testing onnx......\n");
    Ort::Session session(env, model_path, session_options);
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> input_node_dims;
    printf("Number of inputs = %zu\n", num_input_nodes);

    Ort::ModelMetadata tmp = session.GetModelMetadata();

    char* aa = tmp.GetProducerName(allocator);
    printf("%s\n",aa);

    GetValue

    for (int i = 0; i < num_input_nodes; i++) {
        //输出输入节点的名称
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // 输出输入节点的类型
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        input_node_dims = tensor_info.GetShape();
        //输入节点的打印维度
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        //打印各个维度的大小
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
        //batch_size=1
        input_node_dims[0] = 1;
    }




//    std::vector<int64_t> input_node_dims = {10, 20};
//    size_t input_tensor_size = 10 * 20;
//    std::vector<float> input_tensor_values(input_tensor_size);
//    for (unsigned int i = 0; i < input_tensor_size; i++)
//        input_tensor_values[i] = (float)i / (input_tensor_size + 1);
//    // create input tensor object from data values
//    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 2);
//    assert(input_tensor.IsTensor());
//    printf("?????\n");
//    std::vector<int64_t> input_mask_node_dims = {1, 20, 4};
//    size_t input_mask_tensor_size = 1 * 20 * 4;
//    std::vector<float> input_mask_tensor_values(input_mask_tensor_size);
//    for (unsigned int i = 0; i < input_mask_tensor_size; i++)
//        input_mask_tensor_values[i] = (float)i / (input_mask_tensor_size + 1);
//    // create input tensor object from data values
//    auto mask_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//    Ort::Value input_mask_tensor = Ort::Value::CreateTensor<float>(mask_memory_info, input_mask_tensor_values.data(), input_mask_tensor_size, input_mask_node_dims.data(), 3);
//    assert(input_mask_tensor.IsTensor());
//    printf("male\n");
//    std::vector<Ort::Value> ort_inputs;
//    ort_inputs.push_back(std::move(input_tensor));
//    ort_inputs.push_back(std::move(input_mask_tensor));
//    // score model & input tensor, get back output tensor
//    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 2);
//    printf("out!!!\n");
//    // Get pointer to output tensor float values
//    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
//    float* floatarr_mask = output_tensors[1].GetTensorMutableData<float>();

//    printf("Done!\n");
    return 0;
}
