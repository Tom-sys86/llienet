#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>

// RKNN头文件
#include "rknn_api.h"

// 模型配置参数
#define INPUT_WIDTH 600
#define INPUT_HEIGHT 400
#define INPUT_CHANNEL 3
#define OUTPUT_WIDTH 600
#define OUTPUT_HEIGHT 400

// 计算时间差（毫秒）
double get_time_diff(struct timeval *start, struct timeval *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + 
           (end->tv_usec - start->tv_usec) / 1000.0;
}

// 图像预处理
int preprocess_image(const char *image_path, uint8_t **input_data, int *data_size) {
    // 读取图像
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        printf("Error: 无法读取图像 %s\n", image_path);
        return -1;
    }
    
    // 调整大小
    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    
    // BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);
    
    // 分配内存 (NHWC 格式)
    *data_size =1 * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNEL *sizeof(uint8_t);
    *input_data = (uint8_t *)malloc(*data_size);
    if (!*input_data) {
        printf("Error: 内存分配失败\n");
        return -1;
    }

    // 复制数据到 NHWC 格式的内存中
    memcpy(*input_data, rgb_img.data, *data_size);
    return 0;
}

// 后处理并保存结果
int postprocess_and_save_output(float *output_data, const char *save_path) {
       
    // 1. 创建NCHW格式的Mat (1, 3, height, width)
    cv::Mat nchw_mat(3, OUTPUT_HEIGHT * OUTPUT_WIDTH, CV_32F, output_data);
    
    // 2. 将NCHW拆分为各个通道
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; c++) {
        channels[c] = cv::Mat(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_32F, 
                            nchw_mat.ptr<float>(c));
    }
    
    // 3. 合并为NHWC格式 (height, width, channels)
    cv::Mat nhwc_mat;
    cv::merge(channels, nhwc_mat);

    // 反归一化 (0.0-1.0 -> 0-255)
    cv::Mat uint8_img;
    nhwc_mat.convertTo(uint8_img, CV_8UC3, 255.0);
    
    // RGB to BGR
    cv::Mat bgr_img;
    cv::cvtColor(uint8_img, bgr_img, cv::COLOR_RGB2BGR);
    
    // 保存图像
    cv::imwrite(save_path, bgr_img);
    printf("结果已保存至: %s\n", save_path);
    
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("用法: %s <rknn模型路径> <输入图像路径>\n", argv[0]);
        return -1;
    }
    
    char *model_path = argv[1];
    char *image_path = argv[2];
    const char *output_path = "output_result.png";
    
    rknn_context ctx;
    int ret;
    uint8_t *input_data = NULL;
    int input_size = 0;
    struct timeval start, end;
    double inference_time;
    
    // 1. 图像预处理
    printf("--> 预处理图像...\n");
    ret = preprocess_image(image_path, &input_data, &input_size);
    if (ret != 0) {
        goto cleanup;
    }
    
    // 2. 初始化RKNN上下文
    printf("--> 初始化RKNN模型...\n");
    ret = rknn_init(&ctx, model_path, 0, 0,NULL);
    if (ret != 0) {
        printf("Error: rknn_init failed, ret=%d\n", ret);
        goto cleanup;
    }
    
    // 3. 获取模型输入输出信息
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) {
        printf("Error: rknn_query failed, ret=%d\n", ret);
        goto cleanup;
    }
    printf("模型输入数量: %d, 输出数量: %d\n", io_num.n_input, io_num.n_output);
    
    // 4. 配置输入张量
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = input_size;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = input_data;
    
    // 5. 设置输入
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret != 0) {
        printf("Error: rknn_inputs_set failed, ret=%d\n", ret);
        goto cleanup;
    }
    
    // 6. 执行推理
    printf("--> 执行推理...\n");
    gettimeofday(&start, NULL);
    ret = rknn_run(ctx, NULL);
    gettimeofday(&end, NULL);
    if (ret != 0) {
        printf("Error: rknn_run failed, ret=%d\n", ret);
        goto cleanup;
    }
    inference_time = get_time_diff(&start, &end);
    printf("推理耗时: %.2f ms\n", inference_time);
    


rknn_tensor_attr output_attr;
memset(&output_attr, 0, sizeof(output_attr));
output_attr.index = 0;  // 指定查询第 0 个输出的属性
ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
if (ret != 0) {
    printf("Error: rknn_query output attribute failed, ret=%d\n", ret);
    goto cleanup;
}

// 打印输出张量的维度、格式等信息
printf("输出张量维度: ");
for (int i = 0; i < 8; i++) {
    printf("%d ", output_attr.dims[i]);
}
printf("\n");
printf("输出张量数据类型: ");
switch (output_attr.type) {
    case RKNN_TENSOR_UINT8:
        printf("RKNN_TENSOR_UINT8\n");
        break;
    case RKNN_TENSOR_FLOAT32:
        printf("RKNN_TENSOR_FLOAT32\n");
        break;
    // 可根据需要补充其他数据类型的判断和打印
    default:
        printf("未知类型: %d\n", output_attr.type);
        break;
}
printf("输出张量格式: ");
switch (output_attr.fmt) {
    case RKNN_TENSOR_NHWC:
        printf("RKNN_TENSOR_NHWC\n");
        break;
    case RKNN_TENSOR_NCHW:
        printf("RKNN_TENSOR_NCHW\n");
        break;
    // 可根据需要补充其他张量格式的判断和打印
    default:
        printf("未知格式: %d\n", output_attr.fmt);
        break;
}



    // 7. 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    outputs[0].is_prealloc = 0;  //表示有rknn来分配输出数据buff的存放
    outputs[0].want_float = 1;  // 获取浮点型输出
    
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret != 0) {
        printf("Error: rknn_outputs_get failed, ret=%d\n", ret);
        goto cleanup;
    }
    
    // 8. 后处理并保存结果
    printf("--> 后处理结果...\n");
    postprocess_and_save_output((float *)outputs[0].buf, output_path);
    
    // 9. 释放输出
    rknn_outputs_release(ctx, io_num.n_output, outputs);
    
cleanup:
    // 释放资源
    if (input_data) {
        free(input_data);
    }
    if (ctx) {
        rknn_destroy(ctx);
    }
    
    return ret;
}
