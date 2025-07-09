#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <chrono>
#include <stdexcept>
#include <cstring>
#include <QElapsedTimer> // 添加高精度计时器头文件

using namespace cv;

// ======================================================================
// 传统图像处理算法
// ======================================================================

// NEON优化的三通道直方图统计
void MainWindow::neon_histogram_3ch(const uint8_t* data, int total_pixels, int* hist_r, int* hist_g, int* hist_b) {
    // 实现保持不变
    const int block_size = 16;
    const int main_loop = total_pixels / block_size;
    const int remainder = total_pixels % block_size;

    const uint8_t* ptr = data;

    int local_hist_r[256] = {0};
    int local_hist_g[256] = {0};
    int local_hist_b[256] = {0};

    for (int i = 0; i < main_loop; ++i) {
        uint8x16x3_t vec = vld3q_u8(ptr);
        ptr += block_size * 3;

        // R通道处理
        for (int j = 0; j < 16; ++j) {
            local_hist_r[vec.val[0][j]]++;
        }
        // G通道处理
        for (int j = 0; j < 16; ++j) {
            local_hist_g[vec.val[1][j]]++;
        }
        // B通道处理
        for (int j = 0; j < 16; ++j) {
            local_hist_b[vec.val[2][j]]++;
        }

        if ((i + 1) % 4 == 0) {
            __builtin_prefetch(ptr + 256, 0, 3);
        }
    }

    // 处理剩余像素
    for (int i = 0; i < remainder; ++i) {
        local_hist_r[ptr[0]]++;
        local_hist_g[ptr[1]]++;
        local_hist_b[ptr[2]]++;
        ptr += 3;
    }

    // 合并到全局直方图
    for (int i = 0; i < 256; ++i) {
        hist_r[i] += local_hist_r[i];
        hist_g[i] += local_hist_g[i];
        hist_b[i] += local_hist_b[i];
    }
}

// NEON优化的LUT应用
void MainWindow::apply_lut_3ch_neon(const uint8_t *src, uint8_t *dst, int total_pixels,
                                    const uint8_t *lut_r, const uint8_t *lut_g, const uint8_t *lut_b) {
    // 实现保持不变
    const int block_size = 16;
    int main_loop = total_pixels / block_size;
    int remain = total_pixels % block_size;

    const uint8_t *src_ptr = src;
    uint8_t *dst_ptr = dst;

    for (int i = 0; i < main_loop; ++i) {
        uint8x16x3_t vec = vld3q_u8(src_ptr);
        src_ptr += block_size * 3;

        uint8x16_t r = vec.val[0];
        uint8x16_t g = vec.val[1];
        uint8x16_t b = vec.val[2];

        uint8_t r_lut[16], g_lut[16], b_lut[16];
        for (int j = 0; j < 16; ++j) {
            r_lut[j] = lut_r[r[j]];
            g_lut[j] = lut_g[g[j]];
            b_lut[j] = lut_b[b[j]];
        }

        uint8x16x3_t result;
        result.val[0] = vld1q_u8(r_lut);
        result.val[1] = vld1q_u8(g_lut);
        result.val[2] = vld1q_u8(b_lut);

        vst3q_u8(dst_ptr, result);
        dst_ptr += block_size * 3;
    }

    // 处理剩余像素
    for (int i = 0; i < remain; ++i) {
        dst_ptr[0] = lut_r[src_ptr[0]];
        dst_ptr[1] = lut_g[src_ptr[1]];
        dst_ptr[2] = lut_b[src_ptr[2]];
        src_ptr += 3;
        dst_ptr += 3;
    }
}

// 限制直方图均衡化主函数
cv::Mat MainWindow::limited_histogram_equalization_color_opt(const cv::Mat &input, int range_min, int range_max, int threshold) {
    // 实现保持不变
    Mat input_cont = input.isContinuous() ? input : input.clone();
    const int total = input_cont.rows * input_cont.cols;
    Mat output(input_cont.size(), input_cont.type());

    int hist_r[256] = {0};
    int hist_g[256] = {0};
    int hist_b[256] = {0};

    // 步骤1：直方图统计
    neon_histogram_3ch(input_cont.ptr<uint8_t>(), total, hist_r, hist_g, hist_b);

    // 步骤2：生成LUT
    uint8_t lut_r[256], lut_g[256], lut_b[256];

    // R通道LUT生成
    auto generate_lut = [&](int* hist, uint8_t* lut) {
        int sum_range = 0;
        int pixels_above_threshold = 0;
        for (int i = range_min; i < range_max; ++i) {
            sum_range += hist[i];
            if (hist[i] > threshold) {
                pixels_above_threshold += hist[i] - threshold;
                hist[i] = threshold;
            }
        }
        if (pixels_above_threshold > 0) {
            for (int i = range_min; i < range_max; ++i) {
                hist[i] += pixels_above_threshold / (range_max - range_min + 1);
            }
        }

        if (sum_range == 0) {
            for (int i = 0; i < 256; ++i) lut[i] = i;
        } else {
            float scale = (range_max - range_min) / (float)sum_range;
            int cdf = 0;
            for (int i = range_min; i <= range_max; ++i) {
                cdf += hist[i];
                lut[i] = saturate_cast<uint8_t>(range_min + cdf * scale);
            }
            for (int i = 0; i < range_min; ++i) lut[i] = i;
            for (int i = range_max + 1; i < 256; ++i) lut[i] = i;
        }
    };

    generate_lut(hist_r, lut_r);
    generate_lut(hist_g, lut_g);
    generate_lut(hist_b, lut_b);

    // 步骤3：应用LUT
    apply_lut_3ch_neon(input_cont.ptr<uint8_t>(), output.ptr<uint8_t>(), total, lut_r, lut_g, lut_b);

    return output;
}

// 降噪函数
cv::Mat MainWindow::denoiseImage(const cv::Mat& input) {
    // 实现保持不变
    if (input.empty()) return input;

    cv::Mat denoised;
    double sigmaColor = 50 + (cv::mean(input)[0]/255)*25;
    double sigmaSpace = sigmaColor*0.5;
    cv::bilateralFilter(input, denoised, 9, sigmaColor, sigmaSpace);

    return denoised;
}

// 锐化函数
cv::Mat MainWindow::sharpenImage(const cv::Mat& input) {
    // 实现保持不变
    if (input.empty()) return input;

    cv::Mat sharpened;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
                          0, -1,  0,
                      -1,  5, -1,
                      0, -1,  0);

    cv::filter2D(input, sharpened, input.depth(), kernel);

    return sharpened;
}

// 色彩饱和度增强
cv::Mat MainWindow::enhanceSaturation(const cv::Mat& input, float factor) {
    // 实现保持不变
    if (input.empty()) return input;

    // 参数有效性检查
    factor = std::max(0.0f, std::min(factor, 10.0f));

    // 统一处理为3通道
    cv::Mat src;
    if(input.channels() == 4) {
        cv::cvtColor(input, src, cv::COLOR_BGRA2BGR);
    } else {
        src = input.clone();
    }

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    // 浮点运算避免溢出
    cv::Mat sat = channels[1].clone();
    sat.convertTo(sat, CV_32F);
    sat *= factor;
    sat.convertTo(sat, CV_8U, 1, 0);  // 四舍五入转换
    cv::threshold(sat, sat, 255, 255, cv::THRESH_TRUNC);

    channels[1] = sat;

    cv::merge(channels, hsv);
    cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);

    return hsv;
}

// ======================================================================
// 深度学习图像处理算法
// ======================================================================

// 时间差计算
double MainWindow::get_time_diff(struct timeval *start, struct timeval *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_usec - start->tv_usec) / 1000.0;
}

// 图像预处理（替换为main.cc的实现）
cv::Mat MainWindow::dlPreprocessImage(const cv::Mat &input) {
    cv::Mat resized_img;
    cv::resize(input, resized_img, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

    // BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);

    return rgb_img;
}

// 后处理（替换为main.cc的实现）
cv::Mat MainWindow::dlPostprocessOutput(float *output_data) {
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

    // RGB to BGR (在Qt显示时需要RGB，所以这里不需要转换)
    // cv::cvtColor(uint8_img, uint8_img, cv::COLOR_RGB2BGR);

    return uint8_img;
}

// 初始化RKNN模型（替换为main.cc的实现）
bool MainWindow::initRKNNModel(const char* model_path) {
    int ret;
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        qDebug() << "Error: 无法打开模型文件";
        return false;
    }

    fseek(fp, 0, SEEK_END);
    size_t model_size = ftell(fp);
    rewind(fp);

    unsigned char* model_data = new unsigned char[model_size];
    if (fread(model_data, 1, model_size, fp) != model_size) {
        qDebug() << "Error: 读取模型文件失败";
        delete[] model_data;
        fclose(fp);
        return false;
    }
    fclose(fp);

    // 初始化RKNN
    qDebug() << "--> 初始化RKNN模型...";
    ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    delete[] model_data; // 立即释放内存

    if (ret != RKNN_SUCC) {
        qDebug() << "Error: rknn_init failed, ret=" << ret;
        return false;
    }

    // 3. 获取模型输入输出信息
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        qDebug() << "Error: rknn_query failed, ret=" << ret;
        return false;
    }
    qDebug() << "模型输入数量:" << io_num.n_input << "输出数量:" << io_num.n_output;

    // 4. 获取输出属性
    rknn_tensor_attr output_attr;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
    if (ret != RKNN_SUCC) {
        qDebug() << "Error: rknn_query output attr failed, ret=" << ret;
        return false;
    }

    // 打印输出张量信息
    qDebug() << "输出张量维度:";
    for (int i = 0; i < 8; i++) {
        qDebug() << output_attr.dims[i];
    }

    qDebug() << "输出张量数据类型:";
    switch (output_attr.type) {
    case RKNN_TENSOR_UINT8:
        qDebug() << "RKNN_TENSOR_UINT8";
        break;
    case RKNN_TENSOR_FLOAT32:
        qDebug() << "RKNN_TENSOR_FLOAT32";
        break;
    default:
        qDebug() << "未知类型:" << output_attr.type;
        break;
    }

    qDebug() << "输出张量格式:";
    switch (output_attr.fmt) {
    case RKNN_TENSOR_NHWC:
        qDebug() << "RKNN_TENSOR_NHWC";
        break;
    case RKNN_TENSOR_NCHW:
        qDebug() << "RKNN_TENSOR_NCHW";
        break;
    default:
        qDebug() << "未知格式:" << output_attr.fmt;
        break;
    }

    // 5. 设置输入大小
    input_size = INPUT_WIDTH * INPUT_HEIGHT * 3;

    isRKNNInitialized = true;
    return true;
}

// 使用深度学习处理帧
cv::Mat MainWindow::processFrameWithDL(const cv::Mat &input) {
    if (!isRKNNInitialized) {
        qDebug() << "RKNN 模型未初始化!";
        return input.clone();
    }

    struct timeval start, end;
    int ret;

    // 1. 预处理图像
    cv::Mat preprocessed = dlPreprocessImage(input);
    if (preprocessed.empty()) {
        qDebug() << "预处理失败! 输入尺寸: "
                 << input.cols << "x" << input.rows;
        return input.clone();
    }

    // 2. 配置输入张量
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = input_size;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = preprocessed.data;

    // 3. 设置输入
    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret != RKNN_SUCC) {
        qDebug() << "Error: rknn_inputs_set failed, ret=" << ret;
        return input.clone();
    }

    // 4. 执行推理
    gettimeofday(&start, NULL);
    ret = rknn_run(ctx, NULL);
    gettimeofday(&end, NULL);
    if (ret != RKNN_SUCC) {
        qDebug() << "Error: rknn_run failed, ret=" << ret;
        return input.clone();
    }
    double inference_time = get_time_diff(&start, &end);
    qDebug() << "深度学习推理耗时:" << inference_time << "ms";

    // 5. 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    outputs[0].is_prealloc = 0;
    outputs[0].want_float = 1;  // 获取浮点型输出

    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret != RKNN_SUCC) {
        qDebug() << "Error: rknn_outputs_get failed, ret=" << ret;
        return input.clone();
    }

    // 6. 后处理
    cv::Mat result = dlPostprocessOutput((float *)outputs[0].buf);

    cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
    // 7. 释放输出
    rknn_outputs_release(ctx, 1, outputs);

    return result;
}

// ======================================================================
// 辅助函数：显示图像到QLabel
// ======================================================================

void MainWindow::displayImage(const cv::Mat& mat, QLabel* label) {
    if (mat.empty()) {
        label->setText("图像为空");
        return;
    }

    // 转换为RGB格式
    cv::Mat displayMat;
    if (mat.channels() == 1) {
        cv::cvtColor(mat, displayMat, cv::COLOR_GRAY2RGB);
    } else if (mat.channels() == 3) {
        cv::cvtColor(mat, displayMat, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, displayMat, cv::COLOR_BGRA2RGB);
    } else {
        displayMat = mat.clone();
    }

    // 调整大小以适应显示控件
    QSize labelSize = label->size();
    if (labelSize.width() > 0 && labelSize.height() > 0) {
        cv::Mat resizedMat;
        cv::resize(displayMat, resizedMat, cv::Size(labelSize.width(), labelSize.height()));
        displayMat = resizedMat;
    }

    // 创建QImage并显示
    QImage img(displayMat.data, displayMat.cols, displayMat.rows,
               displayMat.step, QImage::Format_RGB888);
    label->setPixmap(QPixmap::fromImage(img));
}

// ======================================================================
// Qt 主窗口逻辑
// ======================================================================

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
    ui(new Ui::MainWindow),
    isRKNNInitialized(false),
    input_size(0),
    output_size(0),
    INPUT_WIDTH(600),
    INPUT_HEIGHT(400),
    OUTPUT_WIDTH(600),
    OUTPUT_HEIGHT(400) {

    // 初始化 UI
    ui->setupUi(this);

    // 设置窗口标题
    this->setWindowTitle("图像增强算法对比");

    // 初始化显示状态
    currentDisplayState = ORIGINAL;

    // 样式表保持不变
    this->setStyleSheet(R"(
    QWidget {
        background-color: #F5F5F5;
        font-family: "Microsoft YaHei";
    }
    QLabel {
        color: #333;
        font-size: 14px;
    }
    QPushButton {
        border-radius: 5px;
        padding: 8px 20px;
        font-size: 14px;
    }
    /* 初始状态 - 蓝色按钮 */
    QPushButton#useButton,
    QPushButton#useButton1 {
        background-color: #007BFF;
        color: white;
    }
    /* 初始状态 - 灰色按钮 */
    QPushButton#returnButton {
        background-color: #6C757D;
        color: white;
    }
    /* 悬停状态 */
    QPushButton#useButton:hover,
    QPushButton#useButton1:hover {
        background-color: #0056b3;
    }
    QPushButton#returnButton:hover {
        background-color: #5a6268;
    }
    /* 按下效果 */
    QPushButton:pressed {
        padding-top: 10px;
        padding-bottom: 6px;
    }
    /* 新增：激活状态（红色） */
    QPushButton#useButton[activated="true"] {
        background-color: #DC3545;  /* 红色 */
    }
    QPushButton#useButton1[activated="true"] {
        background-color: #DC3545;  /* 红色 */
    }
)");

    // 删除 returnButton1 按钮
    //delete ui->returnButton1;
    //ui->returnButton1 = nullptr;

    // 设置按钮文本
    ui->useButton->setText("改进直方图算法");
    ui->returnButton->setText("返回原始图像");
    ui->useButton1->setText("llienet深度学习算法");
    ui->label->setText("处理后图像");
    ui->label_2->setText("原始图像");

    // 初始化摄像头
    camera.open("/dev/video21"); // 注意：路径可能需要调整
    if (!camera.isOpened()) {
        qDebug() << "Error: 无法打开摄像头";
        // 显示错误提示
        ui->originalDisplay->setText("摄像头初始化失败");
        ui->videoDisplay->setText("摄像头初始化失败");
    } else {
        // 初始化深度学习模型
        QString modelPath = "/root/CameraEnhancer8/llienet_model.rknn";
        qDebug() << "模型路径:" << modelPath;

        if (!initRKNNModel(modelPath.toStdString().c_str())) {
            qDebug() << "深度学习模型初始化失败!";
            ui->videoDisplay->setText("深度学习模型初始化失败");
        } else {
            qDebug() << "RKNN模型初始化成功!";
        }
    }

    // 设置定时器
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateVideoFrame);
    timer->start(30); // 约33fps
}

MainWindow::~MainWindow() {
    delete ui;

    // 释放摄像头
    if (camera.isOpened()) {
        camera.release();
    }

    // 释放 RKNN 资源
    if (isRKNNInitialized) {
        rknn_destroy(ctx);
    }
}

// "传统算法"按钮点击
void MainWindow::on_useButton_clicked() {
    currentDisplayState = TRADITIONAL_PROC;
    qDebug() << "切换到传统算法处理";

    // 更新按钮状态
    ui->useButton->setProperty("activated", true);
    ui->useButton->style()->unpolish(ui->useButton);
    ui->useButton->style()->polish(ui->useButton);

    ui->useButton1->setProperty("activated", false);
    ui->useButton1->style()->unpolish(ui->useButton1);
    ui->useButton1->style()->polish(ui->useButton1);
}

// "深度学习"按钮点击
void MainWindow::on_useButton1_clicked() {
    currentDisplayState = DL_PROC;
    qDebug() << "切换到深度学习处理";

    // 更新按钮状态
    ui->useButton1->setProperty("activated", true);
    ui->useButton1->style()->unpolish(ui->useButton1);
    ui->useButton1->style()->polish(ui->useButton1);

    ui->useButton->setProperty("activated", false);
    ui->useButton->style()->unpolish(ui->useButton);
    ui->useButton->style()->polish(ui->useButton);
}

// "返回原始"按钮点击
void MainWindow::on_returnButton_clicked() {
    currentDisplayState = ORIGINAL;
    qDebug() << "返回原始图像";

    // 清除按钮激活状态
    ui->useButton->setProperty("activated", false);
    ui->useButton->style()->unpolish(ui->useButton);
    ui->useButton->style()->polish(ui->useButton);

    ui->useButton1->setProperty("activated", false);
    ui->useButton1->style()->unpolish(ui->useButton1);
    ui->useButton1->style()->polish(ui->useButton1);
}

// 更新摄像头画面
void MainWindow::updateVideoFrame() {
    static cv::Mat currentFrame;
    camera >> currentFrame; // 捕获一帧
    if (currentFrame.empty()) {
        qDebug() << "摄像头帧为空!";
        return;
    }

    // 左侧始终显示原始图像
    displayImage(currentFrame, ui->videoDisplay);

    // 右侧根据当前状态显示不同内容
    switch(currentDisplayState) {
    case ORIGINAL:
        // 显示原始图像
        displayImage(currentFrame, ui->originalDisplay);
        break;

    case TRADITIONAL_PROC:
        // 显示传统算法处理结果
        try {
            // 开始计时
            QElapsedTimer timer;
            timer.start();

            Mat traditionalResult = currentFrame.clone();

            // 应用传统图像处理流程
            traditionalResult = denoiseImage(traditionalResult);
            traditionalResult = limited_histogram_equalization_color_opt(traditionalResult, 1, 220, 80000);
            traditionalResult = enhanceSaturation(traditionalResult, 1.5f);
            traditionalResult = sharpenImage(traditionalResult);

            // 计算处理时间
            qint64 elapsed = timer.elapsed();

            // 在控制台输出处理时间
            qDebug() << "传统算法处理时间:" << elapsed << "ms";

            // 显示处理结果
            displayImage(traditionalResult, ui->originalDisplay);
        } catch (const cv::Exception& e) {
            qDebug() << "传统算法处理错误:" << e.what();
            // 出错时显示原始图像
            displayImage(currentFrame, ui->originalDisplay);
        }
        break;

    case DL_PROC:
        // 显示深度学习处理结果
        try {
            Mat dlResult = processFrameWithDL(currentFrame);
            if (!dlResult.empty()) {
                displayImage(dlResult, ui->originalDisplay);
            } else {
                qDebug() << "深度学习处理结果为空!";
                // 出错时显示原始图像
                displayImage(currentFrame, ui->originalDisplay);
            }
        } catch (const cv::Exception& e) {
            qDebug() << "深度学习处理错误:" << e.what();
            // 出错时显示原始图像
            displayImage(currentFrame, ui->originalDisplay);
        } catch (const std::exception& e) {
            qDebug() << "深度学习处理错误:" << e.what();
            // 出错时显示原始图像
            displayImage(currentFrame, ui->originalDisplay);
        }
        break;
    }
}
