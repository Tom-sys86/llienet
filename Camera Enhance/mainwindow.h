#ifndef MAINWINDOW_H
#define MAINWINDOW_H

// 首先包含 Qt 核心头文件
#include <QMainWindow>
#include <QTimer>
#include <QImage>
#include <QPixmap>
#include <QDebug>
#include <QApplication>
#include <QFile>
#include <QLabel>

// 然后包含 OpenCV
#include <opencv2/opencv.hpp>
#include <arm_neon.h>
#include <sys/time.h>
#include <fstream>
#include <vector>

// 最后包含 RKNN
#include "rknn_api.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

enum DisplayState {
    ORIGINAL,           // 显示原始图像
    TRADITIONAL_PROC,   // 显示传统算法处理结果
    DL_PROC             // 显示深度学习算法处理结果
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    // NEON 优化函数（传统算法）
    void neon_histogram_3ch(const uint8_t* data, int total_pixels, int* hist_r, int* hist_g, int* hist_b);
    void apply_lut_3ch_neon(const uint8_t *src, uint8_t *dst, int total_pixels,
                            const uint8_t *lut_r, const uint8_t *lut_g, const uint8_t *lut_b);
    cv::Mat limited_histogram_equalization_color_opt(const cv::Mat &input, int range_min, int range_max, int threshold);
    cv::Mat denoiseImage(const cv::Mat& input);
    cv::Mat sharpenImage(const cv::Mat& input);
    cv::Mat enhanceSaturation(const cv::Mat& input, float factor = 1.5f);

    // 深度学习处理函数
    cv::Mat dlPreprocessImage(const cv::Mat &input);
    cv::Mat dlPostprocessOutput(float *output_data);
    bool initRKNNModel(const char* model_path);
    cv::Mat processFrameWithDL(const cv::Mat &input);
    void displayImage(const cv::Mat& mat, QLabel* label);
private slots:
    void on_useButton_clicked();      // 传统算法启用
    void on_returnButton_clicked();   // 传统算法禁用
    void on_useButton1_clicked();     // 深度学习启用
    //void on_returnButton1_clicked();  // 深度学习禁用
    void updateVideoFrame();          // 更新摄像头画面

private:
    Ui::MainWindow *ui;               // UI 指针
    cv::VideoCapture camera;          // OpenCV 摄像头对象
    QTimer *timer;                    // 定时器

    DisplayState currentDisplayState;

    // 原始帧缓存
    cv::Mat currentFrame;
    // RKNN 相关变量
    rknn_context ctx;                 // RKNN 上下文
    bool isRKNNInitialized;           // RKNN 是否初始化成功
    int input_size;                   // 模型输入大小
    int output_size;
    int INPUT_WIDTH;                  // 模型输入宽度
    int INPUT_HEIGHT;                 // 模型输入高度
    int OUTPUT_WIDTH;
    int OUTPUT_HEIGHT;
    // 时间计算函数
    double get_time_diff(struct timeval *start, struct timeval *end);

};

#endif // MAINWINDOW_H
