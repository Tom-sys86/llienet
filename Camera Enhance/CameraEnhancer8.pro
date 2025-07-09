QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
QMAKE_CXXFLAGS += -std=c++17
QT += concurrent

INCLUDEPATH += /home/liaopeiwen/MAIN/opencv/install/include/opencv4 \
               $$PWD/rknn_inference/runtime/Linux/librknn_api/include \
               #$$PWD/rknn_inference/3rdparty/cnpy \
              # $$PWD/rknn_inference/3rdparty/fp16 \
               #$$PWD/rknn_inference/3rdparty/opencl/libopencl-stub/include/CL \
               $$PWD/rknn_inference/3rdparty/rga/include \
               $$PWD/rknn_inference/3rdparty/rk_mpi_mmz/include

LIBS += -L/home/liaopeiwen/MAIN/opencv/install/lib \
        -L$$PWD/rknn_inference/runtime/Linux/librknn_api/aarch64 \
        -L/home/liaopeiwen/CameraEnhancer8/rknn_inference/3rdparty/rga/libs/Linux/gcc-aarch64 \
        -L$$PWD/rknn_inference/3rdparty/rk_mpi_mmz/lib/Linux/aarch64


LIBS += \
         -lrk_mpi_mmz \  # 内存管理库
         -lrga \         # 瑞芯微图形加速库
        # -lOpenCL \      # 开放计算库
         #-lcnpy \        # NumPy数据操作库
         #-lfp16 \        # 半精度浮点运算库
         -lopencv_core \
         -lopencv_imgproc \
         -lopencv_highgui \
         -lopencv_videoio \
         -lopencv_imgcodecs \
         -lrknnrt         # RKNn运行时库


#QMAKE_CXXFLAGS += -march=armv8-a

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
