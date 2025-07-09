/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.6.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QLabel *originalDisplay;
    QWidget *horizontalLayoutWidget_3;
    QHBoxLayout *horizontalLayout_3;
    QLabel *videoDisplay;
    QWidget *horizontalLayoutWidget;
    QHBoxLayout *horizontalLayout;
    QPushButton *useButton;
    QWidget *horizontalLayoutWidget_2;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *returnButton;
    QWidget *horizontalLayoutWidget_4;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *useButton1;
    QWidget *horizontalLayoutWidget_5;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label;
    QWidget *horizontalLayoutWidget_6;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_2;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(960, 600);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        verticalLayoutWidget = new QWidget(centralwidget);
        verticalLayoutWidget->setObjectName(QStringLiteral("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(-10, 40, 481, 431));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        originalDisplay = new QLabel(verticalLayoutWidget);
        originalDisplay->setObjectName(QStringLiteral("originalDisplay"));

        verticalLayout->addWidget(originalDisplay);

        horizontalLayoutWidget_3 = new QWidget(centralwidget);
        horizontalLayoutWidget_3->setObjectName(QStringLiteral("horizontalLayoutWidget_3"));
        horizontalLayoutWidget_3->setGeometry(QRect(470, 40, 491, 431));
        horizontalLayout_3 = new QHBoxLayout(horizontalLayoutWidget_3);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        videoDisplay = new QLabel(horizontalLayoutWidget_3);
        videoDisplay->setObjectName(QStringLiteral("videoDisplay"));

        horizontalLayout_3->addWidget(videoDisplay);

        horizontalLayoutWidget = new QWidget(centralwidget);
        horizontalLayoutWidget->setObjectName(QStringLiteral("horizontalLayoutWidget"));
        horizontalLayoutWidget->setGeometry(QRect(0, 470, 311, 81));
        horizontalLayout = new QHBoxLayout(horizontalLayoutWidget);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        useButton = new QPushButton(horizontalLayoutWidget);
        useButton->setObjectName(QStringLiteral("useButton"));
        useButton->setMaximumSize(QSize(180, 70));
        QFont font;
        font.setPointSize(16);
        useButton->setFont(font);

        horizontalLayout->addWidget(useButton);

        horizontalLayoutWidget_2 = new QWidget(centralwidget);
        horizontalLayoutWidget_2->setObjectName(QStringLiteral("horizontalLayoutWidget_2"));
        horizontalLayoutWidget_2->setGeometry(QRect(630, 470, 331, 81));
        horizontalLayout_2 = new QHBoxLayout(horizontalLayoutWidget_2);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        returnButton = new QPushButton(horizontalLayoutWidget_2);
        returnButton->setObjectName(QStringLiteral("returnButton"));
        returnButton->setMaximumSize(QSize(180, 70));
        returnButton->setFont(font);
        returnButton->setIconSize(QSize(16, 16));

        horizontalLayout_2->addWidget(returnButton);

        horizontalLayoutWidget_4 = new QWidget(centralwidget);
        horizontalLayoutWidget_4->setObjectName(QStringLiteral("horizontalLayoutWidget_4"));
        horizontalLayoutWidget_4->setGeometry(QRect(310, 470, 321, 81));
        horizontalLayout_4 = new QHBoxLayout(horizontalLayoutWidget_4);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        useButton1 = new QPushButton(horizontalLayoutWidget_4);
        useButton1->setObjectName(QStringLiteral("useButton1"));
        useButton1->setMaximumSize(QSize(180, 70));
        useButton1->setFont(font);

        horizontalLayout_4->addWidget(useButton1);

        horizontalLayoutWidget_5 = new QWidget(centralwidget);
        horizontalLayoutWidget_5->setObjectName(QStringLiteral("horizontalLayoutWidget_5"));
        horizontalLayoutWidget_5->setGeometry(QRect(150, 0, 201, 41));
        horizontalLayout_5 = new QHBoxLayout(horizontalLayoutWidget_5);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        label = new QLabel(horizontalLayoutWidget_5);
        label->setObjectName(QStringLiteral("label"));
        QFont font1;
        font1.setPointSize(14);
        label->setFont(font1);

        horizontalLayout_5->addWidget(label);

        horizontalLayoutWidget_6 = new QWidget(centralwidget);
        horizontalLayoutWidget_6->setObjectName(QStringLiteral("horizontalLayoutWidget_6"));
        horizontalLayoutWidget_6->setGeometry(QRect(680, 0, 211, 41));
        horizontalLayout_6 = new QHBoxLayout(horizontalLayoutWidget_6);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(0, 0, 0, 0);
        label_2 = new QLabel(horizontalLayoutWidget_6);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setFont(font1);

        horizontalLayout_6->addWidget(label_2);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 960, 27));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        originalDisplay->setText(QApplication::translate("MainWindow", "TextLabel", 0));
        videoDisplay->setText(QApplication::translate("MainWindow", "TextLabel", 0));
        useButton->setText(QApplication::translate("MainWindow", "Use", 0));
        returnButton->setText(QApplication::translate("MainWindow", "Return", 0));
        useButton1->setText(QApplication::translate("MainWindow", "Use", 0));
        label->setText(QApplication::translate("MainWindow", "TextLabel", 0));
        label_2->setText(QApplication::translate("MainWindow", "TextLabel", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
