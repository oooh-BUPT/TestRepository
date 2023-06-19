#include "mainwindow.h"
#include "ui_mainwindow.h"
//#include <opencv2/opencv.hpp>
#include "opencv.hpp"
#include<iostream>
#include<QDir>
#include<QTimer>
#include<assert.h>
#include <QObject>
#include <QDebug>
#include <QPointF>
#include<dnn.hpp>
#include<time.h>
#include<QDateTime>
using namespace std;
//using namespace cv;
QString path= QDir::currentPath();
/*Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"test");
Ort::SessionOptions session_options;
const wchar_t* model_path =L"D:\\FaceIdentify\\build-FaceIdentify-Desktop_Qt_5_9_9_MinGW_32bit-Debug\\test_model.onnx";*/
cv::dnn::Net net = cv::dnn::readNetFromONNX("D:\\FaceIdentify\\build-FaceIdentify-Desktop_Qt_5_9_9_MinGW_32bit-Debug\\test_model.onnx");  // 加载模型
int danger=0;
QString tag[11] ={"周杰伦","温皓麟","丁真","刘诗诗","周笔畅","周润发","吴彦祖","杨紫","赵丽颖","吴京","彭于晏"};
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    /*session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);*/
    classifier.load("D:\\opencv3.4.0\\OpenCV-MinGW-Build-OpenCV-3.4.5\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml");
    label_update = new QTimer(this);
    connect(label_update,&QTimer::timeout,this,&MainWindow::video_update);
    connect(label_update,&QTimer::timeout,this,&MainWindow::time_update);
}

MainWindow::~MainWindow()
{
    delete ui;
}

QImage mat2qim(cv::Mat & mat)
{
    cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    QImage qim((const unsigned char*)mat.data, mat.cols, mat.rows, mat.step,
        QImage::Format_RGB888);
    return qim;
}

void MainWindow::on_openbt_clicked()
{
 /*   QString imgpath="D:\\FaceIdentify\\FaceIdentify\\test.jpeg";
    cv::Mat img;
    img=cv::imread(imgpath.toStdString());
    QPixmap mapimg=QPixmap::fromImage(mat2qim(img));
    ui->label_6->resize(mapimg.width(),mapimg.height());
    ui->label_6->setPixmap(mapimg);*/

    if(flag == 0){
        cap.open(0);
        label_update->start(30);
        flag=1;
        ui->openbt->setText("运行中");
    }
    else{
        label_update->stop();
        cap.release();
        ui->label_2->clear();
        ui->contentlabel_3->clear();
        ui->openbt->setText("开始");
        flag = 0;
    }
}


void MainWindow::video_update()
{
    cap>>video_image;
    cv::cvtColor(video_image,video_image,cv::COLOR_BGR2RGB);
    cv::Mat gray;
    vector<cv::Rect> faces;
    cv::cvtColor(video_image,gray,cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray,gray);
    classifier.detectMultiScale(gray,faces,1.2,3);
    for (int i = 0; i < int(faces.size()); i++)
        {
            cv::RNG rng(i);
            cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0,255), 20);
            cv::rectangle(video_image, faces[static_cast<int>(i)], color, 2, 8, 0);
            cv::Size dsize = cv::Size(224, 224);
            cv::resize(gray(faces[static_cast<int>(i)]),gray,dsize);
            gray.convertTo(gray, CV_32FC3, 1.0f / 255.0f);
            cv::cvtColor(gray,gray,cv::COLOR_GRAY2BGR);
            cv::Mat blob = cv::dnn::blobFromImage(gray);
            float value = -0.5;
            blob = blob + value;
            blob=blob/0.5;// 由图片加载数据 还可以进行缩放、归一化等预处理操作
            net.setInput(blob);  // 设置模型输入
            cv::Mat predict = net.forward(); // 推理结果
            double minValue, maxValue;
            cv::Point minIdx, maxIdx;
            cv::minMaxLoc(predict, &minValue, &maxValue, &minIdx, &maxIdx);
            //std::cout<<maxValue;
            if(maxValue<8)
            {
                ui->contentlabel_3->setText(QString("UNKNOWN"));
                danger++;
                if(danger>100)
                 {
                    cv::imwrite("D:\\FaceIdentify\\build-FaceIdentify-Desktop_Qt_5_9_9_MinGW_32bit-Debug\\unkown\\1.jpg",gray);
                }
            }
            else
            {
                ui->contentlabel_3->setText(QString(tag[maxIdx.x-1]));
            }
        }
    cv::resize(video_image,video_image,cv::Size(video_image.cols/2,video_image.rows/2));
    ui->label_2->setPixmap(QPixmap::fromImage(QImage(video_image.data,video_image.cols,video_image.rows,QImage::Format_RGB888)));
}


void MainWindow::time_update()
{
    QDateTime dateTime= QDateTime::currentDateTime();
    QString hour = dateTime .toString("hh:mm:ss");
    QString day = dateTime.toString("yyyy-MM-dd");
    ui->label_5->setText(day);
    ui->label_4->setText(hour);
}
