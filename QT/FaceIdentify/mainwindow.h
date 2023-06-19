#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include<opencv.hpp>
#include<QTimer>
#include<assert.h>
#include <QObject>
#include <QDebug>
#include <QPointF>
#include<dnn.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_openbt_clicked();
    void video_update();
    void time_update();

private:
    Ui::MainWindow *ui;
    cv::Mat video_image;
    cv::VideoCapture cap;
    QTimer *label_update;
    int flag=0;
    cv::CascadeClassifier classifier;

};
#endif // MAINWINDOW_H
