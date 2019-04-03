#ifndef WIDGET_H
#define WIDGET_H

#include <QOpenGLWidget>
#include "cudaglrender.h"
class Widget : public QOpenGLWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = 0);
    ~Widget();

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

private:
    QTimer *tm_{nullptr};
    CudaGLRender render_;
    QMatrix4x4 pMatrix_;
    QVector3D camera_;

private slots:
    void slotTimeout();
};

#endif // WIDGET_H
