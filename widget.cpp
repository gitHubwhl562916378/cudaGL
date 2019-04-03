#include <QTimer>
#include "widget.h"

Widget::Widget(QWidget *parent)
    : QOpenGLWidget(parent)
{
    tm_ = new QTimer(this);
    connect(tm_,SIGNAL(timeout()),this,SLOT(slotTimeout()));
    tm_->start(30);
}

Widget::~Widget()
{

}

void Widget::initializeGL()
{
    camera_ = QVector3D(0,0,3);
    render_.initsize(QOpenGLContext::currentContext()->extraFunctions());
}

void Widget::resizeGL(int w, int h)
{
    pMatrix_.setToIdentity();
    pMatrix_.perspective(45,float(w)/h,0.01f,100.0f);
}

void Widget::paintGL()
{
    QOpenGLExtraFunctions *f = QOpenGLContext::currentContext()->extraFunctions();
    f->glClearColor(0.0,0.0,0.0,1.0);
    f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    QMatrix4x4 vMatrix;
    vMatrix.lookAt(camera_,QVector3D(0,0,0),QVector3D(0,1,0));

    QMatrix4x4 mMatrix;
    render_.render(f,pMatrix_,vMatrix,mMatrix);
}

void Widget::slotTimeout()
{
    update();
}
