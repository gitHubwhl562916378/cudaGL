#ifndef CUDAGLRENDER_H
#define CUDAGLRENDER_H

#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLExtraFunctions>
#include <cuda_gl_interop.h>
class CudaGLRender
{
public:
    CudaGLRender();
    void initsize(QOpenGLExtraFunctions *f);
    void render(QOpenGLExtraFunctions *f, QMatrix4x4 pMatrix,QMatrix4x4 vMatrix,QMatrix4x4 mMatrix);

protected:


private:
    QOpenGLShaderProgram program_;
    QOpenGLBuffer vbo_,*pixVBO_{nullptr};
    int image_width = 512,image_height = 512;
    QOpenGLTexture *texture_{nullptr};
    unsigned int *cuda_dest_resource{nullptr};
    cudaGraphicsResource *cuda_tex_result_resource{nullptr};
};

#endif // CUDAGLRENDER_H
