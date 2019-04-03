#include <QDebug>
#include <cuda_runtime.h>
#include "cudaglrender.h"
extern "C" bool gpuInit();
extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                   unsigned int *g_odata,
                   int imgw);
CudaGLRender::CudaGLRender()
{
}

void CudaGLRender::initsize(QOpenGLExtraFunctions *f)
{
   program_.addCacheableShaderFromSourceFile(QOpenGLShader::Vertex,"vertex.vsh");
   program_.addCacheableShaderFromSourceFile(QOpenGLShader::Fragment,"fragment.fsh");
   program_.link();

   bool isOk =  gpuInit();
   if(!isOk)return;

   texture_ = new QOpenGLTexture(QOpenGLTexture::Target2D);
   texture_->create();
   texture_->setSize(image_width,image_height);
   texture_->setFormat(QOpenGLTexture::RGBA8_UNorm);
   texture_->allocateStorage();
   texture_->setWrapMode(QOpenGLTexture::ClampToEdge);
   texture_->setMinMagFilters(QOpenGLTexture::NearestMipMapLinear,QOpenGLTexture::NearestMipMapLinear);
   cudaError res;//GL_RGBA8UI_EXT GL_RGBA_INTEGER_EXT
   res = cudaGraphicsGLRegisterImage(&cuda_tex_result_resource,texture_->textureId(),GL_TEXTURE_2D,cudaGraphicsMapFlagsWriteDiscard);
   if(res != cudaSuccess){
       qDebug() << __FILE__ << __LINE__ << "cudaGraphicsGLRegisterImage:" << res;
   }
   res = cudaMalloc((void**)&cuda_dest_resource,image_width * image_height * 4 * sizeof(GLubyte));
   if(res != cudaSuccess){
       qDebug() << __FILE__ << __LINE__ << "cudaMalloc:" << res;
   }

   GLfloat points[]{
       -1,1,0,
       -1,-1,0,
       1,-1,0,
       1,1,0,

       0,0,
       0,1,
       1,1,
       1,0
   };
   vbo_.create();
   vbo_.bind();
   vbo_.allocate(points,sizeof(points));
}

void CudaGLRender::render(QOpenGLExtraFunctions *f, QMatrix4x4 pMatrix, QMatrix4x4 vMatrix, QMatrix4x4 mMatrix)
{
    f->glDisable(GL_DEPTH_TEST);
    f->glEnable(GL_CULL_FACE);
    program_.bind();
    vbo_.bind();
    program_.setUniformValue("uPMatrix",pMatrix);
    program_.setUniformValue("uVMatrix",vMatrix);
    program_.setUniformValue("uMMatrix",mMatrix);
    program_.setUniformValue("sTexture",0);

    cudaError res;
    dim3 block(16,16,1);
    dim3 grid(image_width / block.x,image_height / block.y,1);
    launch_cudaProcess(grid, block, 0, cuda_dest_resource, image_width);
    cudaArray *texture_ptr;
    res = cudaGraphicsMapResources(1,&cuda_tex_result_resource,0);
    if(res != cudaSuccess){
        qDebug() << __FILE__ << __LINE__ << "cudaGraphicsMapResources:" << res;
    }
    res = cudaGraphicsSubResourceGetMappedArray(&texture_ptr,cuda_tex_result_resource,0,0);
    if(res != cudaSuccess){
        qDebug() << __FILE__ << __LINE__ << "cudaGraphicsSubResourceGetMappedArray:" << res;
    }
    res = cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, image_width * image_height * 4 * sizeof(GLubyte), cudaMemcpyDeviceToDevice);
    res = cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0);
    if(res != cudaSuccess){
        qDebug() << __FILE__ << __LINE__ << "cudaGraphicsUnmapResources:" << res;
    }
//    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
//    cudaDeviceSynchronize();

    program_.enableAttributeArray(0);
    program_.enableAttributeArray(1);
    program_.setAttributeBuffer(0,GL_FLOAT,0,3,3*sizeof(GLfloat));
    program_.setAttributeBuffer(1,GL_FLOAT,3 * 4 * sizeof(GLfloat),2,2*sizeof(GLfloat));
    texture_->bind();
    f->glDrawArrays(GL_TRIANGLE_FAN,0,4);
    program_.disableAttributeArray(0);
    program_.disableAttributeArray(1);
    vbo_.release();
    program_.release();
    f->glDisable(GL_CULL_FACE);
}
