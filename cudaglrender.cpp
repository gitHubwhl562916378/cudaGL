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

   pixVBO_ = new QOpenGLBuffer(QOpenGLBuffer::PixelPackBuffer);
   pixVBO_->setUsagePattern(QOpenGLBuffer::DynamicDraw);
   pixVBO_->create();
   pixVBO_->bind();
   pixVBO_->allocate(image_width * image_height * 4 * sizeof(GLubyte));
   cudaError res;//GL_RGBA8UI_EXT GL_RGBA_INTEGER_EXT
   res = cudaGraphicsGLRegisterBuffer(&cuda_tex_result_resource,pixVBO_->bufferId(),cudaGraphicsMapFlagsNone);
   if(res != cudaSuccess){
       qDebug() << __FILE__ << __LINE__ << "cudaGraphicsGLRegisterBuffer:" << res;
   }
   res = cudaMalloc((void**)&cuda_dest_resource,image_width * image_height * 4 * sizeof(GLubyte));
   if(res != cudaSuccess){
       qDebug() << __FILE__ << __LINE__ << "cudaMalloc:" << res;
   }

   texture_ = new QOpenGLTexture(QOpenGLTexture::Target2D);
   texture_->setSize(image_width,image_height);
   texture_->setFormat(QOpenGLTexture::RGBA8_UNorm);
   texture_->setWrapMode(QOpenGLTexture::ClampToEdge);
   texture_->setMinMagFilters(QOpenGLTexture::Linear,QOpenGLTexture::Linear);
   texture_->allocateStorage();

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
    program_.setUniformValue("uPMatrix",pMatrix);
    program_.setUniformValue("uVMatrix",vMatrix);
    program_.setUniformValue("uMMatrix",mMatrix);
    program_.setUniformValue("sTexture",0);

    cudaError res;
    dim3 block(16,16,1);
    dim3 grid(image_width / block.x,image_height / block.y,1);
    launch_cudaProcess(grid, block, 0, cuda_dest_resource, image_width);
    uchar4  *texture_ptr;
    size_t num_bytes;
    res = cudaGraphicsMapResources(1,&cuda_tex_result_resource,0);
    if(res != cudaSuccess){
        qDebug() << __FILE__ << __LINE__ << "cudaGraphicsMapResources:" << res;
    }
    res = cudaGraphicsResourceGetMappedPointer((void**)&texture_ptr,&num_bytes,cuda_tex_result_resource);
    if(res != cudaSuccess){
        qDebug() << __FILE__ << __LINE__ << "cudaGraphicsSubResourceGetMappedArray:" << res;
    }
    res = cudaMemcpy((void**)texture_ptr, (void**)cuda_dest_resource,image_width * image_height * 4 * sizeof(GLubyte), cudaMemcpyDeviceToDevice);
    res = cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0);
    if(res != cudaSuccess){
        qDebug() << __FILE__ << __LINE__ << "cudaGraphicsUnmapResources:" << res;
    }
#if 0 //从显存中的pixVBO_ 加载到显存中的纹理
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pixVBO_->bufferId());
    f->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height,GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    f->glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else //从显存中的pixVBO_先映射到cpu应用空间，先读到cpu再写到显存中的纹理
    texture_->setData(QOpenGLTexture::RGBA,QOpenGLTexture::UInt8,pixVBO_->map(QOpenGLBuffer::ReadOnly));
#endif

    program_.enableAttributeArray(0);
    program_.enableAttributeArray(1);
    vbo_.bind();
    program_.setAttributeBuffer(0,GL_FLOAT,0,3,3*sizeof(GLfloat));
    program_.setAttributeBuffer(1,GL_FLOAT,3 * 4 * sizeof(GLfloat),2,2*sizeof(GLfloat));
    vbo_.release();
    texture_->bind();
    f->glDrawArrays(GL_TRIANGLE_FAN,0,4);
    program_.disableAttributeArray(0);
    program_.disableAttributeArray(1);
    program_.release();
    f->glDisable(GL_CULL_FACE);
}
