CUDA_SOURCE = $$PWD/process.cu
NVCC_OPTIONS = --use-local-env --cl-version 2015 -gencode=arch=compute_35,code=sm_35 \
                               -gencode=arch=compute_37,code=sm_37 \
                               -gencode=arch=compute_50,code=sm_50 \
                               -gencode=arch=compute_52,code=sm_52 \
                               -gencode=arch=compute_60,code=sm_60

win32{
INCLUDEPATH += $$(CUDA_PATH)\include
contains(QT_ARCH,i386){
    QMAKE_LIBDIR += $$(CUDA_PATH)\lib\Win32
    CUDA_LIBS = cuda.lib cudart.lib
    LIBS += $$CUDA_LIBS
    CONFIG(debug, debug | release){
        NVCC_OPTIONS += -Xcompiler \"/EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MDd\"
    }else{
        NVCC_OPTIONS += -Xcompiler \"/EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MD\"
    }

    cuda_d.input = CUDA_SOURCE
    cuda_d.output = cuda/${QMAKE_FILE_BASE}.o
    cuda_d.commands = $$(CUDA_PATH)\bin\nvcc $$join(INCLUDEPATH,'" -I"','-I"','"') --machine 32 -Xcompiler $$NVCC_OPTIONS \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}else{
    QMAKE_LIBDIR += $$(CUDA_PATH)\lib\x64
    CUDA_LIBS = cuda.lib cudart.lib
    LIBS += $$CUDA_LIBS
    CONFIG(debug, debug | release){
        NVCC_OPTIONS += -Xcompiler \"/EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MDd\"
    }else{
        NVCC_OPTIONS += -Xcompiler \"/EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MD\"
    }

    cuda_d.input = CUDA_SOURCE
    cuda_d.output = cuda/${QMAKE_FILE_BASE}.o
    cuda_d.commands = $$(CUDA_PATH)\bin\nvcc $$join(INCLUDEPATH,'" -I"','-I"','"') --machine 64 -Xcompiler $$NVCC_OPTIONS \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
}
