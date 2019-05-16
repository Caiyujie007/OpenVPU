.PHONY:clean all

LIB_SRC= ./lib/libVPUAPI.a

Compile_CMD= 	-O3

all:Malloc CONV POOL ELEMENTWISE NONLINEAR MNIST_LARGE VGG16

clean:
	@rm -f malloc_test conv_test pool_test elementwise_test nonlinear_test mnist_large vgg16

Malloc:
	g++ malloc_test.cpp ${LIB_SRC} -I ./inc ${Compile_CMD} -o malloc_test

CONV:
	g++ conv_test.c ${LIB_SRC} -I ./inc ${Compile_CMD} -o conv_test

POOL:
	g++ pool_test.c ${LIB_SRC} -I ./inc ${Compile_CMD} -o pool_test

ELEMENTWISE:
	g++ elementwise_test.c ${LIB_SRC} -I ./inc ${Compile_CMD} -o elementwise_test

NONLINEAR:
	g++ nonlinear_test.c ${LIB_SRC} -I ./inc ${Compile_CMD} -o nonlinear_test

MNIST_LARGE:
	g++ mnist_large.c ${LIB_SRC} -I ./inc -I ./MNIST_LARGE_DAT ${Compile_CMD} -o mnist_large

VGG16:
	g++ vgg16.c ${LIB_SRC} -I ./inc -I ./VGG16_DAT ${Compile_CMD} -o vgg16

