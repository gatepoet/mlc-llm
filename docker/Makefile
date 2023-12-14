EXEC=docker buildx build ../ --load
CUDA_VERSION=12.1.1
CUDA_VERSION_SHORT=121
PYTHON_VERSION=3.11
SHARED_MEMORY=16G
GPUS='"device=1,2,3,4,5,6,7,8"'
DIST_FOLDER=$(shell pwd)/../dist
MODELS_FOLDER=/models
# MODELS_FOLDER=$(shell pwd)/../dist/models
TAG=local/mlc
MAMBA_USER=$(shell whoami)
MAMBA_USER_ID=$(shell id -u)
MAMBA_GROUP_ID=$(shell id -g)

all: cu121 cu118

cu121:
	make cuda CUDA_VERSION=12.1.1 CUDA_VERSION_SHORT=121 PYTHON_VERSION=3.11

cu121-build:
	make cuda-build CUDA_VERSION=12.1.1 CUDA_VERSION_SHORT=121 PYTHON_VERSION=3.11

cu118:
	make cuda CUDA_VERSION=11.8.0 CUDA_VERSION_SHORT=118 PYTHON_VERSION=3.10

cuda: 
	 $(EXEC) -f Dockerfile.cuda \
	 		-t $(TAG) \
	 		-t $(TAG):cuda \
	 		-t $(TAG):cu$(CUDA_VERSION_SHORT) \
			--build-arg CUDA_VERSION=$(CUDA_VERSION) \
			--build-arg CUDA_VERSION_SHORT=$(CUDA_VERSION_SHORT) \
			--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
			--build-arg MAMBA_USER=$(MAMBA_USER) \
			--build-arg MAMBA_USER_ID=$(MAMBA_USER_ID) \
			--build-arg MAMBA_GROUP_ID=$(MAMBA_GROUP_ID)
cuda-build:
	 $(EXEC) -f Dockerfile.cuda-build \
	 		-t $(TAG) \
	 		-t $(TAG):cuda \
	 		-t $(TAG):cu$(CUDA_VERSION_SHORT) \
			--build-arg CUDA_VERSION=$(CUDA_VERSION) \
			--build-arg CUDA_VERSION_SHORT=$(CUDA_VERSION_SHORT) \
			--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
			--build-arg MAMBA_USER=$(MAMBA_USER) \
			--build-arg MAMBA_USER_ID=$(MAMBA_USER_ID) \
			--build-arg MAMBA_GROUP_ID=$(MAMBA_GROUP_ID) \
			--progress plain


run:
	docker run --rm -it \
           --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	   --shm-size=$(SHARED_MEMORY) --gpus=$(GPUS) \
           --volume $(DIST_FOLDER):/home/$(MAMBA_USER)/mlc-llm/dist \
           --volume $(MODELS_FOLDER):/home/$(MAMBA_USER)/mlc-llm/dist/models \
           --name mlc -p 8080:8000 -p 1234:8000 \
           $(TAG)

