# Creating a custom container
Follow the steps to create a custom container. The exercises could be run inside this container.

```
$ enroot import docker://nvcr.io/nvidia/pytorch:23.06-py3
$ enroot create --name custom_container nvidia+pytorch+23.06-py3.sqsh
$ enroot start custom_container
$ pip install --no-cache-dir lightning
$ pip install --no-cache-dir torchdata portalocker torchvision
$ HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod
$ exit
$ enroot export --output pytorch+lightning+custom.sqsh custom_container
$ enroot start pytorch+lightning+custom.sqsh
```