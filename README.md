# Examples on the LRZ AI Systems
The current repository contains examples that one could run on the LRZ AI Systems. Instructions on how to create a container to run the examples are mentioned below. Once the container is created, the `slurm` job scripts provided could be used to run the job on the systems.

The materials of this repo serve as a basic how-to and will also be used as part of the [trainings](https://www.lrz.de/services/compute/courses/) offered.

# Creating a custom container
Follow the steps to extend a container from the [NVIDIA NGC Catalog Containers](https://catalog.ngc.nvidia.com/containers) which could be used to run all the examples in this repo.
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

# Acknowledgements
The examples are codes found online adapted to the LRZ AI Systems by [Maja Piskac](https://github.com/pimaja2) and [Ajay Navilarekal Rajgopal](https://github.com/ajaynr) (both part of LRZ). Please approach us if we have used one of your works or would like to see one of your works as a part of this repo.
