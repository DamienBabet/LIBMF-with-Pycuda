# LIBMF-with-Pycuda
Ensae-Paristech school project 

We're still working on an implementation of the LIBMF algorithm ( https://www.csie.ntu.edu.tw/~cjlin/libmf/ ) on GPU via pycuda. The LIBMF algorithm parallelizes matrix factorization stochastic descent gradient algorithm for computation on multicores CPU. We aim to test if the algorithm is scalable for a more agressive parallelization on GPU, using CUDA and Python.

INSTALL
Running a first demo example with pycuda was not easy. We worked first on a Mac Book Pro 13 inches from mid-2009, with a NVIDIA GeForce 9400M 256 Mo GPU, and OS X Yosemite 10.10.5. The GPU driver version being 6.5 at best for this GPU, it was only possible to work with Cuda 6.5 on this device. We followed the indications on https://www.quantstart.com/articles/Installing-Nvidia-CUDA-on-Mac-OSX-for-GPU-Based-Parallel-Computing and the official Nvidia guide http://developer.download.nvidia.com/compute/cuda/6_5/rel/docs/CUDA_Getting_Started_Mac.pdf (pdf). The 6.5 version is archived on the legacy Cuda Toolkit page at https://developer.nvidia.com/cuda-toolkit-archive. On the other hand, Pycuda installation documentation is not completely up-to-date and doesn't go further than Mac OS 10.9 (see https://wiki.tiker.net/PyCuda/Installation/Mac ). We used the instructions on https://www.snip2code.com/Snippet/567751/install-pycuda-on-OSX-10-10(Yosemite)-fo (editing the line 75 to read "python test/test_driver.py"), which itself get the siteconf.py from https://gist.github.com/steerapi

The Numba package, part of the Anaconda python distribution, provides another solution to access CUDA via python. It requires however a device with a compute capability of 2 or higher, whereas our GPU has a capability of 1 according to CUDA standards.

We tested the installation with the computation of a Mandelbrot set with the code essai1pycudaMandelbrot.py, and observed a 50x acceleration via Cuda.

The project is still unfinished. At this stage, the algorithm implemented is not yet LIBMF (FPSGD) but rather a simpler version, DSGD, and only for non-negative matrix factorization. We implemented a iterative (not parallelized) version of the algorithm (code in the file libmfiteratif.py) and a demo version with Cuda and fixed imput data (code in the file libmf_cuda_demo.py). The complete version with Cuda and accepting random imput data (or real data) is not yet working.

We also worked on the formatting of real data to use as an imput in our algorithm.
