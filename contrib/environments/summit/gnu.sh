module use /gpfs/alpine/world-shared/csc296/summit/modulefiles

module load git
module load cmake

module rm xl
module load gcc
module load cuda
module load upcxx
module load python

export MHM2_CMAKE_EXTRAS="-DCMAKE_C_COMPILER=$(which mpicc) -DCMAKE_CXX_COMPILER=$(which mpicxx) -DCMAKE_CUDA_COMPILER=$(which nvcc)"
#-DCMAKE_CUDA_ARCHITECTURES=70"
export MHM2_BUILD_THREADS=8
