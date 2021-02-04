module rm upcxx
module rm upcxx-gpu
module rm PrgEnv-intel
module rm PrgEnv-cray
module rm PrgEnv-gnu

module rm craype
module rm craype-mic-knl
module rm craype-haswell
module rm craype-x86-skylake

module load cmem

module load PrgEnv-gnu
module load craype
#module load craype-x86-skylake

module load cmake/3.18.2
module load git

# use custom build of upcxx
# (temporary) --disable-ibv-odp
### upcxx_opts="--with-ibv-physmem-max=0.5 --enable-ibv-multirail --with-ibv-max-hcas=4 --without-cross" ./install_upcxx.sh ibv posix /global/common/software/m2865/upcxx-2020.10.0-ibv-cmem
# upcxx_opts="--enable-ibv-multirail --with-ibv-max-hcas=4 --without-cross --disable-ibv-odp --with-ibv-physmem-max=0.5 --with-default-network=ibv --with-pmi-home=/opt/esslurm --with-ibv-spawner=pmi" with_pmirun_cmd='srun %V -K0 -W60 -n %N %C'  ./install_upcxx.sh ibv posix /global/common/software/m2865/upcxx-2020.10.0-ibv-cmem
export PATH=/global/common/software/m2865/upcxx-2020.10.0-ibv-cmem/bin:$PATH

# use ibv and do not probe for shared heap size
export UPCXX_NETWORK=ibv
export GASNET_PHYSMEM_MAX=64G
export GASNET_PHYSMEM_PROBE=0
export GASNET_TMPDIR=$SCRATCH/tmp
mkdir -p $GASNET_TMPDIR

module list

export MHM2_CMAKE_EXTRAS="-DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_C_COMPILER=$(which gcc) "
echo $MHM2_CMAKE_EXTRAS

