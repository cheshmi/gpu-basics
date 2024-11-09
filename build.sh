#!/bin/bash



### here where you load modules.
module load cmake
module load gcc/10.3.0   
module load rocm/4.2.0



### this part is needed if you want to pass an argument, not needed for tutorial 1
DIM1=1024

while getopts ":m:n:k:" arg; do

  case "${arg}" in
    m)
      DIM1=$OPTARG
      ;;
    *) echo ""

      exit 0
  esac
done



# Do not change below, it is fixed for everyone
SHAREDDIR=/home/l/lcl_uotce4sp4//ce4sp4starter/


#### Build
#rm -rf build
mkdir build
# shellcheck disable=SC2164
cd build
/home/l/lcl_uotce4sp4//ce4sp4starter/cmake/bin/cmake  -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON  -DCMAKE_BUILD_TYPE=Release  ..
make -j 4

cd ..

BINPATH=./build/


### You are not supposed to run your code here. This is for compiling your code. Run your code with another script (run_...) and sbatch command
