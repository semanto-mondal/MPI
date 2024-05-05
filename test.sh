#!/bin/bash

export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_GPU_DIRECT_RDMA=yes
export UCX_TLS=ib

if [ $# -ne 1 ]
then
        echo "Usage: $0 <NPROCS>"
        exit 1
fi

echo "Compile program"
make
echo

echo "Execute program"
srun -N $1 -p parallel --reservation=maintenance ./cpi
echo


echo "Clear temporay file"
make clean
echo