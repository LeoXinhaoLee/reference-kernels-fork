#!/bin/bash


# export CUSTOM_KERNEL_MODULE="kernels/1.py"
# python eval_local.py results/MI300x/1/test1.out

export CUSTOM_KERNEL_MODULE="kernels/5.py"
python eval_local.py results/MI300x/5/test1.out

# export CUSTOM_KERNEL_MODULE="kernels/4.py"
# python eval_local.py results/MI300x/4/test1.out

# for i in {1..3}; do

#     for j in {1..3}; do

#     export CUSTOM_KERNEL_MODULE="kernels/${i}.py"
#     python eval_local.py results/MI300x/${i}/test${j}.out

#     done

# done
