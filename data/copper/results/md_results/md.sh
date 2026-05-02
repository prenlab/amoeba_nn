#!/bin/bash
export TINKER9=/user/tinker-nn/build/path/
export CUDA_VISIBLE_DEVICES=0 # device number; can use 1 or 2 if there are multiple GPU cards
$TINKER9/dynamic9 start.xyz  -k cu.key 5000000 2 2 4 200 1  N > npt.out &
echo $(hostname) > node.log

#number of steps; time step(fs); timee between saves(ps); ensenmble 1nve 2nvt 3nph 4npt; temp(k); pressure(bar); N for gpu acc

