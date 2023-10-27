#include<iostream>
#include<cuda_runtime.h>
#include "arellano.h"
#include "aux_host.h"
#include "mex.h"

__global__ void testor_3(int x, int* d_default_policy, int grid_size){
    int thread_id = threadIdx.x;
    if (thread_id < grid_size)
    {
        d_default_policy[thread_id] = x;
    }
}

void run(Parameters_host parms, int* d_default_policy){
    mexPrintf("Running GPU code...\n");
    mexPrintf("Grid size: %d\n", parms.y_grid_size);
    testor_3<<<1, 100>>>(7, d_default_policy, parms.y_grid_size * parms.b_grid_size);
}
