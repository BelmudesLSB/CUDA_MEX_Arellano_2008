#ifndef arellano_h
#define arellano_h

#include "aux_host.h"

__global__ void testor_3(int x, int* d_default_policy, int grid_size);

void run(Parameters_host parms, int* d_default_policy);

#endif