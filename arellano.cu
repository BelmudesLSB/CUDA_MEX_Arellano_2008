#include<iostream>
#include<cuda_runtime.h>
#include "arellano.h"
#include "aux_host.h"
#include "mex.h"
#include "constants.h"


__global__ void d_guess_vd_vr_q(double* d_V_r_0, double* d_V_d_0, double* d_Q_0){    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < d_b_grid_size * d_y_grid_size)
    {
        d_V_r_0[id] = -20;
        d_Q_0[id] = 1/(1+d_r);
        if (id < d_y_grid_size)
        {
            d_V_d_0[id] = -20;
        }
    }
}

__global__ void update_v_and_default_policy(double* d_V, double* d_V_d_0, double* d_V_r_0, int* d_default_policy){

    int id = blockIdx.x * blockDim.x + threadIdx.x; 
    if (id < d_b_grid_size * d_y_grid_size)
    {   double VD_aux = d_V_d_0[blockIdx.x];
        double VR_aux = d_V_r_0[id];
        if (VR_aux >= VD_aux)
        {
            d_V[id] = VR_aux;
            d_default_policy[id] = 0;
        }
        else
        {
            d_V[id] = VD_aux;
            d_default_policy[id] = 1;
        }
    }

}

__global__ void update_price(int* d_default_policy, double* d_Q_1, double* d_p_grid){
    int b_id = threadIdx.x;
    int y_id = blockIdx.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < d_b_grid_size * d_y_grid_size)
    {
        double aux = 0;
        for (int y_prime = 0; y_prime < d_y_grid_size; y_prime++)
        {
            aux += d_p_grid[y_id * d_y_grid_size + y_prime] * (1-d_default_policy[y_prime * blockDim.x + b_id]) * (1/(1+d_r));
        }
        d_Q_1[thread_id] = aux;
    }
}

//__globbal__ void update_vd(double* )

void solve_arellano_model(Parameters_host parms, double* d_b_grid, double* d_y_grid, double* d_p_grid, double* d_y_grid_under_default, double* d_V, double* d_V_d_0, double* d_V_d_1, double* d_V_r_0, double* d_V_r_1, double* d_Q_0, double* d_Q_1, int* d_default_policy, int* d_bond_policy){
    mexPrintf("Running GPU code...\n");
    d_guess_vd_vr_q<<<parms.y_grid_size, parms.b_grid_size>>>(d_V_r_0, d_V_d_0, d_Q_0);
    update_v_and_default_policy<<<parms.y_grid_size, parms.b_grid_size>>>(d_V, d_V_d_0, d_V_r_0, d_default_policy);
    update_price<<<parms.y_grid_size, parms.b_grid_size>>>(d_default_policy, d_Q_1, d_p_grid);
}

void fill_device_constants(Parameters_host parms){
    cudaError_t cs;
    cs = cudaMemcpyToSymbol(d_b_grid_size, &(parms.b_grid_size), sizeof(int));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
    cs = cudaMemcpyToSymbol(d_y_grid_size, &(parms.y_grid_size), sizeof(int));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
    cs = cudaMemcpyToSymbol(d_b_grid_max, &(parms.b_grid_max), sizeof(double));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
    cs = cudaMemcpyToSymbol(d_b_grid_min, &(parms.b_grid_min), sizeof(double));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
    cs = cudaMemcpyToSymbol(d_gamma, &(parms.gamma), sizeof(double));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
    cs = cudaMemcpyToSymbol(d_r, &(parms.r), sizeof(double));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
    cs = cudaMemcpyToSymbol(d_sigma, &(parms.sigma), sizeof(double));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
    cs = cudaMemcpyToSymbol(d_tol, &(parms.tol), sizeof(double));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
    cs = cudaMemcpyToSymbol(d_iter, &(parms.max_iter), sizeof(int));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
    cs = cudaMemcpyToSymbol(d_theta, &(parms.theta), sizeof(double));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
}
