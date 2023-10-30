#include<iostream>
#include<cuda_runtime.h>
#include "math.h"
#include "arellano.h"
#include "aux_host.h"
#include "mex.h"
#include "constants.h"


__global__ void d_guess_vd_vr_q(double* d_V_r_0, double* d_V_d_0, double* d_Q_0){    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int y_id = blockIdx.x;
    if (thread_id < d_b_grid_size * d_y_grid_size)
    {
        d_V_r_0[thread_id] = -20;
        d_Q_0[thread_id] = 1/(1+d_r);
        if (threadIdx.x == 0)
        {
            d_V_d_0[y_id] = -20;
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

__global__ void update_vd(double* d_V_d_0, double* d_V_d_1, double* d_V, double* d_p_grid, double* d_y_grid_under_default){
    int y_id = threadIdx.x;
    double E_v = 0;
    double E_vd = 0;
    for (int y_prime = 0; y_prime < d_y_grid_size; y_prime++)
    {
        E_v += d_p_grid[y_id * d_y_grid_size + y_prime] * d_V[y_prime * d_b_grid_size + (d_b_grid_size - 1)];
        E_vd += d_p_grid[y_id * d_y_grid_size + y_prime] * d_V_d_0[y_prime];
    }
    d_V_d_1[y_id] = (1/(1-d_gamma)) * pow(d_y_grid_under_default[y_id], 1-d_gamma) + d_beta * (d_theta * E_v + (1-d_theta) * E_vd);
}

__global__ void update_vr_and_bond_policy(double* d_b_grid, double* d_Q_1, double* d_y_grid, double* d_p_grid, double* d_V_r_1, double* d_V, int* d_bond_policy){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    double aux_v = -100000000; 
    if (thread_id < d_b_grid_size * d_y_grid_size)
    {
        for (int x = 0; x<d_b_grid_size; x++)
        {
            double E_V_rx = 0;
            double c = d_y_grid[blockIdx.x] - d_Q_1[blockIdx.x * d_b_grid_size + x] * d_b_grid[x] + d_b_grid[threadIdx.x];
            if (c >= 0)
            {
                for (int y_prime = 0; y_prime< d_y_grid_size; y_prime++)
                {
                    E_V_rx += d_p_grid[blockIdx.x * d_y_grid_size + y_prime] * d_V[y_prime * d_b_grid_size + x];
                }
            }
            double temp = (1/(1-d_gamma)) * pow(c, 1-d_gamma) + d_beta * E_V_rx;
            if (temp >= aux_v)
            {
                aux_v = temp;
                d_bond_policy[thread_id] = x;
            }
            d_V_r_1[thread_id] = aux_v;
        }
    }
    
}

void solve_arellano_model(Parameters_host parms, double* d_b_grid, double* d_y_grid, double* d_p_grid, double* d_y_grid_under_default, double* d_V, double* d_V_d_0, double* d_V_d_1, double* d_V_r_0, double* d_V_r_1, double* d_Q_0, double* d_Q_1, int* d_default_policy, int* d_bond_policy){
    mexPrintf("Running GPU code...\n");
    d_guess_vd_vr_q<<<parms.y_grid_size, parms.b_grid_size>>>(d_V_r_0, d_V_d_0, d_Q_0);
    update_v_and_default_policy<<<parms.y_grid_size, parms.b_grid_size>>>(d_V, d_V_d_0, d_V_r_0, d_default_policy);
    update_price<<<parms.y_grid_size, parms.b_grid_size>>>(d_default_policy, d_Q_1, d_p_grid);
    update_vd<<<1, parms.y_grid_size>>>(d_V_d_0, d_V_d_1, d_V, d_p_grid, d_y_grid_under_default);
    update_vr_and_bond_policy<<<parms.y_grid_size, parms.b_grid_size>>>(d_b_grid, d_Q_1, d_y_grid, d_p_grid, d_V_r_1, d_V, d_bond_policy);
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
    cs = cudaMemcpyToSymbol(d_beta, &(parms.beta), sizeof(double));
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
    }
}
