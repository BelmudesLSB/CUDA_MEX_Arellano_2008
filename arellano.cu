#include<iostream>
#include<cuda_runtime.h>
#include "math.h"
#include "arellano.h"
#include "aux_host.h"
#include "mex.h"
#include "constants.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>


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
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int y_id = blockIdx.x; 
    if (thread_id < d_b_grid_size * d_y_grid_size)
    {   double VD_aux = d_V_d_0[y_id];
        double VR_aux = d_V_r_0[thread_id];
        if (VR_aux >= VD_aux)
        {
            d_V[thread_id] = VR_aux;
            d_default_policy[thread_id] = 0;
        }
        else
        {
            d_V[thread_id] = VD_aux;
            d_default_policy[thread_id] = 1;
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
            aux += d_p_grid[y_id * d_y_grid_size + y_prime] * (1-d_default_policy[y_prime * d_b_grid_size + b_id]) * (1/(1+d_r));
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
    int y_id = blockIdx.x;
    int b_id = threadIdx.x;
    if (thread_id < d_b_grid_size * d_y_grid_size)
    {
        double aux_v = -100000; 
        for (int x = 0; x<d_b_grid_size; x++)
        {
            double E_V_rx = 0;
            double c = d_y_grid[y_id] - d_Q_1[y_id * d_b_grid_size + x] * d_b_grid[x] + d_b_grid[b_id];
            if (c > 0)
            {
                for (int y_prime = 0; y_prime< d_y_grid_size; y_prime++)
                {
                    E_V_rx += d_p_grid[y_id * d_y_grid_size + y_prime] * d_V[y_prime * d_b_grid_size + x];
                }
                double temp = (1/(1-d_gamma)) * pow(c, 1-d_gamma) + d_beta * E_V_rx;
                if (temp >= aux_v)
                {
                    aux_v = temp;
                    d_bond_policy[thread_id] = x;
                }
            }
        }
        d_V_r_1[thread_id] = aux_v;
    }
}

__global__ void compute_distance(double* d_F, double* d_G, double* d_Err){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < d_b_grid_size * d_y_grid_size)
    {
        d_Err[thread_id] = fabs(d_F[thread_id] - d_G[thread_id]);
    }
}

void solve_arellano_model(Parameters_host parms, double* d_b_grid, double* d_y_grid, double* d_p_grid, double* d_y_grid_under_default, double* d_V, double* d_V_d_0, double* d_V_d_1, double* d_V_r_0, double* d_V_r_1, double* d_Q_0, double* d_Q_1, int* d_default_policy, int* d_bond_policy, double* d_Err_q, double* d_Err_vr, double* d_Err_vd){
    mexPrintf("Running GPU code...\n");
    double error_q = 1;
    double error_vr = 1;
    double error_vd = 1;
    int iter = 0;
    d_guess_vd_vr_q<<<parms.y_grid_size, parms.b_grid_size>>>(d_V_r_0, d_V_d_0, d_Q_0);
    while ((error_q > parms.tol || error_vr > parms.tol || error_vd > parms.tol) && iter < parms.max_iter)
    {
        update_v_and_default_policy<<<parms.y_grid_size, parms.b_grid_size>>>(d_V, d_V_d_0, d_V_r_0, d_default_policy);
        update_price<<<parms.y_grid_size, parms.b_grid_size>>>(d_default_policy, d_Q_1, d_p_grid);
        update_vd<<<1, parms.y_grid_size>>>(d_V_d_0, d_V_d_1, d_V, d_p_grid, d_y_grid_under_default);
        update_vr_and_bond_policy<<<parms.y_grid_size, parms.b_grid_size>>>(d_b_grid, d_Q_1, d_y_grid, d_p_grid, d_V_r_1, d_V, d_bond_policy);
        compute_distance<<<parms.y_grid_size, parms.b_grid_size>>>(d_V_r_1, d_V_r_0, d_Err_vr);
        compute_distance<<<1, parms.y_grid_size>>>(d_V_d_1, d_V_d_0, d_Err_vd);
        compute_distance<<<parms.y_grid_size, parms.b_grid_size>>>(d_Q_1, d_Q_0, d_Err_q);
        thrust::device_ptr<double> d_Err_q_ptr = thrust::device_pointer_cast(d_Err_q);
        thrust::device_ptr<double> d_Err_vr_ptr = thrust::device_pointer_cast(d_Err_vr);
        thrust::device_ptr<double> d_Err_vd_ptr = thrust::device_pointer_cast(d_Err_vd);
        thrust::device_vector<double>::iterator iter_q = thrust::max_element(d_Err_q_ptr, d_Err_q_ptr + parms.b_grid_size * parms.y_grid_size);
        thrust::device_vector<double>::iterator iter_vr = thrust::max_element(d_Err_vr_ptr, d_Err_vr_ptr + parms.b_grid_size * parms.y_grid_size);
        thrust::device_vector<double>::iterator iter_vd = thrust::max_element(d_Err_vd_ptr, d_Err_vd_ptr + parms.y_grid_size);
        error_q = *iter_q;
        error_vr = *iter_vr;
        error_vd = *iter_vd;
        iter += 1;
        cudaMemcpy(d_V_d_0, d_V_d_1, parms.y_grid_size * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_V_r_0, d_V_r_1, parms.b_grid_size * parms.y_grid_size * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_Q_0, d_Q_1, parms.b_grid_size * parms.y_grid_size * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    if ((error_q < parms.tol && error_vr < parms.tol && error_vd < parms.tol))
    {
        mexPrintf("Convergence achieved in %d iterations\n", iter);
        mexPrintf("Error in bond price: %f\n", error_q);
        mexPrintf("Error in value function under repayment: %f\n", error_vr);
        mexPrintf("Error in value function under default: %f\n", error_vd);
    }
    else
    {
        mexPrintf("No convergence achieved in %d iterations\n", iter);
    }
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
