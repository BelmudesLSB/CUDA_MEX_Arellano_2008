#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include "solver.h"               
#include "../host/helpers.h"      
#include "mex.h"
#include "device_constants.h"     


/*
  ================================================================
   GPU SOLVER FOR THE ARELLANO (2008) SOVEREIGN DEFAULT MODEL
   ------------------------------------------------------------
   Implements all CUDA kernels and the main solver loop.

   The model solves for equilibrium bond prices (Q), value functions
   under repayment (V_r) and default (V_d), and policy functions
   (default decision and next-period debt).

   The iteration proceeds as:
     1. Initialize guesses for Q, V_r, and V_d.
     2. Iterate until convergence:
          - Update total value function and default policy.
          - Update bond price schedule.
          - Update default-state value.
          - Update repayment-state value and bond policy.
          - Compute maximum absolute differences.
     3. Stop when errors fall below tolerance or max_iter reached.

   Lucas Belmudes — 10/31/2023
  ================================================================
*/


// -----------------------------------------------------------------------------
// 1) INITIAL GUESSES
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// 2) VALUE FUNCTION AND DEFAULT DECISION
// -----------------------------------------------------------------------------
/*
   Compares the value of repayment (V_r) and default (V_d).
   For each (y,b):
     - If V_r ≥ V_d → repay (default_policy = 0)
     - Else          → default (default_policy = 1)
   Stores the optimal value V = max(V_r, V_d).
*/
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

// -----------------------------------------------------------------------------
// 3) PRICE FUNCTION UPDATE
// -----------------------------------------------------------------------------
/*
   Updates Q(b,y) according to the zero-profit condition for lenders:
     Q(b,y) = E[ (1 − d_default_policy(y′,b)) / (1 + r) ]
   i.e., the price equals the discounted probability of repayment.
*/
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

// -----------------------------------------------------------------------------
// 4) DEFAULT-STATE VALUE FUNCTION
// -----------------------------------------------------------------------------
/*
   Updates V_d(y):
     V_d(y) = u(y_d) + β [ θ E[V(y′,b′=Bmax)] + (1−θ) E[V_d(y′)] ]
   where:
     - u(y_d) is utility under default income (fixed at y_def or capped)
     - E[·] are expectations over next-period income states.
*/
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

// -----------------------------------------------------------------------------
// 5) REPAYMENT-STATE VALUE FUNCTION AND BOND POLICY
// -----------------------------------------------------------------------------
/*
   For each (b,y), searches over all possible next-period debt levels b′ to
   maximize:
       V_r(b,y) = max_{b′} [ u(c) + β E[V(b′,y′)] ],
   subject to c = y − Q(b′,y)·b′ + b  and c > 0.
   Stores:
     - Optimal value V_r_1(b,y)
     - Optimal bond index (policy) b′*(b,y)
*/
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

// -----------------------------------------------------------------------------
// 6) DISTANCE COMPUTATION
// -----------------------------------------------------------------------------
/*
   Computes element-wise absolute difference |F − G| to monitor convergence.
   Used for value functions and bond prices.
*/
__global__ void compute_distance(double* d_F, double* d_G, double* d_Err){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < d_b_grid_size * d_y_grid_size)
    {
        d_Err[thread_id] = fabs(d_F[thread_id] - d_G[thread_id]);
    }
}

// -----------------------------------------------------------------------------
// 7) HOST-SIDE SOLVER LOOP
// -----------------------------------------------------------------------------
/*
   Main fixed-point iteration controlling all kernels.

   Iteration steps:
     1. Initialize guesses (V_r_0, V_d_0, Q_0).
     2. Iterate:
          a) update V and default policy
          b) update Q
          c) update V_d
          d) update V_r and bond policy
          e) compute distances and check convergence
     3. Stop when all errors < tol or max_iter reached.
*/
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

// -----------------------------------------------------------------------------
// 8) CONSTANT MEMORY INITIALIZATION
// -----------------------------------------------------------------------------
/*
   Copies scalar parameters from host struct to GPU constant memory.
   This allows kernels to read them quickly without passing as arguments.
*/
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
