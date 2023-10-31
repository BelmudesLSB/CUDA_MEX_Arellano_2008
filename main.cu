#include <iostream>
#include <cuda_runtime.h>
#include "constants.h"
#include "mex.h"
#include "aux_host.h"
#include "arellano.h"

/*
* This code is an implementation of Arellano (2008) using MEX and CUDA.
* Lucas Belmudes, 10/31/2023.
*/

//! By default all variables are in the host. Else, they will have a d_ prefix.

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){
    
    // ! Load and unload the parameters from matlab:
    // Read the input parameters from MATLAB
    const mxArray* parmsStruct = prhs[0];
    // Create an instance of the parameters class
    Parameters_host parms;
    // Load the parameters from MATLAB into the instance of the class:
    parms.b_grid_size = static_cast<int>(mxGetScalar(mxGetField(parmsStruct, 0, "b_grid_size")));
    parms.b_grid_min = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "b_grid_min")));
    parms.b_grid_max = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "b_grid_max")));
    parms.y_grid_size = static_cast<int>(mxGetScalar(mxGetField(parmsStruct, 0, "y_grid_size")));
    parms.y_default = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "y_default")));
    parms.beta = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "beta")));
    parms.gamma = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "gamma")));
    parms.r = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "r")));
    parms.rho = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "rho")));
    parms.sigma = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "sigma")));
    parms.theta = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "theta")));
    parms.tol = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "tol")));
    parms.max_iter = static_cast<int>(mxGetScalar(mxGetField(parmsStruct, 0, "max_iter")));
    parms.m = static_cast<double>(mxGetScalar(mxGetField(parmsStruct, 0, "m")));

    // ! Create the grids:
    // Construct the grids in the host:
    double* b_grid = new double[parms.b_grid_size];
    double* y_grid = new double[parms.y_grid_size];
    double* p_grid = new double[parms.y_grid_size*parms.y_grid_size];
    double* y_grid_under_default = new double[parms.y_grid_size];
    double* V = new double[parms.b_grid_size*parms.y_grid_size];
    double* V_d = new double[parms.y_grid_size];
    double* V_r = new double[parms.b_grid_size*parms.y_grid_size];
    double* Q = new double[parms.b_grid_size*parms.y_grid_size];
    int* default_policy = new int[parms.b_grid_size*parms.y_grid_size];
    int* bond_policy = new int[parms.b_grid_size*parms.y_grid_size];
    // Create all host grids:
    create_bond_grids(b_grid, parms.b_grid_size, parms.b_grid_max, parms.b_grid_min);
    create_income_and_prob_grids(y_grid, p_grid, parms.y_grid_size, parms.sigma, parms.rho, parms.m);
    create_income_under_default(y_grid_under_default, y_grid, parms.y_grid_size, parms.y_default);


    // ! Create the grids in the device:
    // Create device pointer:
    double* d_b_grid;
    double* d_y_grid;
    double* d_p_grid;
    double* d_y_grid_under_default;
    double* d_V;
    double* d_V_d_0;
    double* d_V_d_1;
    double* d_V_r_0;
    double* d_V_r_1;
    double* d_Q_0;
    double* d_Q_1;
    int* d_default_policy;
    int* d_bond_policy;
    double* d_Err_q;
    double* d_Err_vr;
    double* d_Err_vd;
 
    // Allocate memory in the device:
    cudaError_t cs;
    cs = cudaMalloc((void**)&d_b_grid, parms.b_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for b_grid: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_y_grid, parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for y_grid: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_p_grid, parms.y_grid_size*parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for p_grid: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_y_grid_under_default, parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for y_grid_under_default: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_V, parms.b_grid_size*parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for V: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_V_d_0, parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for V_d: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_V_d_1, parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for V_d: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_V_r_0, parms.b_grid_size*parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for V_r: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_V_r_1, parms.b_grid_size*parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for V_r: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_Q_0, parms.b_grid_size*parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for Q: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_Q_1, parms.b_grid_size*parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for Q: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_default_policy, parms.b_grid_size*parms.y_grid_size*sizeof(int));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for default_policy: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_bond_policy, parms.b_grid_size*parms.y_grid_size*sizeof(int));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for bond_policy: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_Err_q, parms.b_grid_size*parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for Err_q: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_Err_vr, parms.b_grid_size*parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for Err_vr: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMalloc((void**)&d_Err_vd, parms.y_grid_size*sizeof(double));
    if (cs != cudaSuccess) {
        mexPrintf("Error allocating memory in the device for Err_vd: %s\n", cudaGetErrorString(cs));
    }
    // Copy the data from the host to the device and check for errors:
    cs = cudaMemcpy(d_b_grid, b_grid, parms.b_grid_size*sizeof(double), cudaMemcpyHostToDevice);
    if (cs != cudaSuccess) {
        mexPrintf("Error copying data from host to device for b_grid: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMemcpy(d_y_grid, y_grid, parms.y_grid_size*sizeof(double), cudaMemcpyHostToDevice);
    if (cs != cudaSuccess) {
        mexPrintf("Error copying data from host to device for y_grid: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMemcpy(d_p_grid, p_grid, parms.y_grid_size*parms.y_grid_size*sizeof(double), cudaMemcpyHostToDevice);
    if (cs != cudaSuccess) {
        mexPrintf("Error copying data from host to device for p_grid: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMemcpy(d_y_grid_under_default, y_grid_under_default, parms.y_grid_size*sizeof(double), cudaMemcpyHostToDevice);
    if (cs != cudaSuccess) {
        mexPrintf("Error copying data from host to device for y_grid_under_default: %s\n", cudaGetErrorString(cs));
    }

    // ! Apply Algorithm:
    fill_device_constants(parms);
    solve_arellano_model(parms, d_b_grid, d_y_grid, d_p_grid, d_y_grid_under_default, d_V, d_V_d_0, d_V_d_1, d_V_r_0, d_V_r_1, d_Q_0, d_Q_1, d_default_policy, d_bond_policy, d_Err_q, d_Err_vr, d_Err_vd);

    // ! Export to MATLAB:
    // Create pointers to matrices in MATLAB:
    mxArray* y_grid_matlab = mxCreateDoubleMatrix(parms.y_grid_size, 1, mxREAL);
    mxArray* b_grid_matlab = mxCreateDoubleMatrix(parms.b_grid_size, 1, mxREAL);
    mxArray* p_grid_matlab = mxCreateDoubleMatrix(parms.y_grid_size * parms.y_grid_size, 1, mxREAL);
    mxArray* y_grid_under_default_matlab = mxCreateDoubleMatrix(parms.y_grid_size, 1, mxREAL);
    mxArray* V_matlab = mxCreateDoubleMatrix(parms.b_grid_size * parms.y_grid_size, 1, mxREAL);
    mxArray* V_d_matlab = mxCreateDoubleMatrix(parms.y_grid_size, 1, mxREAL);
    mxArray* V_r_matlab = mxCreateDoubleMatrix(parms.b_grid_size * parms.y_grid_size, 1, mxREAL);
    mxArray* Q_matlab = mxCreateDoubleMatrix(parms.b_grid_size * parms.y_grid_size, 1, mxREAL);
    mxArray* default_policy_matlab = mxCreateNumericMatrix(parms.b_grid_size * parms.y_grid_size, 1, mxINT32_CLASS, mxREAL);
    mxArray* bond_policy_matlab = mxCreateNumericMatrix(parms.b_grid_size * parms.y_grid_size, 1, mxINT32_CLASS, mxREAL);

    // Create a C++ pointer to the matrices in MATLAB:
    double* y_grid_matlab_ptr = mxGetPr(y_grid_matlab);
    double* b_grid_matlab_ptr = mxGetPr(b_grid_matlab);
    double* p_grid_matlab_ptr = mxGetPr(p_grid_matlab);
    double* y_grid_under_default_matlab_ptr = mxGetPr(y_grid_under_default_matlab);
    double* V_matlab_ptr = mxGetPr(V_matlab);
    double* V_d_matlab_ptr = mxGetPr(V_d_matlab);
    double* V_r_matlab_ptr = mxGetPr(V_r_matlab);
    double* Q_matlab_ptr = mxGetPr(Q_matlab);
    int* default_policy_matlab_ptr = (int*)mxGetData(default_policy_matlab);
    int* bond_policy_matlab_ptr = (int*)mxGetData(bond_policy_matlab);

    
    // ! Take all the results to the host:
    cs = cudaMemcpy(V, d_V, parms.b_grid_size * parms.y_grid_size*sizeof(double), cudaMemcpyDeviceToHost);
    if (cs != cudaSuccess){
        mexPrintf("Error copying data from device to host for V: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMemcpy(V_d, d_V_d_1, parms.y_grid_size*sizeof(double), cudaMemcpyDeviceToHost);
    if (cs != cudaSuccess) {
        mexPrintf("Error copying data from device to host for V_d: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMemcpy(V_r, d_V_r_1, parms.b_grid_size * parms.y_grid_size*sizeof(double), cudaMemcpyDeviceToHost);
    if (cs != cudaSuccess) {
        mexPrintf("Error copying data from device to host for V_r: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMemcpy(Q, d_Q_1, parms.b_grid_size * parms.y_grid_size*sizeof(double), cudaMemcpyDeviceToHost);
    if (cs != cudaSuccess) {
        mexPrintf("Error copying data from device to host for Q: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMemcpy(default_policy, d_default_policy, parms.b_grid_size * parms.y_grid_size*sizeof(int), cudaMemcpyDeviceToHost);
    if (cs != cudaSuccess) {
        mexPrintf("Error copying data from device to host for default_policy: %s\n", cudaGetErrorString(cs));
    }
    cs = cudaMemcpy(bond_policy, d_bond_policy, parms.b_grid_size * parms.y_grid_size*sizeof(int), cudaMemcpyDeviceToHost);
    if (cs != cudaSuccess) {
        mexPrintf("Error copying data from device to host for bond_policy: %s\n", cudaGetErrorString(cs));
    }



    // ! Export result from the host to MATLAB.  
    // Copy the data from the host to the MATLAB pointers:
    copy_vector(y_grid, y_grid_matlab_ptr, parms.y_grid_size);
    copy_vector(b_grid, b_grid_matlab_ptr, parms.b_grid_size);
    copy_vector(p_grid, p_grid_matlab_ptr, parms.y_grid_size*parms.y_grid_size);
    copy_vector(y_grid_under_default, y_grid_under_default_matlab_ptr, parms.y_grid_size);
    copy_vector(V, V_matlab_ptr, parms.b_grid_size*parms.y_grid_size);
    copy_vector(V_d, V_d_matlab_ptr, parms.y_grid_size);
    copy_vector(V_r, V_r_matlab_ptr, parms.b_grid_size*parms.y_grid_size);
    copy_vector(Q, Q_matlab_ptr, parms.b_grid_size*parms.y_grid_size);
    copy_vector(default_policy, default_policy_matlab_ptr, parms.b_grid_size*parms.y_grid_size);
    copy_vector(bond_policy, bond_policy_matlab_ptr, parms.b_grid_size*parms.y_grid_size);



    // ! Export to MATLAB.  
    // Create the output struct:
    const int nfields = 10;
    const char* fieldNames[nfields] = {"y_grid", "b_grid", "P", "y_grid_under_default", "V", "V_d", "V_r", "Q", "default_policy", "bond_policy"};
    plhs[0] = mxCreateStructMatrix(1, 1, nfields, fieldNames);
    // Copy the matrices to the output struct:
    mxSetField(plhs[0], 0, "y_grid", y_grid_matlab);
    mxSetField(plhs[0], 0, "b_grid", b_grid_matlab);
    mxSetField(plhs[0], 0, "P", p_grid_matlab);
    mxSetField(plhs[0], 0, "y_grid_under_default", y_grid_under_default_matlab);
    mxSetField(plhs[0], 0, "V", V_matlab);
    mxSetField(plhs[0], 0, "V_d", V_d_matlab);
    mxSetField(plhs[0], 0, "V_r", V_r_matlab);
    mxSetField(plhs[0], 0, "Q", Q_matlab);
    mxSetField(plhs[0], 0, "default_policy", default_policy_matlab);
    mxSetField(plhs[0], 0, "bond_policy", bond_policy_matlab);
    


    // ! Free the memory:
    delete[] b_grid;
    delete[] y_grid;
    delete[] p_grid;
    delete[] y_grid_under_default;
    delete[] V;
    delete[] V_d;
    delete[] V_r;
    delete[] Q;
    delete[] default_policy;
    delete[] bond_policy;
    cudaFree(d_b_grid);
    cudaFree(d_y_grid);
    cudaFree(d_p_grid);
    cudaFree(d_y_grid_under_default);
    cudaFree(d_V);
    cudaFree(d_V_d_0);
    cudaFree(d_V_d_1);
    cudaFree(d_V_r_0);
    cudaFree(d_V_r_1);
    cudaFree(d_Q_0);
    cudaFree(d_Q_1);
    cudaFree(d_default_policy);
    cudaFree(d_bond_policy);
    cudaFree(d_Err_q);
    cudaFree(d_Err_vr);
    cudaFree(d_Err_vd);
}