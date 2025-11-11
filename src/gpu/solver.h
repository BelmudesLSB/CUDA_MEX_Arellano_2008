#ifndef SOLVER_H
#define SOLVER_H

#include "../host/helpers.h"

__global__ void d_guess_vd_vr_q(double* d_V_r_0, double* d_V_d_0, double* d_Q_0);

__global__ void update_v_and_default_policy(double* d_V, double* d_V_d_0, double* d_V_r_0, int* d_default_policy);

__global__ void update_price(int d_default_policy, double* d_Q_1, double* d_p_grid);

__global__ void update_vd(double* d_V_d_0, double* d_V_d_1, double* d_V, double* d_p_grid, double* d_y_grid_under_default);

__global__ void update_vr_and_bond_policy(double* d_b_grid, double* d_Q_1, double* d_y_grid, double* d_p_grid, double* d_V_r_1, double* d_V, int* d_bond_policy);

__global__ void compute_distance(double* d_F, double* d_G, double* d_Err);

void solve_arellano_model(Parameters_host parms, double* d_b_grid, double* d_y_grid, double* d_p_grid, double* d_y_grid_under_default, double* d_V, double* d_V_d_0, double* d_V_d_1, double* d_V_r_0, double* d_V_r_1, double* d_Q_0, double* d_Q_1, int* d_default_policy, int* d_bond_policy, double* d_Err_q, double* d_Err_vr, double* d_Err_vd);

void fill_device_constants(Parameters_host parms);
#endif