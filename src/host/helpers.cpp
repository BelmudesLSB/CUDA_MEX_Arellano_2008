#include <iostream>
#include <cmath>
#include "helpers.h"


// Construction of bond grids:
void create_bond_grids(double* prt_bond_grid, int Nb, double Bmax, double Bmin){
    double bstep = (Bmax - Bmin)/(Nb - 1);
    for(int i = 0; i < Nb; i++){
        prt_bond_grid[i] = Bmin + i*bstep;
    }
}

// Construction of transition probability matrix and income grids:
void create_income_and_prob_grids(double* prt_y_grid, double* prt_p_grid,  int Ny,  double Sigma,  double Rho,  double M){
    double sigma_y = sqrt(pow(Sigma,2)/(1-pow(Rho,2)));
    double omega = (2*M*sigma_y)/(Ny-1);
    for (int i=0; i<Ny; i++){ 
        prt_y_grid[i] = (-M*sigma_y)  + omega * i;
    }   
    for (int i=0; i<Ny; i++){
        for (int j=0; j<Ny; j++){
            if (j==0 || j==Ny-1){
                if (j==0){
                    prt_p_grid[i*Ny+j] = normalCDF((prt_y_grid[0]-Rho*prt_y_grid[i]+omega/2)/Sigma);
                }
                else {
                    prt_p_grid[i*Ny+j] = 1-normalCDF((prt_y_grid[Ny-1]-Rho*prt_y_grid[i]-omega/2)/Sigma);
                }
            } else {
                prt_p_grid[i*Ny+j] = normalCDF((prt_y_grid[j]-Rho*prt_y_grid[i]+omega/2)/Sigma)-normalCDF((prt_y_grid[j]-Rho*prt_y_grid[i]-omega/2)/Sigma);
            }
        }
    }
    for (int i=0; i<Ny; i++){
        prt_y_grid[i] = exp(prt_y_grid[i]);
    }
}

// Create the income grid for the default state:
void create_income_under_default(double* prt_y_grid_default, double* prt_y_grid,  int Ny,  double y_def){
    for (int i=0; i<Ny; i++){
        if (prt_y_grid[i]>y_def){
            prt_y_grid_default[i] = y_def;
        } else {
            prt_y_grid_default[i] = prt_y_grid[i];
        }
    }
}

// Normal cumulative distribution function:
double normalCDF(double x){
    return std::erfc(-x / std::sqrt(2)) / 2;
}

// Utility function:
double utility(double c,  double gamma, double c_lb){
    if (c>=c_lb){
        return pow(c,1-gamma)/(1-gamma);
    } else {
        return -1000000;
    }
}

// Copy vector:
void copy_vector(double* prt_vector, double* prt_vector_copy, int size){
    for (int i=0; i<size; i++){
        prt_vector_copy[i] = prt_vector[i];
    }
}

// Copy vector:
void copy_vector(int* prt_vector, int* prt_vector_copy, int size){
    for (int i=0; i<size; i++){
        prt_vector_copy[i] = (prt_vector[i]);
    }
}
