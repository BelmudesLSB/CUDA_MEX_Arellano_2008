#ifndef HELPERS_H
#define HELPERS_H

#include <iostream>

// Construction of bond grids:
void create_bond_grids(double* prt_bond_grid, int Nb, double Bmax, double Bmin);

// Construction of transition probability matrix and income grids:
void create_income_and_prob_grids(double* prt_y_grid, double* prt_p_grid,  int Ny,  double Sigma,  double Rho,  double M);

// Create the income grid for the default state:
void create_income_under_default(double* prt_y_grid_default, double* prt_y_grid,  int Ny,  double y_def);

// Normal cumulative distribution function:
double normalCDF(double x);

// copy vector:
void copy_vector(double* prt_vector1, double* prt_vector2, int size);

// copy vector:
void copy_vector(int* prt_vector1, int* prt_vector2, int size);

// Utility function:
double utility(double c,  double gamma, double c_lb);

// Parameter class:
class Parameters_host
{
    public:
        int b_grid_size;
        double b_grid_min;
        double b_grid_max;
        int y_grid_size;
        double y_default;
        double beta;
        double gamma;
        double r;
        double rho;
        double sigma;
        double theta;
        double tol;
        int max_iter;
        double m;
};



#endif
