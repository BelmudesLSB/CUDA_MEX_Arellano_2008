%% Clear all:
clc;
clear all;

%% Compile the MEXCUDA file:
mexcuda main.cu aux_host.cpp arellano.cu

%% Calibrate the model:
params.b_grid_size = 256;              % Number of points in the grid for the bond price.
params.b_grid_min = -0.8;               % Minimum value of the bond price.
params.b_grid_max = 0.00;               % Maximum value of the bond price.
params.y_grid_size = 51;                % Number of points in the grid for the income.
params.y_default = 0.969;               % Maximum income under default.
params.beta = 0.953;                    % Discount factor.
params.gamma = 2;                       % Risk aversion.
params.r = 0.017;                       % Interest rate.
params.rho = 0.945;                     % Persistence of the income.
params.sigma = 0.025;                   % Standard deviation of the income.
params.theta = 0.282;                   % Probability of a re-entry.
params.max_iter = 2000;                 % Maximum number of iterations.
params.tol = 1e-7;                      % Tolerance for the convergence.
params.m = 3;                           % Number of standard deviations for the income grid.

%% Run the MEXCUDA and store results:

tic;
solution = main(params);
toc;

