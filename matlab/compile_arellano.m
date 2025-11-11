%% ============================================
%  File: compile_arellano.m
%  Purpose: Compile the CUDA implementation of Arellano (2008) model
%  Usage: Run this script from the project root folder
%  ============================================

clear; clc;
disp('=== Compiling Arellano (2008) sovereign default model ===');

% --- Adjust this path if needed ---
cd('C:\Users\belmu\OneDrive\Escritorio\Lucas\Repositories\Arellano_CUDA_MEX');

% Remove previous compiled files
delete('src\*.mex*');

% Check for GPU
if gpuDeviceCount == 0
    error('No GPU detected. Please check your CUDA setup.');
end

% --- Direct compile command ---
disp('Compiling with MEXCUDA...');

mexcuda('-output', 'src/arellano_mex', ...
        'src/gpu/mex_entrypoint.cu', ...
        'src/gpu/solver.cu', ...
        'src/host/helpers.cpp');

disp('✅ Compilation completed successfully.');
disp('You can now call the model from MATLAB using:');
disp('>> arellano_mex');