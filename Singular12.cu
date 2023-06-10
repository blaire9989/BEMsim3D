/* 1-GPU implementation for computing elements in the BEM matrix with singularities in the underlying integrals.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "Singular12.h"

/// @brief 1-GPU implementation for computing elements in the BEM matrix with singularities in the underlying integrals
/// @param est: an Estimate object
/// @param ind: index of the GPU used
Singular12::Singular12(Estimate* est, int ind): Singular(est) {
    cudaSetDevice(ind);
    computeInvariants();
}

/// @brief Driver code for computing all the singular matrix elements
void Singular12::computeInvariants() {
    // Allocate memory on the GPU
    cudaMalloc(&d_zvals, (Nx + 1) * (Ny + 1) * sizeof(float));
    cudaMemcpy(d_zvals, (float*)(zvals.data()), (Nx + 1) * (Ny + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_p1, order1 * sizeof(float));
    cudaMemcpy(d_p1, (float*)(p1.data()), order1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_w1, order1 * sizeof(float));
    cudaMemcpy(d_w1, (float*)(w1.data()), order1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_p2, order2 * sizeof(float));
    cudaMemcpy(d_p2, (float*)(p2.data()), order2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_w2, order2 * sizeof(float));
    cudaMemcpy(d_w2, (float*)(w2.data()), order2 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform relevant computations on the GPU
    selfEE();
    neighborEE();
    neighborEM();

    // Deallocate memory on the GPU
    cudaFree(d_zvals);
    cudaFree(d_p1);
    cudaFree(d_w1);
    cudaFree(d_p2);
    cudaFree(d_w2);
    cudaFree(d_self);
    cudaFree(d_hori);
    cudaFree(d_vert);
}

/// @brief Compute elements in the diagonal blocks of the BEM matrix that involve overlapping basis element pairs
void Singular12::selfEE() {
    int total = Nx * Ny;
    cudaMalloc(&d_self, 11 * total * sizeof(float));
    clearData<<< total / 256 + 1, 256 >>>(d_self, total);
    computeSelfEE<<< total / 256 + 1, 256 >>>(total, 0, d_self, Nx, Ny, d_zvals, d, order1, d_p1, d_w1);

    // Transfer data to the host and reformat
    LS1 = MatrixXf::Zero(4 * Nx, 4 * Ny);
    LS2 = MatrixXf::Zero(4 * Nx, 4 * Ny);
    Matrix4f div_pattern;
    div_pattern << -1.0, 1.0, -1.0, 1.0,
                   0.0, -1.0, 1.0, -1.0,
                   0.0, 0.0, -1.0, 1.0,
                   0.0, 0.0, 0.0, -1.0;
    MatrixXf self = MatrixXf::Zero(11, total);
    cudaMemcpy((float*)(self.data()), d_self, 11 * total * sizeof(float), cudaMemcpyDeviceToHost);
    parallel_for(total, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        int num_x = i / Ny;
        int num_y = i - Ny * num_x;
        LS1(4 * num_x + 0, 4 * num_y + 0) = self(0, i);
        LS1(4 * num_x + 0, 4 * num_y + 1) = self(1, i);
        LS1(4 * num_x + 0, 4 * num_y + 2) = self(2, i);
        LS1(4 * num_x + 0, 4 * num_y + 3) = self(3, i);
        LS1(4 * num_x + 1, 4 * num_y + 1) = self(4, i);
        LS1(4 * num_x + 1, 4 * num_y + 2) = self(5, i);
        LS1(4 * num_x + 1, 4 * num_y + 3) = self(6, i);
        LS1(4 * num_x + 2, 4 * num_y + 2) = self(7, i);
        LS1(4 * num_x + 2, 4 * num_y + 3) = self(8, i);
        LS1(4 * num_x + 3, 4 * num_y + 3) = self(9, i);
        LS2.block(4 * num_x, 4 * num_y, 4, 4) = self(10, i) * div_pattern;
    }
    } );
}

/// @brief Compute elements in the diagonal blocks of the BEM matrix that involve neighboring basis element pairs
void Singular12::neighborEE() {
    int total;
    Matrix4f div_pattern;
    div_pattern << -1.0, 1.0, -1.0, 1.0,
                   1.0, -1.0, 1.0, -1.0,
                   -1.0, 1.0, -1.0, 1.0,
                   1.0, -1.0, 1.0, -1.0;

    // Horizontal neighbor terms
    total = (Nx - 1) * Ny;
    cudaMalloc(&d_hori, 17 * total * sizeof(float));
    clearData<<< total / 256 + 1, 256 >>>(d_hori, total);
    computeNeighborEE<<< total / 256 + 1, 256 >>>(total, 0, 0, d_hori, Nx, Ny, d_zvals, d, order2, d_p2, d_w2);

    // Transfer data to the host and reformat
    LH1 = MatrixXf::Zero(4 * (Nx - 1), 4 * Ny);
    LH2 = MatrixXf::Zero(4 * (Nx - 1), 4 * Ny);
    MatrixXf hori = MatrixXf::Zero(17, total);
    cudaMemcpy((float*)(hori.data()), d_hori, 17 * total * sizeof(float), cudaMemcpyDeviceToHost);
    parallel_for(total, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        int num_y = i / (Nx - 1);
        int num_x = i - (Nx - 1) * num_y;
        LH1(4 * num_x + 0, 4 * num_y + 0) = hori(0, i);
        LH1(4 * num_x + 0, 4 * num_y + 1) = hori(1, i);
        LH1(4 * num_x + 0, 4 * num_y + 2) = hori(2, i);
        LH1(4 * num_x + 0, 4 * num_y + 3) = hori(3, i);
        LH1(4 * num_x + 1, 4 * num_y + 0) = hori(4, i);
        LH1(4 * num_x + 1, 4 * num_y + 1) = hori(5, i);
        LH1(4 * num_x + 1, 4 * num_y + 2) = hori(6, i);
        LH1(4 * num_x + 1, 4 * num_y + 3) = hori(7, i);
        LH1(4 * num_x + 2, 4 * num_y + 0) = hori(8, i);
        LH1(4 * num_x + 2, 4 * num_y + 1) = hori(9, i);
        LH1(4 * num_x + 2, 4 * num_y + 2) = hori(10, i);
        LH1(4 * num_x + 2, 4 * num_y + 3) = hori(11, i);
        LH1(4 * num_x + 3, 4 * num_y + 0) = hori(12, i);
        LH1(4 * num_x + 3, 4 * num_y + 1) = hori(13, i);
        LH1(4 * num_x + 3, 4 * num_y + 2) = hori(14, i);
        LH1(4 * num_x + 3, 4 * num_y + 3) = hori(15, i);
        LH2.block(4 * num_x, 4 * num_y, 4, 4) = hori(16, i) * div_pattern;
    }
    } );

    // Vertical neighbor terms
    total = (Ny - 1) * Nx;
    cudaMalloc(&d_vert, 17 * total * sizeof(float));
    clearData<<< total / 256 + 1, 256 >>>(d_vert, total);
    computeNeighborEE<<< total / 256 + 1, 256 >>>(total, 0, 1, d_vert, Nx, Ny, d_zvals, d, order2, d_p2, d_w2);

    // Transfer data to the host and reformat
    LV1 = MatrixXf::Zero(4 * Nx, 4 * (Ny - 1));
    LV2 = MatrixXf::Zero(4 * Nx, 4 * (Ny - 1));
    MatrixXf vert = MatrixXf::Zero(17, total);
    cudaMemcpy((float*)(vert.data()), d_vert, 17 * total * sizeof(float), cudaMemcpyDeviceToHost);
    parallel_for(total, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        int num_x = i / (Ny - 1);
        int num_y = i - (Ny - 1) * num_x;
        LV1(4 * num_x + 0, 4 * num_y + 0) = vert(0, i);
        LV1(4 * num_x + 0, 4 * num_y + 1) = vert(1, i);
        LV1(4 * num_x + 0, 4 * num_y + 2) = vert(2, i);
        LV1(4 * num_x + 0, 4 * num_y + 3) = vert(3, i);
        LV1(4 * num_x + 1, 4 * num_y + 0) = vert(4, i);
        LV1(4 * num_x + 1, 4 * num_y + 1) = vert(5, i);
        LV1(4 * num_x + 1, 4 * num_y + 2) = vert(6, i);
        LV1(4 * num_x + 1, 4 * num_y + 3) = vert(7, i);
        LV1(4 * num_x + 2, 4 * num_y + 0) = vert(8, i);
        LV1(4 * num_x + 2, 4 * num_y + 1) = vert(9, i);
        LV1(4 * num_x + 2, 4 * num_y + 2) = vert(10, i);
        LV1(4 * num_x + 2, 4 * num_y + 3) = vert(11, i);
        LV1(4 * num_x + 3, 4 * num_y + 0) = vert(12, i);
        LV1(4 * num_x + 3, 4 * num_y + 1) = vert(13, i);
        LV1(4 * num_x + 3, 4 * num_y + 2) = vert(14, i);
        LV1(4 * num_x + 3, 4 * num_y + 3) = vert(15, i);
        LV2.block(4 * num_x, 4 * num_y, 4, 4) = vert(16, i) * div_pattern;
    }
    } );
}