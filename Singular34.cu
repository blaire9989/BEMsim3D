/* 4-GPU implementation for computing elements in the BEM matrix with singularities in the underlying integrals.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "Singular34.h"

/// @brief 4-GPU implementation for computing elements in the BEM matrix with singularities in the underlying integrals
/// @param est: an Estimate object
/// @param ind0, ind1, ind2, ind3: indices of the GPUs used
Singular34::Singular34(Estimate* est, int ind0, int ind1, int ind2, int ind3): Singular(est) {
    devNumber[0] = ind0;
    devNumber[1] = ind1;
    devNumber[2] = ind2;
    devNumber[3] = ind3;
    computeInvariants();
}

/// @brief Driver code for computing all the singular matrix elements
void Singular34::computeInvariants() {
    // Allocate memory on the GPUs
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMalloc(&d_zvals[dev], (Nx + 1) * (Ny + 1) * sizeof(float));
        cudaMemcpy(d_zvals[dev], (float*)(zvals.data()), (Nx + 1) * (Ny + 1) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_p1[dev], order1 * sizeof(float));
        cudaMemcpy(d_p1[dev], (float*)(p1.data()), order1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_w1[dev], order1 * sizeof(float));
        cudaMemcpy(d_w1[dev], (float*)(w1.data()), order1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_p2[dev], order2 * sizeof(float));
        cudaMemcpy(d_p2[dev], (float*)(p2.data()), order2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_w2[dev], order2 * sizeof(float));
        cudaMemcpy(d_w2[dev], (float*)(w2.data()), order2 * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Perform relevant computations on the GPUs
    selfEE();
    neighborEE();
    neighborEM();

    // Deallocate memory on the GPUs
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaFree(d_zvals[dev]);
        cudaFree(d_p1[dev]);
        cudaFree(d_w1[dev]);
        cudaFree(d_p2[dev]);
        cudaFree(d_w2[dev]);
        cudaFree(d_self[dev]);
        cudaFree(d_hori[dev]);
        cudaFree(d_vert[dev]);
    }
}

/// @brief Compute elements in the diagonal blocks of the BEM matrix that involve overlapping basis element pairs
void Singular34::selfEE() {
    int quota = ceil(Nx * Ny / 4.0);
    Vector4i total, offset;
    total << quota, quota, quota, Nx * Ny - 3 * quota;
    offset << 0, quota, 2 * quota, 3 * quota;
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMalloc(&d_self[dev], 11 * total(dev) * sizeof(float));
    }

    // Compute self terms
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        clearData<<< total(dev) / 256 + 1, 256 >>>(d_self[dev], total(dev));
        computeSelfEE<<< total(dev) / 256 + 1, 256 >>>(total(dev), offset(dev), d_self[dev], Nx, Ny, d_zvals[dev], d, order1, d_p1[dev], d_w1[dev]);
    }

    // Transfer data to the host and reformat
    LS1 = MatrixXf::Zero(4 * Nx, 4 * Ny);
    LS2 = MatrixXf::Zero(4 * Nx, 4 * Ny);
    Matrix4f div_pattern;
    div_pattern << -1.0, 1.0, -1.0, 1.0,
                   0.0, -1.0, 1.0, -1.0,
                   0.0, 0.0, -1.0, 1.0,
                   0.0, 0.0, 0.0, -1.0;
    for (int dev = 0; dev < 4; dev++) {
        MatrixXf self = MatrixXf::Zero(11, total(dev));
        cudaSetDevice(devNumber[dev]);
        cudaMemcpy((float*)(self.data()), d_self[dev], 11 * total(dev) * sizeof(float), cudaMemcpyDeviceToHost);
        parallel_for(total(dev), [&](int start, int end) {
        for (int i = start; i < end; i++) {
            int ind = i + offset(dev);
            int num_x = ind / Ny;
            int num_y = ind - Ny * num_x;
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
}

/// @brief Compute elements in the diagonal blocks of the BEM matrix that involve neighboring basis element pairs
void Singular34::neighborEE() {
    int quota;
    Vector4i total, offset;
    Matrix4f div_pattern;
    div_pattern << -1.0, 1.0, -1.0, 1.0,
                   1.0, -1.0, 1.0, -1.0,
                   -1.0, 1.0, -1.0, 1.0,
                   1.0, -1.0, 1.0, -1.0;

    // Horizontal neighbor terms
    quota = ceil(Ny * (Nx - 1) / 4.0);
    total << quota, quota, quota, (Nx - 1) * Ny - 3 * quota;
    offset << 0, quota, 2 * quota, 3 * quota;
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMalloc(&d_hori[dev], 17 * total(dev) * sizeof(float));
    }

    // Compute on GPUs
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        clearData<<< total(dev) / 256 + 1, 256 >>>(d_hori[dev], total(dev));
        computeNeighborEE<<< total(dev) / 256 + 1, 256 >>>(total(dev), offset(dev), 0, d_hori[dev], Nx, Ny, d_zvals[dev], d, order2, d_p2[dev], d_w2[dev]);
    }

    // Transfer data to the host and reformat
    LH1 = MatrixXf::Zero(4 * (Nx - 1), 4 * Ny);
    LH2 = MatrixXf::Zero(4 * (Nx - 1), 4 * Ny);
    for (int dev = 0; dev < 4; dev++) {
        MatrixXf hori = MatrixXf::Zero(17, total(dev));
        cudaSetDevice(devNumber[dev]);
        cudaMemcpy((float*)(hori.data()), d_hori[dev], 17 * total(dev) * sizeof(float), cudaMemcpyDeviceToHost);
        parallel_for(total(dev), [&](int start, int end) {
        for (int i = start; i < end; i++) {
            int ind = i + offset(dev);
            int num_y = ind / (Nx - 1);
            int num_x = ind - (Nx - 1) * num_y;
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
    }

    // Vertical neighbor terms
    quota = ceil(Nx * (Ny - 1) / 4.0);
    total << quota, quota, quota, (Ny - 1) * Nx - 3 * quota;
    offset << 0, quota, 2 * quota, 3 * quota;
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMalloc(&d_vert[dev], 17 * total(dev) * sizeof(float));
    }

    // Compute on GPUs
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        clearData<<< total(dev) / 256 + 1, 256 >>>(d_vert[dev], total(dev));
        computeNeighborEE<<< total(dev) / 256 + 1, 256 >>>(total(dev), offset(dev), 1, d_vert[dev], Nx, Ny, d_zvals[dev], d, order2, d_p2[dev], d_w2[dev]);
    }

    // Transfer data to the host and reformat
    LV1 = MatrixXf::Zero(4 * Nx, 4 * (Ny - 1));
    LV2 = MatrixXf::Zero(4 * Nx, 4 * (Ny - 1));
    for (int dev = 0; dev < 4; dev++) {
        MatrixXf vert = MatrixXf::Zero(17, total(dev));
        cudaSetDevice(devNumber[dev]);
        cudaMemcpy((float*)(vert.data()), d_vert[dev], 17 * total(dev) * sizeof(float), cudaMemcpyDeviceToHost);
        parallel_for(total(dev), [&](int start, int end) {
        for (int i = start; i < end; i++) {
            int ind = i + offset(dev);
            int num_x = ind / (Ny - 1);
            int num_y = ind - (Ny - 1) * num_x;
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
}