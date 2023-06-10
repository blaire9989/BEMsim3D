/* 4-GPU implementation for computing matrix-vector products in BEM, for lossy surface materials.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "MVProd4.h"

/// @brief 4-GPU implementation for computing matrix-vector products in BEM, for lossy surface materials
/// @param est: an Estimate object with information on the simulated surface
/// @param singular: a Singular object that computes the matrix elements with singularties in the underlying integrals
/// @param grid: a Grid object with information on the 3D grid of point sources
/// @param ind0, ind1, ind2, ind3: indices of the GPUs used
MVProd4::MVProd4(Estimate* est, Singular* singular, Grid* grid, int ind0, int ind1, int ind2, int ind3): MVProd(est, singular, grid) {
    alpha.x = 1.0f;
    alpha.y = 0.0f;
    beta.x = 1.0f;
    beta.y = 0.0f;
    h_y1 = VectorXcf::Zero(hori_num);
    h_y2 = VectorXcf::Zero(vert_num);
    h_y3 = VectorXcf::Zero(hori_num);
    h_y4 = VectorXcf::Zero(vert_num);
    devNumber[0] = ind0;
    devNumber[1] = ind1;
    devNumber[2] = ind2;
    devNumber[3] = ind3;
    cusparseCreate(&handle0);
    cusparseCreate(&handle1);
    cusparseCreate(&handle2);
    cusparseCreate(&handle3);
    initializeNear();
    initializeFar();
}

/// @brief Allocate sparse correction matrices and the data structures holding the input, output vectors on the GPU
void MVProd4::initializeNear() {
    // Sparsity patterns are stored in binary files and transferred to GPUs
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        nnzA[dev] = est->A[dev].rows();
        cudaMalloc(&d_Arows[dev], nnzA[dev] * sizeof(int));
        cudaMemcpy(d_Arows[dev], (int*)((est->A[dev].block(0, 0, nnzA[dev], 1)).data()), nnzA[dev] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_Acols[dev], nnzA[dev] * sizeof(int));
        cudaMemcpy(d_Acols[dev], (int*)((est->A[dev].block(0, 1, nnzA[dev], 1)).data()), nnzA[dev] * sizeof(int), cudaMemcpyHostToDevice);
        nnzB[dev] = est->B[dev].rows();
        cudaMalloc(&d_Brows[dev], nnzB[dev] * sizeof(int));
        cudaMemcpy(d_Brows[dev], (int*)((est->B[dev].block(0, 0, nnzB[dev], 1)).data()), nnzB[dev] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_Bcols[dev], nnzB[dev] * sizeof(int));
        cudaMemcpy(d_Bcols[dev], (int*)((est->B[dev].block(0, 1, nnzB[dev], 1)).data()), nnzB[dev] * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    // Create sparse matrices and dense vectors on GPUs
    Vector4i numrows(hori_num, hori_num, hori_num, vert_num);
    Vector4i numcols(hori_num, vert_num, vert_num, vert_num);
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMalloc(&d_Aee[dev], nnzA[dev] * sizeof(cuComplex));
        cudaMalloc(&d_Aem[dev], nnzA[dev] * sizeof(cuComplex));
        cudaMalloc(&d_Amm[dev], nnzA[dev] * sizeof(cuComplex));
        cudaMalloc(&d_Bee[dev], nnzB[dev] * sizeof(cuComplex));
        cudaMalloc(&d_Bem[dev], nnzB[dev] * sizeof(cuComplex));
        cudaMalloc(&d_Bmm[dev], nnzB[dev] * sizeof(cuComplex));
        cusparseCreateCoo(&Aee[dev], numrows(dev), numcols(dev), nnzA[dev], d_Arows[dev], d_Acols[dev], d_Aee[dev], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cusparseCreateCoo(&Aem[dev], numrows(dev), numcols(dev), nnzA[dev], d_Arows[dev], d_Acols[dev], d_Aem[dev], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cusparseCreateCoo(&Amm[dev], numrows(dev), numcols(dev), nnzA[dev], d_Arows[dev], d_Acols[dev], d_Amm[dev], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cusparseCreateCoo(&Bee[dev], numrows(dev), numcols(dev), nnzB[dev], d_Brows[dev], d_Bcols[dev], d_Bee[dev], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cusparseCreateCoo(&Bem[dev], numrows(dev), numcols(dev), nnzB[dev], d_Brows[dev], d_Bcols[dev], d_Bem[dev], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cusparseCreateCoo(&Bmm[dev], numrows(dev), numcols(dev), nnzB[dev], d_Brows[dev], d_Bcols[dev], d_Bmm[dev], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cudaMalloc(&d_x1[dev], hori_num * sizeof(cuComplex));
        cudaMalloc(&d_x2[dev], vert_num * sizeof(cuComplex));
        cudaMalloc(&d_x3[dev], hori_num * sizeof(cuComplex));
        cudaMalloc(&d_x4[dev], vert_num * sizeof(cuComplex));
        cudaMalloc(&d_y1[dev], hori_num * sizeof(cuComplex));
        cudaMalloc(&d_y2[dev], vert_num * sizeof(cuComplex));
        cudaMalloc(&d_y3[dev], hori_num * sizeof(cuComplex));
        cudaMalloc(&d_y4[dev], vert_num * sizeof(cuComplex));
        cusparseCreateDnVec(&x1[dev], hori_num, d_x1[dev], CUDA_C_32F);
        cusparseCreateDnVec(&x2[dev], vert_num, d_x2[dev], CUDA_C_32F);
        cusparseCreateDnVec(&x3[dev], hori_num, d_x3[dev], CUDA_C_32F);
        cusparseCreateDnVec(&x4[dev], vert_num, d_x4[dev], CUDA_C_32F);
        cusparseCreateDnVec(&y1[dev], hori_num, d_y1[dev], CUDA_C_32F);
        cusparseCreateDnVec(&y2[dev], vert_num, d_y2[dev], CUDA_C_32F);
        cusparseCreateDnVec(&y3[dev], hori_num, d_y3[dev], CUDA_C_32F);
        cusparseCreateDnVec(&y4[dev], vert_num, d_y4[dev], CUDA_C_32F);
        cudaMalloc(&d_work[dev], 0);
    }
    cudaDeviceSynchronize();

    // Transfer matrix element data that require singularity removal computation to GPUs
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        VectorXcf matrix_data;
        singular->computeQuarter(dev);
        matrix_data = singular->quarter.block(0, 0, nnzA[dev], 1);
        cudaMalloc(&d_base1[dev], nnzA[dev] * sizeof(cuComplex));
        cudaMemcpy(d_base1[dev], (fcomp*)(matrix_data.data()), nnzA[dev] * sizeof(cuComplex), cudaMemcpyHostToDevice);
        matrix_data = singular->quarter.block(0, 1, nnzA[dev], 1);
        cudaMalloc(&d_base2[dev], nnzA[dev] * sizeof(cuComplex));
        cudaMemcpy(d_base2[dev], (fcomp*)(matrix_data.data()), nnzA[dev] * sizeof(cuComplex), cudaMemcpyHostToDevice);
        matrix_data = singular->quarter.block(0, 2, nnzA[dev], 1);
        cudaMemcpy(d_Aem[dev], (fcomp*)(matrix_data.data()), nnzA[dev] * sizeof(cuComplex), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    // Allocate height field data and necessary quadrature points / weights
    MatrixXf zdata = est->zvals.cast<float>();
    VectorXf xhigh = quadrature_points.block(3, 0, 1, 4).transpose().cast<float>();
    VectorXf whigh = quadrature_weights.block(3, 0, 1, 4).transpose().cast<float>();
    VectorXf xlow = quadrature_points.block(1, 0, 1, 2).transpose().cast<float>();
    VectorXf wlow = quadrature_weights.block(1, 0, 1, 2).transpose().cast<float>();
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMalloc(&zvals[dev], (Nx + 1) * (Ny + 1) * sizeof(float));
        cudaMemcpy(zvals[dev], (float*)(zdata.data()), (Nx + 1) * (Ny + 1) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&xvech[dev], 4 * sizeof(float));
        cudaMemcpy(xvech[dev], (float*)(xhigh.data()), 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&wvech[dev], 4 * sizeof(float));
        cudaMemcpy(wvech[dev], (float*)(whigh.data()), 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&xvecl[dev], 2 * sizeof(float));
        cudaMemcpy(xvecl[dev], (float*)(xlow.data()), 2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&wvecl[dev], 2 * sizeof(float));
        cudaMemcpy(wvecl[dev], (float*)(wlow.data()), 2 * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
}

/// @brief Allocate memory for point source approximation coefficients on the GPU; create FFT plans for computation
void MVProd4::initializeFar() {
    // Allocate point source approximation coefficients and indices
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMalloc(&d_hori_x[dev], num_pts * hori_num * sizeof(Tfcomp));
        cudaMalloc(&d_hori_z[dev], num_pts * hori_num * sizeof(Tfcomp));
        cudaMalloc(&d_hori_d[dev], num_pts * hori_num * sizeof(Tfcomp));
        cudaMalloc(&d_hori_f[dev], hori_row * hori_col * sizeof(int));
        cudaMalloc(&d_hori_b[dev], num_pts * hori_num * sizeof(int));
        cudaMalloc(&d_vert_y[dev], num_pts * vert_num * sizeof(Tfcomp));
        cudaMalloc(&d_vert_z[dev], num_pts * vert_num * sizeof(Tfcomp));
        cudaMalloc(&d_vert_d[dev], num_pts * vert_num * sizeof(Tfcomp));
        cudaMalloc(&d_vert_f[dev], vert_row * vert_col * sizeof(int));
        cudaMalloc(&d_vert_b[dev], num_pts * vert_num * sizeof(int));
    }
    cudaDeviceSynchronize();

    // Allocate far matrix multiplication data structures
    for (int dev = 0; dev < 2; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMalloc(&g0_data[dev], N * sizeof(Tfcomp));
        cudaMalloc(&g1_data[dev], N * sizeof(Tfcomp));
        cudaMalloc(&geo0_data[dev], N * sizeof(Tfcomp));
        cufftCreate(&plan[dev]);
        size_t* worksize = (size_t *)malloc(sizeof(size_t));
        cufftMakePlan3d(plan[dev], totalX, totalY, totalZ, CUFFT_C2C, worksize);
    }
    for (int dev = 2; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMalloc(&g2_data[dev - 2], N * sizeof(Tfcomp));
        cudaMalloc(&g3_data[dev - 2], N * sizeof(Tfcomp));
        cudaMalloc(&g4_data[dev - 2], N * sizeof(Tfcomp));
        cudaMalloc(&geo1_data[dev - 2], N * sizeof(Tfcomp));
        cudaMalloc(&geo2_data[dev - 2], N * sizeof(Tfcomp));
        cufftCreate(&plan[dev]);
        size_t* worksize = (size_t *)malloc(sizeof(size_t));
        cufftMakePlan3d(plan[dev], totalX, totalY, totalZ, CUFFT_C2C, worksize);
    }
    cudaDeviceSynchronize();
}

/// @brief Perform initializations for computing matrix-vector products in a simulation with given media parameters and wavelengths
/// @brief Compute the point source approximations, the sparse correction matrices, and Fourier transform of Green's functions
/// @param eta1: index of refraction of the medium where the light is incident from, usually 1.0 (air)
/// @param eta2: index of refraction of the surface material (could be complex-valued)
/// @param lambda: the currently simulated wavelength
void MVProd4::setParameters(double eta1, dcomp eta2, double lambda) {
    // Initialize parameters
    double omega = c / lambda * 2 * M_PI;
    dcomp eps1 = 1 / (mu * c * c) * eta1 * eta1;
    dcomp eps2 = 1 / (mu * c * c) * eta2 * eta2;
    Tfcomp e1 = Tfcomp((float)eta1, 0.0f);
    Tfcomp e2 = Tfcomp((float)real(eta2), (float)imag(eta2));
    Tfcomp k1 = (float)(2 * M_PI / lambda) * e1;
    Tfcomp k2 = (float)(2 * M_PI / lambda) * e2;
    Tfcomp const1 = Tfcomp((float)real(cuDB * omega * mu), (float)imag(cuDB * omega * mu));
    Tfcomp const21 = Tfcomp((float)real(cuDB / (omega * eps1)), (float)imag(cuDB / (omega * eps1)));
    Tfcomp const22 = Tfcomp((float)real(cuDB / (omega * eps2)), (float)imag(cuDB / (omega * eps2)));
    Tfcomp c1 = 2.0f * const1;
    Tfcomp c2 = const21 + const22;
    Tfcomp c3 = -(e1 * e1 + e2 * e2) * const1;
    Tfcomp c4 = -e1 * e1 * const21 - e2 * e2 * const22;
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        updateEEMM<<< nnzA[dev] / 256 + 1, 256 >>>(nnzA[dev], d_Aee[dev], d_Amm[dev], d_base1[dev], d_base2[dev], c1, c2, c3, c4);
    }
    cudaDeviceSynchronize();

    // Compute point source approximation coefficients and transfer to GPUs
    grid->computeCoefficients(eta1, eta2, lambda);
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMemcpy(d_hori_x[dev], (fcomp*)(grid->hori_x.data()), num_pts * hori_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hori_z[dev], (fcomp*)(grid->hori_z.data()), num_pts * hori_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hori_d[dev], (fcomp*)(grid->hori_d.data()), num_pts * hori_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hori_f[dev], (int*)(grid->hori_f.data()), hori_row * hori_col * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hori_b[dev], (int*)(grid->hori_b.data()), num_pts * hori_num * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vert_y[dev], (fcomp*)(grid->vert_y.data()), num_pts * vert_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vert_z[dev], (fcomp*)(grid->vert_z.data()), num_pts * vert_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vert_d[dev], (fcomp*)(grid->vert_d.data()), num_pts * vert_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vert_f[dev], (int*)(grid->vert_f.data()), vert_row * vert_col * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vert_b[dev], (int*)(grid->vert_b.data()), num_pts * vert_num * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    // Compute near matrix elements
    cudaSetDevice(devNumber[0]);
    clearData<<< nnzB[0] / 256 + 1, 256 >>>(d_Bee[0], nnzB[0]);
    clearData<<< nnzB[0] / 256 + 1, 256 >>>(d_Bem[0], nnzB[0]);
    clearData<<< nnzB[0] / 256 + 1, 256 >>>(d_Bmm[0], nnzB[0]);
    individual<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], Nx, Ny, zvals[0], d, k1, k2, const1, const21, const22, xvech[0], wvech[0], xvecl[0], wvecl[0], d_Brows[0], d_Bcols[0], 0, 0, 0, 0, 1, -1, -1, d_Bee[0], d_Bmm[0], d_Bem[0]);
    individual<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], Nx, Ny, zvals[0], d, k1, k2, const1, const21, const22, xvech[0], wvech[0], xvecl[0], wvecl[0], d_Brows[0], d_Bcols[0], 0, 0, 1, 0, 1, -1, 1, d_Bee[0], d_Bmm[0], d_Bem[0]);
    individual<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], Nx, Ny, zvals[0], d, k1, k2, const1, const21, const22, xvech[0], wvech[0], xvecl[0], wvecl[0], d_Brows[0], d_Bcols[0], 1, 0, 0, 0, 1, 1, -1, d_Bee[0], d_Bmm[0], d_Bem[0]);
    individual<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], Nx, Ny, zvals[0], d, k1, k2, const1, const21, const22, xvech[0], wvech[0], xvecl[0], wvecl[0], d_Brows[0], d_Bcols[0], 1, 0, 1, 0, 1, 1, 1, d_Bee[0], d_Bmm[0], d_Bem[0]);
    correctionHH<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], dx, dy, dz, k1, const1, const21, totalY, totalZ, d_Brows[0], d_Bcols[0], num_pts, d_hori_b[0], d_hori_x[0], d_hori_z[0], d_hori_d[0], d_Bee[0], d_Bem[0]);
    postProcess<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], true, eta0FL, eta1, eta2, d_Brows[0], d_Bcols[0], d_Bee[0], d_Bmm[0], d_Bem[0]);
    cudaSetDevice(devNumber[1]);
    clearData<<< nnzB[1] / 256 + 1, 256 >>>(d_Bee[1], nnzB[1]);
    clearData<<< nnzB[1] / 256 + 1, 256 >>>(d_Bem[1], nnzB[1]);
    clearData<<< nnzB[1] / 256 + 1, 256 >>>(d_Bmm[1], nnzB[1]);
    individual<<< nnzB[1] / 256 + 1, 256 >>>(nnzB[1], Nx, Ny, zvals[1], d, k1, k2, const1, const21, const22, xvech[1], wvech[1], xvecl[1], wvecl[1], d_Brows[1], d_Bcols[1], 0, 0, 0, 0, 2, -1, -1, d_Bee[1], d_Bmm[1], d_Bem[1]);
    individual<<< nnzB[1] / 256 + 1, 256 >>>(nnzB[1], Nx, Ny, zvals[1], d, k1, k2, const1, const21, const22, xvech[1], wvech[1], xvecl[1], wvecl[1], d_Brows[1], d_Bcols[1], 0, 0, 0, 1, 2, -1, 1, d_Bee[1], d_Bmm[1], d_Bem[1]);
    individual<<< nnzB[1] / 256 + 1, 256 >>>(nnzB[1], Nx, Ny, zvals[1], d, k1, k2, const1, const21, const22, xvech[1], wvech[1], xvecl[1], wvecl[1], d_Brows[1], d_Bcols[1], 1, 0, 0, 0, 2, 1, -1, d_Bee[1], d_Bmm[1], d_Bem[1]);
    individual<<< nnzB[1] / 256 + 1, 256 >>>(nnzB[1], Nx, Ny, zvals[1], d, k1, k2, const1, const21, const22, xvech[1], wvech[1], xvecl[1], wvecl[1], d_Brows[1], d_Bcols[1], 1, 0, 0, 1, 2, 1, 1, d_Bee[1], d_Bmm[1], d_Bem[1]);
    correctionHV<<< nnzB[1] / 256 + 1, 256 >>>(nnzB[1], dx, dy, dz, k1, const1, const21, totalY, totalZ, d_Brows[1], d_Bcols[1], num_pts, d_hori_b[1], d_vert_b[1], d_hori_x[1], d_hori_z[1], d_hori_d[1], d_vert_y[1], d_vert_z[1], d_vert_d[1], d_Bee[1], d_Bem[1]);
    postProcess<<< nnzB[1] / 256 + 1, 256 >>>(nnzB[1], false, eta0FL, eta1, eta2, d_Brows[1], d_Bcols[1], d_Bee[1], d_Bmm[1], d_Bem[1]);
    cudaSetDevice(devNumber[2]);
    clearData<<< nnzB[2] / 256 + 1, 256 >>>(d_Bee[2], nnzB[2]);
    clearData<<< nnzB[2] / 256 + 1, 256 >>>(d_Bem[2], nnzB[2]);
    clearData<<< nnzB[2] / 256 + 1, 256 >>>(d_Bmm[2], nnzB[2]);
    individual<<< nnzB[2] / 256 + 1, 256 >>>(nnzB[2], Nx, Ny, zvals[2], d, k1, k2, const1, const21, const22, xvech[2], wvech[2], xvecl[2], wvecl[2], d_Brows[2], d_Bcols[2], 0, 0, 0, 0, 2, -1, -1, d_Bee[2], d_Bmm[2], d_Bem[2]);
    individual<<< nnzB[2] / 256 + 1, 256 >>>(nnzB[2], Nx, Ny, zvals[2], d, k1, k2, const1, const21, const22, xvech[2], wvech[2], xvecl[2], wvecl[2], d_Brows[2], d_Bcols[2], 0, 0, 0, 1, 2, -1, 1, d_Bee[2], d_Bmm[2], d_Bem[2]);
    individual<<< nnzB[2] / 256 + 1, 256 >>>(nnzB[2], Nx, Ny, zvals[2], d, k1, k2, const1, const21, const22, xvech[2], wvech[2], xvecl[2], wvecl[2], d_Brows[2], d_Bcols[2], 1, 0, 0, 0, 2, 1, -1, d_Bee[2], d_Bmm[2], d_Bem[2]);
    individual<<< nnzB[2] / 256 + 1, 256 >>>(nnzB[2], Nx, Ny, zvals[2], d, k1, k2, const1, const21, const22, xvech[2], wvech[2], xvecl[2], wvecl[2], d_Brows[2], d_Bcols[2], 1, 0, 0, 1, 2, 1, 1, d_Bee[2], d_Bmm[2], d_Bem[2]);
    correctionHV<<< nnzB[2] / 256 + 1, 256 >>>(nnzB[2], dx, dy, dz, k1, const1, const21, totalY, totalZ, d_Brows[2], d_Bcols[2], num_pts, d_hori_b[2], d_vert_b[2], d_hori_x[2], d_hori_z[2], d_hori_d[2], d_vert_y[2], d_vert_z[2], d_vert_d[2], d_Bee[2], d_Bem[2]);
    postProcess<<< nnzB[2] / 256 + 1, 256 >>>(nnzB[2], false, eta0FL, eta1, eta2, d_Brows[2], d_Bcols[2], d_Bee[2], d_Bmm[2], d_Bem[2]);
    cudaSetDevice(devNumber[3]);
    clearData<<< nnzB[3] / 256 + 1, 256 >>>(d_Bee[3], nnzB[3]);
    clearData<<< nnzB[3] / 256 + 1, 256 >>>(d_Bem[3], nnzB[3]);
    clearData<<< nnzB[3] / 256 + 1, 256 >>>(d_Bmm[3], nnzB[3]);
    individual<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], Nx, Ny, zvals[3], d, k1, k2, const1, const21, const22, xvech[3], wvech[3], xvecl[3], wvecl[3], d_Brows[3], d_Bcols[3], 0, 0, 0, 0, 3, -1, -1, d_Bee[3], d_Bmm[3], d_Bem[3]);
    individual<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], Nx, Ny, zvals[3], d, k1, k2, const1, const21, const22, xvech[3], wvech[3], xvecl[3], wvecl[3], d_Brows[3], d_Bcols[3], 0, 0, 0, 1, 3, -1, 1, d_Bee[3], d_Bmm[3], d_Bem[3]);
    individual<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], Nx, Ny, zvals[3], d, k1, k2, const1, const21, const22, xvech[3], wvech[3], xvecl[3], wvecl[3], d_Brows[3], d_Bcols[3], 0, 1, 0, 0, 3, 1, -1, d_Bee[3], d_Bmm[3], d_Bem[3]);
    individual<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], Nx, Ny, zvals[3], d, k1, k2, const1, const21, const22, xvech[3], wvech[3], xvecl[3], wvecl[3], d_Brows[3], d_Bcols[3], 0, 1, 0, 1, 3, 1, 1, d_Bee[3], d_Bmm[3], d_Bem[3]);
    correctionVV<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], dx, dy, dz, k1, const1, const21, totalY, totalZ, d_Brows[3], d_Bcols[3], num_pts, d_vert_b[3], d_vert_y[3], d_vert_z[3], d_vert_d[3], d_Bee[3], d_Bem[3]);
    postProcess<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], true, eta0FL, eta1, eta2, d_Brows[3], d_Bcols[3], d_Bee[3], d_Bmm[3], d_Bem[3]);
    cudaDeviceSynchronize();

    // Compute Green's function values and perform Fourier transforms
    cudaSetDevice(devNumber[0]);
    computeGreens<<< N / 256 + 1, 256 >>>(g0_data[0], dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 0);
    cufftExecC2C(plan[0], (cufftComplex*)g0_data[0], (cufftComplex*)g0_data[0], CUFFT_FORWARD);
    computeGreens<<< N / 256 + 1, 256 >>>(g1_data[0], dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 1);
    cufftExecC2C(plan[0], (cufftComplex*)g1_data[0], (cufftComplex*)g1_data[0], CUFFT_FORWARD);
    cudaSetDevice(devNumber[1]);
    computeGreens<<< N / 256 + 1, 256 >>>(g0_data[1], dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 2);
    cufftExecC2C(plan[1], (cufftComplex*)g0_data[1], (cufftComplex*)g0_data[1], CUFFT_FORWARD);
    computeGreens<<< N / 256 + 1, 256 >>>(g1_data[1], dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 3);
    cufftExecC2C(plan[1], (cufftComplex*)g1_data[1], (cufftComplex*)g1_data[1], CUFFT_FORWARD);
    cudaSetDevice(devNumber[2]);
    computeGreens<<< N / 256 + 1, 256 >>>(g2_data[0], dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 4);
    cufftExecC2C(plan[2], (cufftComplex*)g2_data[0], (cufftComplex*)g2_data[0], CUFFT_FORWARD);
    computeGreens<<< N / 256 + 1, 256 >>>(g3_data[0], dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 5);
    cufftExecC2C(plan[2], (cufftComplex*)g3_data[0], (cufftComplex*)g3_data[0], CUFFT_FORWARD);
    computeGreens<<< N / 256 + 1, 256 >>>(g4_data[0], dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 6);
    cufftExecC2C(plan[2], (cufftComplex*)g4_data[0], (cufftComplex*)g4_data[0], CUFFT_FORWARD);
    cudaSetDevice(devNumber[3]);
    computeGreens<<< N / 256 + 1, 256 >>>(g2_data[1], dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 4);
    cufftExecC2C(plan[3], (cufftComplex*)g2_data[1], (cufftComplex*)g2_data[1], CUFFT_FORWARD);
    computeGreens<<< N / 256 + 1, 256 >>>(g3_data[1], dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 5);
    cufftExecC2C(plan[3], (cufftComplex*)g3_data[1], (cufftComplex*)g3_data[1], CUFFT_FORWARD);
    computeGreens<<< N / 256 + 1, 256 >>>(g4_data[1], dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 6);
    cufftExecC2C(plan[3], (cufftComplex*)g4_data[1], (cufftComplex*)g4_data[1], CUFFT_FORWARD);
    cudaDeviceSynchronize();
}

/// @brief Perform matrix-vector multiplication using the BEM matrix
/// @param x: the input vector
/// @return The product vector
VectorXcf MVProd4::multiply(VectorXcf x) {
    VectorXcf h_x1 = x.block(0, 0, hori_num, 1);
    VectorXcf h_x2 = x.block(hori_num, 0, vert_num, 1);
    VectorXcf h_x3 = x.block(hori_num + vert_num, 0, hori_num, 1);
    VectorXcf h_x4 = x.block(2 * hori_num + vert_num, 0, vert_num, 1);
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMemcpy(d_x1[dev], (fcomp*)(h_x1.data()), hori_num * sizeof(Tfcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x2[dev], (fcomp*)(h_x2.data()), vert_num * sizeof(Tfcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x3[dev], (fcomp*)(h_x3.data()), hori_num * sizeof(Tfcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x4[dev], (fcomp*)(h_x4.data()), vert_num * sizeof(Tfcomp), cudaMemcpyHostToDevice);
    }
    gpu0();
    gpu1();
    gpu2();
    gpu3();
    VectorXcf y = VectorXcf::Zero(2 * hori_num + 2 * vert_num);
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaMemcpy((fcomp*)(h_y1.data()), d_y1[dev], hori_num * sizeof(fcomp), cudaMemcpyDeviceToHost);
        cudaMemcpy((fcomp*)(h_y2.data()), d_y2[dev], vert_num * sizeof(fcomp), cudaMemcpyDeviceToHost);
        cudaMemcpy((fcomp*)(h_y3.data()), d_y3[dev], hori_num * sizeof(fcomp), cudaMemcpyDeviceToHost);
        cudaMemcpy((fcomp*)(h_y4.data()), d_y4[dev], vert_num * sizeof(fcomp), cudaMemcpyDeviceToHost);
        y.block(0, 0, hori_num, 1) += h_y1;
        y.block(hori_num, 0, vert_num, 1) += h_y2;
        y.block(hori_num + vert_num, 0, hori_num, 1) += h_y3;
        y.block(2 * hori_num + vert_num, 0, vert_num, 1) += h_y4;
    }
    return y;
}

/// @brief Matrix-vector multiplication computations done on GPU 0
void MVProd4::gpu0() {
    // Initialization: clearing all vectors to become zero-valued
    cudaSetDevice(devNumber[0]);
    cusparseDnVecSetValues(x1[0], d_x1[0]);
    cusparseDnVecSetValues(x2[0], d_x2[0]);
    cusparseDnVecSetValues(x3[0], d_x3[0]);
    cusparseDnVecSetValues(x4[0], d_x4[0]);
    clearData<<< hori_num / 256 + 1, 256 >>>(d_y1[0], hori_num);
    clearData<<< vert_num / 256 + 1, 256 >>>(d_y2[0], vert_num);
    clearData<<< hori_num / 256 + 1, 256 >>>(d_y3[0], hori_num);
    clearData<<< vert_num / 256 + 1, 256 >>>(d_y4[0], vert_num);
    cusparseDnVecSetValues(y1[0], d_y1[0]);
    cusparseDnVecSetValues(y2[0], d_y2[0]);
    cusparseDnVecSetValues(y3[0], d_y3[0]);
    cusparseDnVecSetValues(y4[0], d_y4[0]);

    // Sparse near matrix multiplication
    cusparseSpMV(handle0, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aee[0], x1[0], &beta, y1[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aee[0], x1[0], &beta, y1[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bee[0], x1[0], &beta, y1[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bee[0], x1[0], &beta, y1[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[0], x3[0], &beta, y1[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[0], x3[0], &beta, y1[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[0], x3[0], &beta, y1[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[0], x3[0], &beta, y1[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[0], x1[0], &beta, y3[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[0], x1[0], &beta, y3[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[0], x1[0], &beta, y3[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[0], x1[0], &beta, y3[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Amm[0], x3[0], &beta, y3[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Amm[0], x3[0], &beta, y3[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bmm[0], x3[0], &beta, y3[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);
    cusparseSpMV(handle0, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bmm[0], x3[0], &beta, y3[0], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[0]);

    // Far matrix multiplication for block Zee: FFT pair 1
    clearData<<< N / 256 + 1, 256 >>>(geo0_data[0], N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo0_data[0], d_x1[0], hori_row, hori_col, num_pts, d_hori_x[0], d_hori_f[0]);
    cufftExecC2C(plan[0], (cufftComplex*)geo0_data[0], (cufftComplex*)geo0_data[0], CUFFT_FORWARD);
    convolve<<< N / 256 + 1, 256 >>>(g0_data[0], geo0_data[0], N);
    cufftExecC2C(plan[0], (cufftComplex*)geo0_data[0], (cufftComplex*)geo0_data[0], CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1[0], geo0_data[0], hori_num, num_pts, N, d_hori_x[0], d_hori_b[0], false);
    
    // Far matrix multiplication for block Zee: FFT pair 2
    clearData<<< N / 256 + 1, 256 >>>(geo0_data[0], N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo0_data[0], d_x2[0], vert_row, vert_col, num_pts, d_vert_y[0], d_vert_f[0]);
    cufftExecC2C(plan[0], (cufftComplex*)geo0_data[0], (cufftComplex*)geo0_data[0], CUFFT_FORWARD);
    convolve<<< N / 256 + 1, 256 >>>(g0_data[0], geo0_data[0], N);
    cufftExecC2C(plan[0], (cufftComplex*)geo0_data[0], (cufftComplex*)geo0_data[0], CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2[0], geo0_data[0], vert_num, num_pts, N, d_vert_y[0], d_vert_b[0], false);
    
    // Far matrix multiplication for block Zee: FFT pair 3
    clearData<<< N / 256 + 1, 256 >>>(geo0_data[0], N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo0_data[0], d_x1[0], hori_row, hori_col, num_pts, d_hori_z[0], d_hori_f[0]);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo0_data[0], d_x2[0], vert_row, vert_col, num_pts, d_vert_z[0], d_vert_f[0]);
    cufftExecC2C(plan[0], (cufftComplex*)geo0_data[0], (cufftComplex*)geo0_data[0], CUFFT_FORWARD);
    convolve<<< N / 256 + 1, 256 >>>(g0_data[0], geo0_data[0], N);
    cufftExecC2C(plan[0], (cufftComplex*)geo0_data[0], (cufftComplex*)geo0_data[0], CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1[0], geo0_data[0], hori_num, num_pts, N, d_hori_z[0], d_hori_b[0], false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2[0], geo0_data[0], vert_num, num_pts, N, d_vert_z[0], d_vert_b[0], false);

    // Far matrix multiplication for block Zee: FFT pair 4
    clearData<<< N / 256 + 1, 256 >>>(geo0_data[0], N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo0_data[0], d_x1[0], hori_row, hori_col, num_pts, d_hori_d[0], d_hori_f[0]);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo0_data[0], d_x2[0], vert_row, vert_col, num_pts, d_vert_d[0], d_vert_f[0]);
    cufftExecC2C(plan[0], (cufftComplex*)geo0_data[0], (cufftComplex*)geo0_data[0], CUFFT_FORWARD);
    convolve<<< N / 256 + 1, 256 >>>(g1_data[0], geo0_data[0], N);
    cufftExecC2C(plan[0], (cufftComplex*)geo0_data[0], (cufftComplex*)geo0_data[0], CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1[0], geo0_data[0], hori_num, num_pts, N, d_hori_d[0], d_hori_b[0], true);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2[0], geo0_data[0], vert_num, num_pts, N, d_vert_d[0], d_vert_b[0], true);
}

/// @brief Matrix-vector multiplication computations done on GPU 1
void MVProd4::gpu1() {
    // Initialization: clearing all vectors to become zero-valued
    cudaSetDevice(devNumber[1]);
    cusparseDnVecSetValues(x1[1], d_x1[1]);
    cusparseDnVecSetValues(x2[1], d_x2[1]);
    cusparseDnVecSetValues(x3[1], d_x3[1]);
    cusparseDnVecSetValues(x4[1], d_x4[1]);
    clearData<<< hori_num / 256 + 1, 256 >>>(d_y1[1], hori_num);
    clearData<<< vert_num / 256 + 1, 256 >>>(d_y2[1], vert_num);
    clearData<<< hori_num / 256 + 1, 256 >>>(d_y3[1], hori_num);
    clearData<<< vert_num / 256 + 1, 256 >>>(d_y4[1], vert_num);
    cusparseDnVecSetValues(y1[1], d_y1[1]);
    cusparseDnVecSetValues(y2[1], d_y2[1]);
    cusparseDnVecSetValues(y3[1], d_y3[1]);
    cusparseDnVecSetValues(y4[1], d_y4[1]);

    // Sparse near matrix multiplication
    cusparseSpMV(handle1, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aee[1], x2[1], &beta, y1[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bee[1], x2[1], &beta, y1[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[1], x4[1], &beta, y1[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[1], x4[1], &beta, y1[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aee[1], x1[1], &beta, y2[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bee[1], x1[1], &beta, y2[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[1], x3[1], &beta, y2[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[1], x3[1], &beta, y2[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[1], x2[1], &beta, y3[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[1], x2[1], &beta, y3[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Amm[1], x4[1], &beta, y3[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bmm[1], x4[1], &beta, y3[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[1], x1[1], &beta, y4[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[1], x1[1], &beta, y4[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Amm[1], x3[1], &beta, y4[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);
    cusparseSpMV(handle1, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bmm[1], x3[1], &beta, y4[1], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[1]);

    // Far matrix multiplication for block Zmm: FFT pair 1
    clearData<<< N / 256 + 1, 256 >>>(geo0_data[1], N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo0_data[1], d_x3[1], hori_row, hori_col, num_pts, d_hori_x[1], d_hori_f[1]);
    cufftExecC2C(plan[1], (cufftComplex*)geo0_data[1], (cufftComplex*)geo0_data[1], CUFFT_FORWARD);
    convolve<<< N / 256 + 1, 256 >>>(g0_data[1], geo0_data[1], N);
    cufftExecC2C(plan[1], (cufftComplex*)geo0_data[1], (cufftComplex*)geo0_data[1], CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3[1], geo0_data[1], hori_num, num_pts, N, d_hori_x[1], d_hori_b[1], true);
    
    // Far matrix multiplication for block Zmm: FFT pair 2
    clearData<<< N / 256 + 1, 256 >>>(geo0_data[1], N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo0_data[1], d_x4[1], vert_row, vert_col, num_pts, d_vert_y[1], d_vert_f[1]);
    cufftExecC2C(plan[1], (cufftComplex*)geo0_data[1], (cufftComplex*)geo0_data[1], CUFFT_FORWARD);
    convolve<<< N / 256 + 1, 256 >>>(g0_data[1], geo0_data[1], N);
    cufftExecC2C(plan[1], (cufftComplex*)geo0_data[1], (cufftComplex*)geo0_data[1], CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4[1], geo0_data[1], vert_num, num_pts, N, d_vert_y[1], d_vert_b[1], true);

    // Far matrix multiplication for block Zmm: FFT pair 3
    clearData<<< N / 256 + 1, 256 >>>(geo0_data[1], N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo0_data[1], d_x3[1], hori_row, hori_col, num_pts, d_hori_z[1], d_hori_f[1]);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo0_data[1], d_x4[1], vert_row, vert_col, num_pts, d_vert_z[1], d_vert_f[1]);
    cufftExecC2C(plan[1], (cufftComplex*)geo0_data[1], (cufftComplex*)geo0_data[1], CUFFT_FORWARD);
    convolve<<< N / 256 + 1, 256 >>>(g0_data[1], geo0_data[1], N);
    cufftExecC2C(plan[1], (cufftComplex*)geo0_data[1], (cufftComplex*)geo0_data[1], CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3[1], geo0_data[1], hori_num, num_pts, N, d_hori_z[1], d_hori_b[1], true);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4[1], geo0_data[1], vert_num, num_pts, N, d_vert_z[1], d_vert_b[1], true);

    // Far matrix multiplication for block Zmm: FFT pair 4
    clearData<<< N / 256 + 1, 256 >>>(geo0_data[1], N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo0_data[1], d_x3[1], hori_row, hori_col, num_pts, d_hori_d[1], d_hori_f[1]);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo0_data[1], d_x4[1], vert_row, vert_col, num_pts, d_vert_d[1], d_vert_f[1]);
    cufftExecC2C(plan[1], (cufftComplex*)geo0_data[1], (cufftComplex*)geo0_data[1], CUFFT_FORWARD);
    convolve<<< N / 256 + 1, 256 >>>(g1_data[1], geo0_data[1], N);
    cufftExecC2C(plan[1], (cufftComplex*)geo0_data[1], (cufftComplex*)geo0_data[1], CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3[1], geo0_data[1], hori_num, num_pts, N, d_hori_d[1], d_hori_b[1], false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4[1], geo0_data[1], vert_num, num_pts, N, d_vert_d[1], d_vert_b[1], false);
}

/// @brief Matrix-vector multiplication computations done on GPU 2
void MVProd4::gpu2() {
    // Initialization: clearing all vectors to become zero-valued
    cudaSetDevice(devNumber[2]);
    cusparseDnVecSetValues(x1[2], d_x1[2]);
    cusparseDnVecSetValues(x2[2], d_x2[2]);
    cusparseDnVecSetValues(x3[2], d_x3[2]);
    cusparseDnVecSetValues(x4[2], d_x4[2]);
    clearData<<< hori_num / 256 + 1, 256 >>>(d_y1[2], hori_num);
    clearData<<< vert_num / 256 + 1, 256 >>>(d_y2[2], vert_num);
    clearData<<< hori_num / 256 + 1, 256 >>>(d_y3[2], hori_num);
    clearData<<< vert_num / 256 + 1, 256 >>>(d_y4[2], vert_num);
    cusparseDnVecSetValues(y1[2], d_y1[2]);
    cusparseDnVecSetValues(y2[2], d_y2[2]);
    cusparseDnVecSetValues(y3[2], d_y3[2]);
    cusparseDnVecSetValues(y4[2], d_y4[2]);

    // Sparse near matrix multiplication
    cusparseSpMV(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aee[2], x2[2], &beta, y1[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bee[2], x2[2], &beta, y1[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[2], x4[2], &beta, y1[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[2], x4[2], &beta, y1[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aee[2], x1[2], &beta, y2[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bee[2], x1[2], &beta, y2[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[2], x3[2], &beta, y2[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[2], x3[2], &beta, y2[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[2], x2[2], &beta, y3[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[2], x2[2], &beta, y3[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Amm[2], x4[2], &beta, y3[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bmm[2], x4[2], &beta, y3[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[2], x1[2], &beta, y4[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[2], x1[2], &beta, y4[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Amm[2], x3[2], &beta, y4[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);
    cusparseSpMV(handle2, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bmm[2], x3[2], &beta, y4[2], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[2]);

    // Far matrix multiplication for block Zem: FFT group 1
    clearData<<< N / 256 + 1, 256 >>>(geo1_data[0], N);
    clearData<<< N / 256 + 1, 256 >>>(geo2_data[0], N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo2_data[0], d_x4[2], vert_row, vert_col, num_pts, d_vert_y[2], d_vert_f[2]);
    cufftExecC2C(plan[2], (cufftComplex*)geo2_data[0], (cufftComplex*)geo2_data[0], CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[0], g4_data[0], geo2_data[0], N, false);
    clearData<<< N / 256 + 1, 256 >>>(geo2_data[0], N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo2_data[0], d_x3[2], hori_row, hori_col, num_pts, d_hori_z[2], d_hori_f[2]);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo2_data[0], d_x4[2], vert_row, vert_col, num_pts, d_vert_z[2], d_vert_f[2]);
    cufftExecC2C(plan[2], (cufftComplex*)geo2_data[0], (cufftComplex*)geo2_data[0], CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[0], g3_data[0], geo2_data[0], N, true);
    cufftExecC2C(plan[2], (cufftComplex*)geo1_data[0], (cufftComplex*)geo1_data[0], CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1[2], geo1_data[0], hori_num, num_pts, N, d_hori_x[2], d_hori_b[2], false);

    // Far matrix multiplication for block Zem: FFT group 2
    clearData<<< N / 256 + 1, 256 >>>(geo1_data[0], N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[0], g2_data[0], geo2_data[0], N, false);
    clearData<<< N / 256 + 1, 256 >>>(geo2_data[0], N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo2_data[0], d_x3[2], hori_row, hori_col, num_pts, d_hori_x[2], d_hori_f[2]);
    cufftExecC2C(plan[2], (cufftComplex*)geo2_data[0], (cufftComplex*)geo2_data[0], CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[0], g4_data[0], geo2_data[0], N, true);
    cufftExecC2C(plan[2], (cufftComplex*)geo1_data[0], (cufftComplex*)geo1_data[0], CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2[2], geo1_data[0], vert_num, num_pts, N, d_vert_y[2], d_vert_b[2], false);

    // Far matrix multiplication for block Zem: FFT group 3
    clearData<<< N / 256 + 1, 256 >>>(geo1_data[0], N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[0], g3_data[0], geo2_data[0], N, false);
    clearData<<< N / 256 + 1, 256 >>>(geo2_data[0], N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo2_data[0], d_x4[2], vert_row, vert_col, num_pts, d_vert_y[2], d_vert_f[2]);
    cufftExecC2C(plan[2], (cufftComplex*)geo2_data[0], (cufftComplex*)geo2_data[0], CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[0], g2_data[0], geo2_data[0], N, true);
    cufftExecC2C(plan[2], (cufftComplex*)geo1_data[0], (cufftComplex*)geo1_data[0], CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1[2], geo1_data[0], hori_num, num_pts, N, d_hori_z[2], d_hori_b[2], false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2[2], geo1_data[0], vert_num, num_pts, N, d_vert_z[2], d_vert_b[2], false);
}

/// @brief Matrix-vector multiplication computations done on GPU 3
void MVProd4::gpu3() {
    // Initialization: clearing all vectors to become zero-valued
    cudaSetDevice(devNumber[3]);
    cusparseDnVecSetValues(x1[3], d_x1[3]);
    cusparseDnVecSetValues(x2[3], d_x2[3]);
    cusparseDnVecSetValues(x3[3], d_x3[3]);
    cusparseDnVecSetValues(x4[3], d_x4[3]);
    clearData<<< hori_num / 256 + 1, 256 >>>(d_y1[3], hori_num);
    clearData<<< vert_num / 256 + 1, 256 >>>(d_y2[3], vert_num);
    clearData<<< hori_num / 256 + 1, 256 >>>(d_y3[3], hori_num);
    clearData<<< vert_num / 256 + 1, 256 >>>(d_y4[3], vert_num);
    cusparseDnVecSetValues(y1[3], d_y1[3]);
    cusparseDnVecSetValues(y2[3], d_y2[3]);
    cusparseDnVecSetValues(y3[3], d_y3[3]);
    cusparseDnVecSetValues(y4[3], d_y4[3]);

    // Sparse near matrix multiplication
    cusparseSpMV(handle3, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aee[3], x2[3], &beta, y2[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aee[3], x2[3], &beta, y2[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bee[3], x2[3], &beta, y2[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bee[3], x2[3], &beta, y2[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[3], x4[3], &beta, y2[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[3], x4[3], &beta, y2[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[3], x4[3], &beta, y2[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[3], x4[3], &beta, y2[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[3], x2[3], &beta, y4[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[3], x2[3], &beta, y4[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[3], x2[3], &beta, y4[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[3], x2[3], &beta, y4[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Amm[3], x4[3], &beta, y4[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Amm[3], x4[3], &beta, y4[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bmm[3], x4[3], &beta, y4[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);
    cusparseSpMV(handle3, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bmm[3], x4[3], &beta, y4[3], CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work[3]);

    // Far matrix multiplication for block Zme: FFT group 1
    clearData<<< N / 256 + 1, 256 >>>(geo1_data[1], N);
    clearData<<< N / 256 + 1, 256 >>>(geo2_data[1], N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo2_data[1], d_x2[3], vert_row, vert_col, num_pts, d_vert_y[3], d_vert_f[3]);
    cufftExecC2C(plan[3], (cufftComplex*)geo2_data[1], (cufftComplex*)geo2_data[1], CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[1], g4_data[1], geo2_data[1], N, false);
    clearData<<< N / 256 + 1, 256 >>>(geo2_data[1], N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo2_data[1], d_x1[3], hori_row, hori_col, num_pts, d_hori_z[3], d_hori_f[3]);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo2_data[1], d_x2[3], vert_row, vert_col, num_pts, d_vert_z[3], d_vert_f[3]);
    cufftExecC2C(plan[3], (cufftComplex*)geo2_data[1], (cufftComplex*)geo2_data[1], CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[1], g3_data[1], geo2_data[1], N, true);
    cufftExecC2C(plan[3], (cufftComplex*)geo1_data[1], (cufftComplex*)geo1_data[1], CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3[3], geo1_data[1], hori_num, num_pts, N, d_hori_x[3], d_hori_b[3], false);

    // Far matrix multiplication for block Zme: FFT group 2
    clearData<<< N / 256 + 1, 256 >>>(geo1_data[1], N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[1], g2_data[1], geo2_data[1], N, false);
    clearData<<< N / 256 + 1, 256 >>>(geo2_data[1], N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo2_data[1], d_x1[3], hori_row, hori_col, num_pts, d_hori_x[3], d_hori_f[3]);
    cufftExecC2C(plan[3], (cufftComplex*)geo2_data[1], (cufftComplex*)geo2_data[1], CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[1], g4_data[1], geo2_data[1], N, true);
    cufftExecC2C(plan[3], (cufftComplex*)geo1_data[1], (cufftComplex*)geo1_data[1], CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4[3], geo1_data[1], vert_num, num_pts, N, d_vert_y[3], d_vert_b[3], false);

    // Far matrix multiplication for block Zme: FFT group 3
    clearData<<< N / 256 + 1, 256 >>>(geo1_data[1], N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[1], g3_data[1], geo2_data[1], N, false);
    clearData<<< N / 256 + 1, 256 >>>(geo2_data[1], N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo2_data[1], d_x2[3], vert_row, vert_col, num_pts, d_vert_y[3], d_vert_f[3]);
    cufftExecC2C(plan[3], (cufftComplex*)geo2_data[1], (cufftComplex*)geo2_data[1], CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo1_data[1], g2_data[1], geo2_data[1], N, true);
    cufftExecC2C(plan[3], (cufftComplex*)geo1_data[1], (cufftComplex*)geo1_data[1], CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3[3], geo1_data[1], hori_num, num_pts, N, d_hori_z[3], d_hori_b[3], false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4[3], geo1_data[1], vert_num, num_pts, N, d_vert_z[3], d_vert_b[3], false);
}

/// @brief Destroy the FFT computation plans and deallocated associated memory
void MVProd4::cleanAll() {
    cusparseDestroy(handle0);
    cusparseDestroy(handle1);
    cusparseDestroy(handle2);
    cusparseDestroy(handle3);
    for (int dev = 0; dev < 4; dev++) {
        cudaSetDevice(devNumber[dev]);
        cudaFree(d_Arows[dev]);
        cudaFree(d_Acols[dev]);
        cudaFree(d_Brows[dev]);
        cudaFree(d_Bcols[dev]);
        cudaFree(d_Aee[dev]);
        cudaFree(d_Aem[dev]);
        cudaFree(d_Amm[dev]);
        cudaFree(d_Bee[dev]);
        cudaFree(d_Bem[dev]);
        cudaFree(d_Bmm[dev]);
        cudaFree(d_x1[dev]);
        cudaFree(d_x2[dev]);
        cudaFree(d_x3[dev]);
        cudaFree(d_x4[dev]);
        cudaFree(d_y1[dev]);
        cudaFree(d_y2[dev]);
        cudaFree(d_y3[dev]);
        cudaFree(d_y4[dev]);
        cudaFree(d_base1[dev]);
        cudaFree(d_base2[dev]);
        cudaFree(zvals[dev]);
        cudaFree(xvech[dev]);
        cudaFree(wvech[dev]);
        cudaFree(xvecl[dev]);
        cudaFree(wvecl[dev]);
        cudaFree(d_hori_x[dev]);
        cudaFree(d_hori_z[dev]);
        cudaFree(d_hori_d[dev]);
        cudaFree(d_hori_f[dev]);
        cudaFree(d_hori_b[dev]);
        cudaFree(d_vert_y[dev]);
        cudaFree(d_vert_z[dev]);
        cudaFree(d_vert_d[dev]);
        cudaFree(d_vert_f[dev]);
        cudaFree(d_vert_b[dev]);
        cudaFree(d_work[dev]);
        cusparseDestroySpMat(Aee[dev]);
        cusparseDestroySpMat(Aem[dev]);
        cusparseDestroySpMat(Amm[dev]);
        cusparseDestroySpMat(Bee[dev]);
        cusparseDestroySpMat(Bem[dev]);
        cusparseDestroySpMat(Bmm[dev]);
        cusparseDestroyDnVec(x1[dev]);
        cusparseDestroyDnVec(x2[dev]);
        cusparseDestroyDnVec(x3[dev]);
        cusparseDestroyDnVec(x4[dev]);
        cusparseDestroyDnVec(y1[dev]);
        cusparseDestroyDnVec(y2[dev]);
        cusparseDestroyDnVec(y3[dev]);
        cusparseDestroyDnVec(y4[dev]);
        cufftDestroy(plan[dev]);
    }
    for (int count = 0; count < 2; count++) {
        cudaSetDevice(devNumber[count]);
        cudaFree(g0_data[count]);
        cudaFree(g1_data[count]);
        cudaFree(geo0_data[count]);
    }
    for (int count = 0; count < 2; count++) {
        cudaSetDevice(devNumber[count + 2]);
        cudaFree(g2_data[count]);
        cudaFree(g3_data[count]);
        cudaFree(g4_data[count]);
        cudaFree(geo1_data[count]);
        cudaFree(geo2_data[count]);
    }
}