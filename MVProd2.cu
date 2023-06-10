/* Speed-oriented, 1-GPU implementation for computing matrix-vector products in BEM.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "MVProd2.h"

/// @brief Speed-oriented, 1-GPU implementation for computing matrix-vector products in BEM
/// @param est: an Estimate object with information on the simulated surface
/// @param singular: a Singular object that computes the matrix elements with singularties in the underlying integrals
/// @param grid: a Grid object with information on the 3D grid of point sources
/// @param isDielectric: a boolean value that indicates whether the surface material is dielectric
/// @param ind: index of the GPU used
MVProd2::MVProd2(Estimate* est, Singular* singular, Grid* grid, bool isDielectric, int ind): MVProd(est, singular, grid) {
    this->isDielectric = isDielectric;
    alpha.x = 1.0f;
    alpha.y = 0.0f;
    beta.x = 1.0f;
    beta.y = 0.0f;
    h_y1 = VectorXcf::Zero(hori_num);
    h_y2 = VectorXcf::Zero(vert_num);
    h_y3 = VectorXcf::Zero(hori_num);
    h_y4 = VectorXcf::Zero(vert_num);
    cudaSetDevice(ind);
    cusparseCreate(&handle);
    initializeNear();
    initializeFar();
}

/// @brief Allocate sparse correction matrices and the data structures holding the input, output vectors on the GPU
void MVProd2::initializeNear() {
    // Sparsity patterns are stored in binary files and transferred to the GPU
    for (int i = 0; i < 4; i++) {
        nnzA[i] = est->A[i].rows();
        cudaMalloc(&d_Arows[i], nnzA[i] * sizeof(int));
        cudaMemcpy(d_Arows[i], (int*)((est->A[i].block(0, 0, nnzA[i], 1)).data()), nnzA[i] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_Acols[i], nnzA[i] * sizeof(int));
        cudaMemcpy(d_Acols[i], (int*)((est->A[i].block(0, 1, nnzA[i], 1)).data()), nnzA[i] * sizeof(int), cudaMemcpyHostToDevice);
        nnzB[i] = est->B[i].rows();
        cudaMalloc(&d_Brows[i], nnzB[i] * sizeof(int));
        cudaMemcpy(d_Brows[i], (int*)((est->B[i].block(0, 0, nnzB[i], 1)).data()), nnzB[i] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_Bcols[i], nnzB[i] * sizeof(int));
        cudaMemcpy(d_Bcols[i], (int*)((est->B[i].block(0, 1, nnzB[i], 1)).data()), nnzB[i] * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    // Create sparse matrices and dense vectors on GPUs
    Vector4i numrows(hori_num, hori_num, hori_num, vert_num);
    Vector4i numcols(hori_num, vert_num, vert_num, vert_num);
    for (int i = 0; i < 4; i++) {
        cudaMalloc(&d_Aee[i], nnzA[i] * sizeof(cuComplex));
        cudaMalloc(&d_Aem[i], nnzA[i] * sizeof(cuComplex));
        cudaMalloc(&d_Amm[i], nnzA[i] * sizeof(cuComplex));
        cudaMalloc(&d_Bee[i], nnzB[i] * sizeof(cuComplex));
        cudaMalloc(&d_Bem[i], nnzB[i] * sizeof(cuComplex));
        cudaMalloc(&d_Bmm[i], nnzB[i] * sizeof(cuComplex));
        cusparseCreateCoo(&Aee[i], numrows(i), numcols(i), nnzA[i], d_Arows[i], d_Acols[i], d_Aee[i], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cusparseCreateCoo(&Aem[i], numrows(i), numcols(i), nnzA[i], d_Arows[i], d_Acols[i], d_Aem[i], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cusparseCreateCoo(&Amm[i], numrows(i), numcols(i), nnzA[i], d_Arows[i], d_Acols[i], d_Amm[i], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cusparseCreateCoo(&Bee[i], numrows(i), numcols(i), nnzB[i], d_Brows[i], d_Bcols[i], d_Bee[i], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cusparseCreateCoo(&Bem[i], numrows(i), numcols(i), nnzB[i], d_Brows[i], d_Bcols[i], d_Bem[i], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
        cusparseCreateCoo(&Bmm[i], numrows(i), numcols(i), nnzB[i], d_Brows[i], d_Bcols[i], d_Bmm[i], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
    }
    cudaMalloc(&d_x1, hori_num * sizeof(cuComplex));
    cudaMalloc(&d_x2, vert_num * sizeof(cuComplex));
    cudaMalloc(&d_x3, hori_num * sizeof(cuComplex));
    cudaMalloc(&d_x4, vert_num * sizeof(cuComplex));
    cudaMalloc(&d_y1, hori_num * sizeof(cuComplex));
    cudaMalloc(&d_y2, vert_num * sizeof(cuComplex));
    cudaMalloc(&d_y3, hori_num * sizeof(cuComplex));
    cudaMalloc(&d_y4, vert_num * sizeof(cuComplex));
    cusparseCreateDnVec(&x1, hori_num, d_x1, CUDA_C_32F);
    cusparseCreateDnVec(&x2, vert_num, d_x2, CUDA_C_32F);
    cusparseCreateDnVec(&x3, hori_num, d_x3, CUDA_C_32F);
    cusparseCreateDnVec(&x4, vert_num, d_x4, CUDA_C_32F);
    cusparseCreateDnVec(&y1, hori_num, d_y1, CUDA_C_32F);
    cusparseCreateDnVec(&y2, vert_num, d_y2, CUDA_C_32F);
    cusparseCreateDnVec(&y3, hori_num, d_y3, CUDA_C_32F);
    cusparseCreateDnVec(&y4, vert_num, d_y4, CUDA_C_32F);
    cudaMalloc(&d_work, 0);
    cudaDeviceSynchronize();

    // Transfer matrix element data that require singularity removal computation to GPUs
    for (int i = 0; i < 4; i++) {
        VectorXcf matrix_data;
        singular->computeQuarter(i);
        matrix_data = singular->quarter.block(0, 0, nnzA[i], 1);
        cudaMalloc(&d_base1[i], nnzA[i] * sizeof(cuComplex));
        cudaMemcpy(d_base1[i], (fcomp*)(matrix_data.data()), nnzA[i] * sizeof(cuComplex), cudaMemcpyHostToDevice);
        matrix_data = singular->quarter.block(0, 1, nnzA[i], 1);
        cudaMalloc(&d_base2[i], nnzA[i] * sizeof(cuComplex));
        cudaMemcpy(d_base2[i], (fcomp*)(matrix_data.data()), nnzA[i] * sizeof(cuComplex), cudaMemcpyHostToDevice);
        matrix_data = singular->quarter.block(0, 2, nnzA[i], 1);
        cudaMemcpy(d_Aem[i], (fcomp*)(matrix_data.data()), nnzA[i] * sizeof(cuComplex), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    // Allocate height field data and necessary quadrature points / weights
    MatrixXf zdata = est->zvals.cast<float>();
    VectorXf xhigh = quadrature_points.block(3, 0, 1, 4).transpose().cast<float>();
    VectorXf whigh = quadrature_weights.block(3, 0, 1, 4).transpose().cast<float>();
    VectorXf xlow = quadrature_points.block(1, 0, 1, 2).transpose().cast<float>();
    VectorXf wlow = quadrature_weights.block(1, 0, 1, 2).transpose().cast<float>();
    cudaMalloc(&zvals, (Nx + 1) * (Ny + 1) * sizeof(float));
    cudaMemcpy(zvals, (float*)(zdata.data()), (Nx + 1) * (Ny + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&xvech, 4 * sizeof(float));
    cudaMemcpy(xvech, (float*)(xhigh.data()), 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&wvech, 4 * sizeof(float));
    cudaMemcpy(wvech, (float*)(whigh.data()), 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&xvecl, 2 * sizeof(float));
    cudaMemcpy(xvecl, (float*)(xlow.data()), 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&wvecl, 2 * sizeof(float));
    cudaMemcpy(wvecl, (float*)(wlow.data()), 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

/// @brief Allocate memory for point source approximation coefficients on the GPU; create FFT plans for computation
void MVProd2::initializeFar() {
    // Allocate point source approximation coefficients and indices
    cudaMalloc(&d_hori_x, num_pts * hori_num * sizeof(Tfcomp));
    cudaMalloc(&d_hori_z, num_pts * hori_num * sizeof(Tfcomp));
    cudaMalloc(&d_hori_d, num_pts * hori_num * sizeof(Tfcomp));
    cudaMalloc(&d_hori_f, hori_row * hori_col * sizeof(int));
    cudaMalloc(&d_hori_b, num_pts * hori_num * sizeof(int));
    cudaMalloc(&d_vert_y, num_pts * vert_num * sizeof(Tfcomp));
    cudaMalloc(&d_vert_z, num_pts * vert_num * sizeof(Tfcomp));
    cudaMalloc(&d_vert_d, num_pts * vert_num * sizeof(Tfcomp));
    cudaMalloc(&d_vert_f, vert_row * vert_col * sizeof(int));
    cudaMalloc(&d_vert_b, num_pts * vert_num * sizeof(int));
    if (isDielectric) {
        cudaMalloc(&d_hori_X, num_pts * hori_num * sizeof(Tfcomp));
        cudaMalloc(&d_hori_Z, num_pts * hori_num * sizeof(Tfcomp));
        cudaMalloc(&d_hori_D, num_pts * hori_num * sizeof(Tfcomp));
        cudaMalloc(&d_vert_Y, num_pts * vert_num * sizeof(Tfcomp));
        cudaMalloc(&d_vert_Z, num_pts * vert_num * sizeof(Tfcomp));
        cudaMalloc(&d_vert_D, num_pts * vert_num * sizeof(Tfcomp));
    }
    cudaDeviceSynchronize();

    // Allocate far matrix multiplication data structures
    cudaMalloc(&g0_data, N * sizeof(Tfcomp));
    cudaMalloc(&g1_data, N * sizeof(Tfcomp));
    cudaMalloc(&g2_data, N * sizeof(Tfcomp));
    cudaMalloc(&g3_data, N * sizeof(Tfcomp));
    cudaMalloc(&geo0_data, N * sizeof(Tfcomp));
    cudaMalloc(&geo1_data, N * sizeof(Tfcomp));
    cufftCreate(&plan);
    size_t* worksize = (size_t *)malloc(sizeof(size_t));
    cufftMakePlan3d(plan, totalX, totalY, totalZ, CUFFT_C2C, worksize);
    if (isDielectric) {
        cudaMalloc(&g4_data, N * sizeof(Tfcomp));
        cudaMalloc(&g5_data, N * sizeof(Tfcomp));
        cudaMalloc(&g6_data, N * sizeof(Tfcomp));
        cudaMalloc(&g7_data, N * sizeof(Tfcomp));
    }
    cudaDeviceSynchronize();
}

/// @brief Perform initializations for computing matrix-vector products in a simulation with given media parameters and wavelengths
/// @brief Compute the point source approximations, the sparse correction matrices, and Fourier transform of Green's functions
/// @param eta1: index of refraction of the medium where the light is incident from, usually 1.0 (air)
/// @param eta2: index of refraction of the surface material (could be complex-valued)
/// @param lambda: the currently simulated wavelength
void MVProd2::setParameters(double eta1, dcomp eta2, double lambda) {
    // Initialize parameters
    double omega = c / lambda * 2 * M_PI;
    dcomp eps1 = 1 / (mu * c * c) * eta1 * eta1;
    dcomp eps2 = 1 / (mu * c * c) * eta2 * eta2;
    e1 = Tfcomp((float)eta1, 0.0f);
    e2 = Tfcomp((float)real(eta2), (float)imag(eta2));
    const1 = Tfcomp((float)real(cuDB * omega * mu), (float)imag(cuDB * omega * mu));
    const21 = Tfcomp((float)real(cuDB / (omega * eps1)), (float)imag(cuDB / (omega * eps1)));
    const22 = Tfcomp((float)real(cuDB / (omega * eps2)), (float)imag(cuDB / (omega * eps2)));
    Tfcomp k1 = (float)(2 * M_PI / lambda) * e1;
    Tfcomp k2 = (float)(2 * M_PI / lambda) * e2;
    Tfcomp c1 = 2.0f * const1;
    Tfcomp c2 = const21 + const22;
    Tfcomp c3 = -(e1 * e1 + e2 * e2) * const1;
    Tfcomp c4 = -e1 * e1 * const21 - e2 * e2 * const22;
    for (int i = 0; i < 4; i++)
        updateEEMM<<< nnzA[i] / 256 + 1, 256 >>>(nnzA[i], d_Aee[i], d_Amm[i], d_base1[i], d_base2[i], c1, c2, c3, c4);

    // Compute point source approximation coefficients
    grid->computeCoefficients(eta1, eta2, lambda);
    cudaMemcpy(d_hori_f, (int*)(grid->hori_f.data()), hori_row * hori_col * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hori_b, (int*)(grid->hori_b.data()), num_pts * hori_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vert_f, (int*)(grid->vert_f.data()), vert_row * vert_col * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vert_b, (int*)(grid->vert_b.data()), num_pts * vert_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hori_x, (fcomp*)(grid->hori_x.data()), num_pts * hori_num * sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hori_z, (fcomp*)(grid->hori_z.data()), num_pts * hori_num * sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hori_d, (fcomp*)(grid->hori_d.data()), num_pts * hori_num * sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vert_y, (fcomp*)(grid->vert_y.data()), num_pts * vert_num * sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vert_z, (fcomp*)(grid->vert_z.data()), num_pts * vert_num * sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vert_d, (fcomp*)(grid->vert_d.data()), num_pts * vert_num * sizeof(fcomp), cudaMemcpyHostToDevice);
    if (isDielectric) {
        cudaMemcpy(d_hori_X, (fcomp*)(grid->hori_X.data()), num_pts * hori_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hori_Z, (fcomp*)(grid->hori_Z.data()), num_pts * hori_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hori_D, (fcomp*)(grid->hori_D.data()), num_pts * hori_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vert_Y, (fcomp*)(grid->vert_Y.data()), num_pts * vert_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vert_Z, (fcomp*)(grid->vert_Z.data()), num_pts * vert_num * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vert_D, (fcomp*)(grid->vert_D.data()), num_pts * vert_num * sizeof(fcomp), cudaMemcpyHostToDevice);
    }

    // Compute near matrix elements
    clearData<<< nnzB[0] / 256 + 1, 256 >>>(d_Bee[0], nnzB[0]);
    clearData<<< nnzB[0] / 256 + 1, 256 >>>(d_Bem[0], nnzB[0]);
    clearData<<< nnzB[0] / 256 + 1, 256 >>>(d_Bmm[0], nnzB[0]);
    individual<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[0], d_Bcols[0], 0, 0, 0, 0, 1, -1, -1, d_Bee[0], d_Bmm[0], d_Bem[0]);
    individual<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[0], d_Bcols[0], 0, 0, 1, 0, 1, -1, 1, d_Bee[0], d_Bmm[0], d_Bem[0]);
    individual<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[0], d_Bcols[0], 1, 0, 0, 0, 1, 1, -1, d_Bee[0], d_Bmm[0], d_Bem[0]);
    individual<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[0], d_Bcols[0], 1, 0, 1, 0, 1, 1, 1, d_Bee[0], d_Bmm[0], d_Bem[0]);
    correctionHH<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], dx, dy, dz, k1, const1, const21, totalY, totalZ, d_Brows[0], d_Bcols[0], num_pts, d_hori_b, d_hori_x, d_hori_z, d_hori_d, d_Bee[0], d_Bem[0]);
    for (int i = 1; i <= 2; i++) {
        clearData<<< nnzB[i] / 256 + 1, 256 >>>(d_Bee[i], nnzB[i]);
        clearData<<< nnzB[i] / 256 + 1, 256 >>>(d_Bem[i], nnzB[i]);
        clearData<<< nnzB[i] / 256 + 1, 256 >>>(d_Bmm[i], nnzB[i]);
        individual<<< nnzB[i] / 256 + 1, 256 >>>(nnzB[i], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[i], d_Bcols[i], 0, 0, 0, 0, 2, -1, -1, d_Bee[i], d_Bmm[i], d_Bem[i]);
        individual<<< nnzB[i] / 256 + 1, 256 >>>(nnzB[i], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[i], d_Bcols[i], 0, 0, 0, 1, 2, -1, 1, d_Bee[i], d_Bmm[i], d_Bem[i]);
        individual<<< nnzB[i] / 256 + 1, 256 >>>(nnzB[i], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[i], d_Bcols[i], 1, 0, 0, 0, 2, 1, -1, d_Bee[i], d_Bmm[i], d_Bem[i]);
        individual<<< nnzB[i] / 256 + 1, 256 >>>(nnzB[i], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[i], d_Bcols[i], 1, 0, 0, 1, 2, 1, 1, d_Bee[i], d_Bmm[i], d_Bem[i]);
        correctionHV<<< nnzB[i] / 256 + 1, 256 >>>(nnzB[i], dx, dy, dz, k1, const1, const21, totalY, totalZ, d_Brows[i], d_Bcols[i], num_pts, d_hori_b, d_vert_b, d_hori_x, d_hori_z, d_hori_d, d_vert_y, d_vert_z, d_vert_d, d_Bee[i], d_Bem[i]);
    }
    clearData<<< nnzB[3] / 256 + 1, 256 >>>(d_Bee[3], nnzB[3]);
    clearData<<< nnzB[3] / 256 + 1, 256 >>>(d_Bem[3], nnzB[3]);
    clearData<<< nnzB[3] / 256 + 1, 256 >>>(d_Bmm[3], nnzB[3]);
    individual<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[3], d_Bcols[3], 0, 0, 0, 0, 3, -1, -1, d_Bee[3], d_Bmm[3], d_Bem[3]);
    individual<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[3], d_Bcols[3], 0, 0, 0, 1, 3, -1, 1, d_Bee[3], d_Bmm[3], d_Bem[3]);
    individual<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[3], d_Bcols[3], 0, 1, 0, 0, 3, 1, -1, d_Bee[3], d_Bmm[3], d_Bem[3]);
    individual<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], Nx, Ny, zvals, d, k1, k2, const1, const21, const22, xvech, wvech, xvecl, wvecl, d_Brows[3], d_Bcols[3], 0, 1, 0, 1, 3, 1, 1, d_Bee[3], d_Bmm[3], d_Bem[3]);
    correctionVV<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], dx, dy, dz, k1, const1, const21, totalY, totalZ, d_Brows[3], d_Bcols[3], num_pts, d_vert_b, d_vert_y, d_vert_z, d_vert_d, d_Bee[3], d_Bem[3]);
    
    // Correct near matrix elements
    if (isDielectric) {
        correctionHH<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], dx, dy, dz, k2, const1, const22, totalY, totalZ, d_Brows[0], d_Bcols[0], num_pts, d_hori_b, d_hori_X, d_hori_Z, d_hori_D, d_Bmm[0], d_Bem[0]);
        correctionHV<<< nnzB[1] / 256 + 1, 256 >>>(nnzB[1], dx, dy, dz, k2, const1, const22, totalY, totalZ, d_Brows[1], d_Bcols[1], num_pts, d_hori_b, d_vert_b, d_hori_X, d_hori_Z, d_hori_D, d_vert_Y, d_vert_Z, d_vert_D, d_Bmm[1], d_Bem[1]);
        correctionHV<<< nnzB[2] / 256 + 1, 256 >>>(nnzB[2], dx, dy, dz, k2, const1, const22, totalY, totalZ, d_Brows[2], d_Bcols[2], num_pts, d_hori_b, d_vert_b, d_hori_X, d_hori_Z, d_hori_D, d_vert_Y, d_vert_Z, d_vert_D, d_Bmm[2], d_Bem[2]);
        correctionVV<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], dx, dy, dz, k2, const1, const22, totalY, totalZ, d_Brows[3], d_Bcols[3], num_pts, d_vert_b, d_vert_Y, d_vert_Z, d_vert_D, d_Bmm[3], d_Bem[3]);
    }
    postProcess<<< nnzB[0] / 256 + 1, 256 >>>(nnzB[0], true, eta0FL, eta1, eta2, d_Brows[0], d_Bcols[0], d_Bee[0], d_Bmm[0], d_Bem[0]);
    postProcess<<< nnzB[1] / 256 + 1, 256 >>>(nnzB[1], false, eta0FL, eta1, eta2, d_Brows[1], d_Bcols[1], d_Bee[1], d_Bmm[1], d_Bem[1]);
    postProcess<<< nnzB[2] / 256 + 1, 256 >>>(nnzB[2], false, eta0FL, eta1, eta2, d_Brows[2], d_Bcols[2], d_Bee[2], d_Bmm[2], d_Bem[2]);
    postProcess<<< nnzB[3] / 256 + 1, 256 >>>(nnzB[3], true, eta0FL, eta1, eta2, d_Brows[3], d_Bcols[3], d_Bee[3], d_Bmm[3], d_Bem[3]);
    
    // Compute Green's function values and perform Fourier transforms
    computeGreens<<< N / 256 + 1, 256 >>>(g0_data, dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 7);
    cufftExecC2C(plan, (cufftComplex*)g0_data, (cufftComplex*)g0_data, CUFFT_FORWARD);
    computeGreens<<< N / 256 + 1, 256 >>>(g1_data, dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 4);
    cufftExecC2C(plan, (cufftComplex*)g1_data, (cufftComplex*)g1_data, CUFFT_FORWARD);
    computeGreens<<< N / 256 + 1, 256 >>>(g2_data, dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 5);
    cufftExecC2C(plan, (cufftComplex*)g2_data, (cufftComplex*)g2_data, CUFFT_FORWARD);
    computeGreens<<< N / 256 + 1, 256 >>>(g3_data, dx, dy, dz, eta0FL, e1, k1, const1, const21, N, totalX, totalY, totalZ, 6);
    cufftExecC2C(plan, (cufftComplex*)g3_data, (cufftComplex*)g3_data, CUFFT_FORWARD);
    if (isDielectric) {
        computeGreens<<< N / 256 + 1, 256 >>>(g4_data, dx, dy, dz, eta0FL, e2, k2, const1, const22, N, totalX, totalY, totalZ, 7);
        cufftExecC2C(plan, (cufftComplex*)g4_data, (cufftComplex*)g4_data, CUFFT_FORWARD);
        computeGreens<<< N / 256 + 1, 256 >>>(g5_data, dx, dy, dz, eta0FL, e2, k2, const1, const22, N, totalX, totalY, totalZ, 4);
        cufftExecC2C(plan, (cufftComplex*)g5_data, (cufftComplex*)g5_data, CUFFT_FORWARD);
        computeGreens<<< N / 256 + 1, 256 >>>(g6_data, dx, dy, dz, eta0FL, e2, k2, const1, const22, N, totalX, totalY, totalZ, 5);
        cufftExecC2C(plan, (cufftComplex*)g6_data, (cufftComplex*)g6_data, CUFFT_FORWARD);
        computeGreens<<< N / 256 + 1, 256 >>>(g7_data, dx, dy, dz, eta0FL, e2, k2, const1, const22, N, totalX, totalY, totalZ, 6);
        cufftExecC2C(plan, (cufftComplex*)g7_data, (cufftComplex*)g7_data, CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();
}

/// @brief Perform matrix-vector multiplication using the BEM matrix
/// @param x: the input vector
/// @return The product vector
VectorXcf MVProd2::multiply(VectorXcf x) {
    VectorXcf h_x1 = x.block(0, 0, hori_num, 1);
    VectorXcf h_x2 = x.block(hori_num, 0, vert_num, 1);
    VectorXcf h_x3 = x.block(hori_num + vert_num, 0, hori_num, 1);
    VectorXcf h_x4 = x.block(2 * hori_num + vert_num, 0, vert_num, 1);
    cudaMemcpy(d_x1, (fcomp*)(h_x1.data()), hori_num * sizeof(Tfcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, (fcomp*)(h_x2.data()), vert_num * sizeof(Tfcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x3, (fcomp*)(h_x3.data()), hori_num * sizeof(Tfcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x4, (fcomp*)(h_x4.data()), vert_num * sizeof(Tfcomp), cudaMemcpyHostToDevice);
    near();
    far0();
    far1();
    if (isDielectric) {
        far2();
        far3();
    }
    VectorXcf y = VectorXcf::Zero(2 * hori_num + 2 * vert_num);
    cudaMemcpy((fcomp*)(h_y1.data()), d_y1, hori_num * sizeof(fcomp), cudaMemcpyDeviceToHost);
    cudaMemcpy((fcomp*)(h_y2.data()), d_y2, vert_num * sizeof(fcomp), cudaMemcpyDeviceToHost);
    cudaMemcpy((fcomp*)(h_y3.data()), d_y3, hori_num * sizeof(fcomp), cudaMemcpyDeviceToHost);
    cudaMemcpy((fcomp*)(h_y4.data()), d_y4, vert_num * sizeof(fcomp), cudaMemcpyDeviceToHost);
    y.block(0, 0, hori_num, 1) = h_y1;
    y.block(hori_num, 0, vert_num, 1) = h_y2;
    y.block(hori_num + vert_num, 0, hori_num, 1) = h_y3;
    y.block(2 * hori_num + vert_num, 0, vert_num, 1) = h_y4;
    return y;
}

/// @brief Multiply the sparse correction matrix to the given vector
void MVProd2::near() {
    // Initialization: clearing all vectors to become zero-valued
    cusparseDnVecSetValues(x1, d_x1);
    cusparseDnVecSetValues(x2, d_x2);
    cusparseDnVecSetValues(x3, d_x3);
    cusparseDnVecSetValues(x4, d_x4);
    clearData<<< hori_num / 256 + 1, 256 >>>(d_y1, hori_num);
    clearData<<< vert_num / 256 + 1, 256 >>>(d_y2, vert_num);
    clearData<<< hori_num / 256 + 1, 256 >>>(d_y3, hori_num);
    clearData<<< vert_num / 256 + 1, 256 >>>(d_y4, vert_num);
    cusparseDnVecSetValues(y1, d_y1);
    cusparseDnVecSetValues(y2, d_y2);
    cusparseDnVecSetValues(y3, d_y3);
    cusparseDnVecSetValues(y4, d_y4);

    // Sparse near matrix multiplication 0
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aee[0], x1, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aee[0], x1, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bee[0], x1, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bee[0], x1, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[0], x3, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[0], x3, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[0], x3, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[0], x3, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[0], x1, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[0], x1, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[0], x1, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[0], x1, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Amm[0], x3, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Amm[0], x3, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bmm[0], x3, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bmm[0], x3, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);

    // Sparse near matrix multiplication 1
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aee[1], x2, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bee[1], x2, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[1], x4, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[1], x4, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aee[1], x1, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bee[1], x1, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[1], x3, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[1], x3, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[1], x2, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[1], x2, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Amm[1], x4, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bmm[1], x4, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[1], x1, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[1], x1, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Amm[1], x3, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bmm[1], x3, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);

    // Sparse near matrix multiplication 2
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aee[2], x2, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bee[2], x2, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[2], x4, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[2], x4, &beta, y1, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aee[2], x1, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bee[2], x1, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[2], x3, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[2], x3, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[2], x2, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[2], x2, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Amm[2], x4, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bmm[2], x4, &beta, y3, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[2], x1, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[2], x1, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Amm[2], x3, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bmm[2], x3, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);

    // Sparse near matrix multiplication 3
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aee[3], x2, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aee[3], x2, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bee[3], x2, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bee[3], x2, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[3], x4, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[3], x4, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[3], x4, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[3], x4, &beta, y2, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Aem[3], x2, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Aem[3], x2, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bem[3], x2, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bem[3], x2, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Amm[3], x4, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Amm[3], x4, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, Bmm[3], x4, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, Bmm[3], x4, &beta, y4, CUDA_C_32F, CUSPARSE_SPMV_COO_ALG1, d_work);
}

/// @brief Multiply the base approximation matrix to the given vector: group 0
void MVProd2::far0() {
    // Far matrix multiplication for block (Medium1, Zee & Zme): FFT group 1
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x2, vert_row, vert_col, num_pts, d_vert_y, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g3_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g0_data, geo1_data, N, const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2, geo1_data, vert_num, num_pts, N, d_vert_y, d_vert_b, false);
    
    // Far matrix multiplication for block (Medium1, Zee & Zme): FFT group 2
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo1_data, d_x1, hori_row, hori_col, num_pts, d_hori_z, d_hori_f);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x2, vert_row, vert_col, num_pts, d_vert_z, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g2_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3, geo0_data, hori_num, num_pts, N, d_hori_x, d_hori_b, false);

    // Far matrix multiplication for block (Medium1, Zee & Zme): FFT group 3
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g1_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g0_data, geo1_data, N, const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1, geo1_data, hori_num, num_pts, N, d_hori_z, d_hori_b, false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2, geo1_data, vert_num, num_pts, N, d_vert_z, d_vert_b, false);

    // Far matrix multiplication for block (Medium1, Zee & Zme): FFT group 4
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo1_data, d_x1, hori_row, hori_col, num_pts, d_hori_x, d_hori_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g3_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4, geo0_data, vert_num, num_pts, N, d_vert_y, d_vert_b, false);

    // Far matrix multiplication for block (Medium1, Zee & Zme): FFT group 5
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g2_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g0_data, geo1_data, N, const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1, geo1_data, hori_num, num_pts, N, d_hori_x, d_hori_b, false);

    // Far matrix multiplication for block (Medium1, Zee & Zme): FFT group 6
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x2, vert_row, vert_col, num_pts, d_vert_y, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g1_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3, geo0_data, hori_num, num_pts, N, d_hori_z, d_hori_b, false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4, geo0_data, vert_num, num_pts, N, d_vert_z, d_vert_b, false);

    // Far matrix multiplication for block (Medium1, Zee & Zme): FFT group 7
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo0_data, d_x1, hori_row, hori_col, num_pts, d_hori_d, d_hori_f);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo0_data, d_x2, vert_row, vert_col, num_pts, d_vert_d, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_FORWARD);
    convolveScale<<< N / 256 + 1, 256 >>>(g0_data, geo0_data, N, const21);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1, geo0_data, hori_num, num_pts, N, d_hori_d, d_hori_b, true);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2, geo0_data, vert_num, num_pts, N, d_vert_d, d_vert_b, true);
}

/// @brief Multiply the base approximation matrix to the given vector: group 1
void MVProd2::far1() {
    // Far matrix multiplication for block (Medium1, Zem & Zmm): FFT group 1
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x4, vert_row, vert_col, num_pts, d_vert_y, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g3_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g0_data, geo1_data, N, e1 * e1 * const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4, geo1_data, vert_num, num_pts, N, d_vert_y, d_vert_b, true);
    
    // Far matrix multiplication for block (Medium1, Zem & Zmm): FFT group 2
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo1_data, d_x3, hori_row, hori_col, num_pts, d_hori_z, d_hori_f);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x4, vert_row, vert_col, num_pts, d_vert_z, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g2_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1, geo0_data, hori_num, num_pts, N, d_hori_x, d_hori_b, false);

    // Far matrix multiplication for block (Medium1, Zem & Zmm): FFT group 3
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g1_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g0_data, geo1_data, N, e1 * e1 * const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3, geo1_data, hori_num, num_pts, N, d_hori_z, d_hori_b, true);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4, geo1_data, vert_num, num_pts, N, d_vert_z, d_vert_b, true);

    // Far matrix multiplication for block (Medium1, Zem & Zmm): FFT group 4
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo1_data, d_x3, hori_row, hori_col, num_pts, d_hori_x, d_hori_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g3_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2, geo0_data, vert_num, num_pts, N, d_vert_y, d_vert_b, false);

    // Far matrix multiplication for block (Medium1, Zem & Zmm): FFT group 5
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g2_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g0_data, geo1_data, N, e1 * e1 * const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3, geo1_data, hori_num, num_pts, N, d_hori_x, d_hori_b, true);

    // Far matrix multiplication for block (Medium1, Zem & Zmm): FFT group 6
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x4, vert_row, vert_col, num_pts, d_vert_y, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g1_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1, geo0_data, hori_num, num_pts, N, d_hori_z, d_hori_b, false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2, geo0_data, vert_num, num_pts, N, d_vert_z, d_vert_b, false);

    // Far matrix multiplication for block (Medium1, Zem & Zmm): FFT group 7
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo0_data, d_x3, hori_row, hori_col, num_pts, d_hori_d, d_hori_f);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo0_data, d_x4, vert_row, vert_col, num_pts, d_vert_d, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_FORWARD);
    convolveScale<<< N / 256 + 1, 256 >>>(g0_data, geo0_data, N, e1 * e1 * const21);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3, geo0_data, hori_num, num_pts, N, d_hori_d, d_hori_b, false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4, geo0_data, vert_num, num_pts, N, d_vert_d, d_vert_b, false);
}

/// @brief Multiply the base approximation matrix to the given vector: group 2
void MVProd2::far2() {
    // Far matrix multiplication for block (Medium2, Zee & Zme): FFT group 1
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x2, vert_row, vert_col, num_pts, d_vert_Y, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g7_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g4_data, geo1_data, N, const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2, geo1_data, vert_num, num_pts, N, d_vert_Y, d_vert_b, false);
    
    // Far matrix multiplication for block (Medium2, Zee & Zme): FFT group 2
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo1_data, d_x1, hori_row, hori_col, num_pts, d_hori_Z, d_hori_f);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x2, vert_row, vert_col, num_pts, d_vert_Z, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g6_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3, geo0_data, hori_num, num_pts, N, d_hori_X, d_hori_b, false);

    // Far matrix multiplication for block (Medium2, Zee & Zme): FFT group 3
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g5_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g4_data, geo1_data, N, const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1, geo1_data, hori_num, num_pts, N, d_hori_Z, d_hori_b, false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2, geo1_data, vert_num, num_pts, N, d_vert_Z, d_vert_b, false);

    // Far matrix multiplication for block (Medium2, Zee & Zme): FFT group 4
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo1_data, d_x1, hori_row, hori_col, num_pts, d_hori_X, d_hori_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g7_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4, geo0_data, vert_num, num_pts, N, d_vert_Y, d_vert_b, false);

    // Far matrix multiplication for block (Medium2, Zee & Zme): FFT group 5
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g6_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g4_data, geo1_data, N, const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1, geo1_data, hori_num, num_pts, N, d_hori_X, d_hori_b, false);

    // Far matrix multiplication for block (Medium2, Zee & Zme): FFT group 6
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x2, vert_row, vert_col, num_pts, d_vert_Y, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g5_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3, geo0_data, hori_num, num_pts, N, d_hori_Z, d_hori_b, false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4, geo0_data, vert_num, num_pts, N, d_vert_Z, d_vert_b, false);

    // Far matrix multiplication for block (Medium2, Zee & Zme): FFT group 7
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo0_data, d_x1, hori_row, hori_col, num_pts, d_hori_D, d_hori_f);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo0_data, d_x2, vert_row, vert_col, num_pts, d_vert_D, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_FORWARD);
    convolveScale<<< N / 256 + 1, 256 >>>(g4_data, geo0_data, N, const22);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1, geo0_data, hori_num, num_pts, N, d_hori_D, d_hori_b, true);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2, geo0_data, vert_num, num_pts, N, d_vert_D, d_vert_b, true);
}

/// @brief Multiply the base approximation matrix to the given vector: group 3
void MVProd2::far3() {
    // Far matrix multiplication for block (Medium2, Zem & Zmm): FFT group 1
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x4, vert_row, vert_col, num_pts, d_vert_Y, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g7_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g4_data, geo1_data, N, e2 * e2 * const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4, geo1_data, vert_num, num_pts, N, d_vert_Y, d_vert_b, true);
    
    // Far matrix multiplication for block (Medium2, Zem & Zmm): FFT group 2
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo1_data, d_x3, hori_row, hori_col, num_pts, d_hori_Z, d_hori_f);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x4, vert_row, vert_col, num_pts, d_vert_Z, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g6_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1, geo0_data, hori_num, num_pts, N, d_hori_X, d_hori_b, false);

    // Far matrix multiplication for block (Medium2, Zem & Zmm): FFT group 3
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g5_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g4_data, geo1_data, N, e2 * e2 * const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3, geo1_data, hori_num, num_pts, N, d_hori_Z, d_hori_b, true);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4, geo1_data, vert_num, num_pts, N, d_vert_Z, d_vert_b, true);

    // Far matrix multiplication for block (Medium2, Zem & Zmm): FFT group 4
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo1_data, d_x3, hori_row, hori_col, num_pts, d_hori_X, d_hori_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g7_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2, geo0_data, vert_num, num_pts, N, d_vert_Y, d_vert_b, false);

    // Far matrix multiplication for block (Medium2, Zem & Zmm): FFT group 5
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g6_data, geo1_data, N, false);
    convolveScale<<< N / 256 + 1, 256 >>>(g4_data, geo1_data, N, e2 * e2 * const1);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3, geo1_data, hori_num, num_pts, N, d_hori_X, d_hori_b, true);

    // Far matrix multiplication for block (Medium2, Zem & Zmm): FFT group 6
    clearData<<< N / 256 + 1, 256 >>>(geo1_data, N);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo1_data, d_x4, vert_row, vert_col, num_pts, d_vert_Y, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo1_data, (cufftComplex*)geo1_data, CUFFT_FORWARD);
    convolveTransfer<<< N / 256 + 1, 256 >>>(geo0_data, g5_data, geo1_data, N, true);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y1, geo0_data, hori_num, num_pts, N, d_hori_Z, d_hori_b, false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y2, geo0_data, vert_num, num_pts, N, d_vert_Z, d_vert_b, false);

    // Far matrix multiplication for block (Medium2, Zem & Zmm): FFT group 7
    clearData<<< N / 256 + 1, 256 >>>(geo0_data, N);
    scatter<<< hori_col / 256 + 1, 256 >>>(geo0_data, d_x3, hori_row, hori_col, num_pts, d_hori_D, d_hori_f);
    scatter<<< vert_col / 256 + 1, 256 >>>(geo0_data, d_x4, vert_row, vert_col, num_pts, d_vert_D, d_vert_f);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_FORWARD);
    convolveScale<<< N / 256 + 1, 256 >>>(g4_data, geo0_data, N, e2 * e2 * const22);
    cufftExecC2C(plan, (cufftComplex*)geo0_data, (cufftComplex*)geo0_data, CUFFT_INVERSE);
    accumulate<<< hori_num / 256 + 1, 256 >>>(d_y3, geo0_data, hori_num, num_pts, N, d_hori_D, d_hori_b, false);
    accumulate<<< vert_num / 256 + 1, 256 >>>(d_y4, geo0_data, vert_num, num_pts, N, d_vert_D, d_vert_b, false);
}

/// @brief Destroy the FFT computation plans and deallocated associated memory
void MVProd2::cleanAll() {
    cusparseDestroy(handle);
    for (int i = 0; i < 4; i++) {
        cudaFree(d_Arows[i]);
        cudaFree(d_Acols[i]);
        cudaFree(d_Brows[i]);
        cudaFree(d_Bcols[i]);
        cudaFree(d_Aee[i]);
        cudaFree(d_Aem[i]);
        cudaFree(d_Amm[i]);
        cudaFree(d_Bee[i]);
        cudaFree(d_Bem[i]);
        cudaFree(d_Bmm[i]);
        cudaFree(d_base1[i]);
        cudaFree(d_base2[i]);
        cusparseDestroySpMat(Aee[i]);
        cusparseDestroySpMat(Aem[i]);
        cusparseDestroySpMat(Amm[i]);
        cusparseDestroySpMat(Bee[i]);
        cusparseDestroySpMat(Bem[i]);
        cusparseDestroySpMat(Bmm[i]);
    }
    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_x3);
    cudaFree(d_x4);
    cudaFree(d_y1);
    cudaFree(d_y2);
    cudaFree(d_y3);
    cudaFree(d_y4);
    cudaFree(zvals);
    cudaFree(xvech);
    cudaFree(wvech);
    cudaFree(xvecl);
    cudaFree(wvecl);
    cudaFree(d_hori_x);
    cudaFree(d_hori_z);
    cudaFree(d_hori_d);
    cudaFree(d_hori_f);
    cudaFree(d_hori_b);
    cudaFree(d_vert_y);
    cudaFree(d_vert_z);
    cudaFree(d_vert_d);
    cudaFree(d_vert_f);
    cudaFree(d_vert_b);
    cudaFree(d_work);
    cusparseDestroyDnVec(x1);
    cusparseDestroyDnVec(x2);
    cusparseDestroyDnVec(x3);
    cusparseDestroyDnVec(x4);
    cusparseDestroyDnVec(y1);
    cusparseDestroyDnVec(y2);
    cusparseDestroyDnVec(y3);
    cusparseDestroyDnVec(y4);
    cufftDestroy(plan);
    cudaFree(g0_data);
    cudaFree(g1_data);
    cudaFree(g2_data);
    cudaFree(g3_data);
    cudaFree(geo0_data);
    cudaFree(geo1_data);
    if (isDielectric) {
        cudaFree(g4_data);
        cudaFree(g5_data);
        cudaFree(g6_data);
        cudaFree(g7_data);
        cudaFree(d_hori_X);
        cudaFree(d_hori_Z);
        cudaFree(d_hori_D);
        cudaFree(d_vert_Y);
        cudaFree(d_vert_Z);
        cudaFree(d_vert_D);
    }
}