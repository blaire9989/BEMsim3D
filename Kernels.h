#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cusparse.h>
#include <thrust/complex.h>

typedef thrust::complex<float> Tfcomp;

__global__ void clearData(float* data, int N);
__global__ void clearData(cuComplex* data, int N);
__global__ void clearData(Tfcomp* data, int N);
__global__ void updateEEMM(int N, cuComplex* ee, cuComplex* mm, cuComplex* base1, cuComplex* base2, Tfcomp c1, Tfcomp c2, Tfcomp c3, Tfcomp c4);
__global__ void individual(int N, int Nx, int Ny, float* zvals, float d, Tfcomp k1, Tfcomp k2, Tfcomp const1, Tfcomp const21, Tfcomp const22, float* xvech, float* wvech, float* xvecl, float* wvecl, int* m_inds, int* n_inds, int mx_add, int my_add, int nx_add, int ny_add, int type, int side1, int side2, cuComplex* store1, cuComplex* store2, cuComplex* store3);
__global__ void correctionHH(int N, float dx, float dy, float dz, Tfcomp k, Tfcomp const1, Tfcomp const2, int totalY, int totalZ, int* m_inds, int* n_inds, int num_pts, int* hori_i, Tfcomp* hori_x, Tfcomp* hori_z, Tfcomp* hori_d, cuComplex* ee_old, cuComplex* em_old);
__global__ void correctionHV(int N, float dx, float dy, float dz, Tfcomp k, Tfcomp const1, Tfcomp const2, int totalY, int totalZ, int* m_inds, int* n_inds, int num_pts, int* hori_i, int* vert_i, Tfcomp* hori_x, Tfcomp* hori_z, Tfcomp* hori_d, Tfcomp* vert_y, Tfcomp* vert_z, Tfcomp* vert_d, cuComplex* ee_old, cuComplex* em_old);
__global__ void correctionVV(int N, float dx, float dy, float dz, Tfcomp k, Tfcomp const1, Tfcomp const2, int totalY, int totalZ, int* m_inds, int* n_inds, int num_pts, int* vert_i, Tfcomp* vert_y, Tfcomp* vert_z, Tfcomp* vert_d, cuComplex* ee_old, cuComplex* em_old);
__global__ void postProcess(int N, bool hhvv, float eta0, Tfcomp eta1, Tfcomp eta2, int* m_inds, int* n_inds, cuComplex* ee, cuComplex* mm, cuComplex* em);
__global__ void computeGreens(Tfcomp* greens, float dx, float dy, float dz, float eta0f, Tfcomp eta, Tfcomp k, Tfcomp const1, Tfcomp const2, int N, int totalX, int totalY, int totalZ, int type);
__global__ void scatter(Tfcomp* geometry, cuComplex* xvec, int num_row, int num_col, int num_pts, Tfcomp* coef, int* forward_inds);
__global__ void convolve(Tfcomp* greens, Tfcomp* geometry, int N);
__global__ void convolveScale(Tfcomp* greens, Tfcomp* geometry, int N, Tfcomp scale);
__global__ void convolveTransfer(Tfcomp* target, Tfcomp* greens, Tfcomp* geometry, int N, bool negate);
__global__ void accumulate(cuComplex* yvec, Tfcomp* geometry, int total, int num_pts, int N, Tfcomp* coef, int* backward_inds, bool negate);
__global__ void computeSelfEE(int N, int offset, float* self, int Nx, int Ny, float* zvals, float d, int order1, float* p1, float* w1);
__global__ void computeNeighborEE(int N, int offset, int orientation, float* neighbor, int Nx, int Ny, float* zvals, float d, int order2, float* p2, float* w2);