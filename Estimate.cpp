/* This is an important preprocessing module of the simulation code. 
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "Estimate.h"

/// @brief Estimates the size of data arrays used for the simulation, based on the proposed simulation size
/// @param x_um: total length of the surface sample along the x direction, in microns
/// @param y_um: total length of the surface sample along the y direction, in microns 
/// @param z: a matrix that stores the discretized height values of the surface 
/// @param useProvided: specified whether to apply the provided shift to the surface height values
/// @param shift: an optional, uniform shift to the height values to make the height values centered at z = 0
Estimate::Estimate(double x_um, double y_um, MatrixXd z, bool useProvided, double shift) {
    // Determine the size of basis elements
    d = x_um / (z.rows() - 1);
    y_um = d * (z.cols() - 1);
    
    // Trim the input height field
    int trim_x = 0, trim_y = 0;
    xrange = gridXY(x_um, z.rows() - 1, trim_x);
    yrange = gridXY(y_um, z.cols() - 1, trim_y);
    Nx = z.rows() - 1 - 2 * trim_x;
    Ny = z.cols() - 1 - 2 * trim_y;
    zvals = z.block(trim_x, trim_y, Nx + 1, Ny + 1);
    if (useProvided)
        zvals = zvals - shift * MatrixXd::Ones(Nx + 1, Ny + 1);
    else
        zvals = zvals - zvals.mean() * MatrixXd::Ones(Nx + 1, Ny + 1);
    zrange = gridZ();
    
    // Compute sparse correction matrix indices
    computeNearIndices(1);
    computeNearIndices(14);
}

/// @brief Determines the number of points in the 3D grid created around the surface, along the x or y dimension
/// @brief We try to make this number a product of small primes (2, 3, 5, 7)
/// @param size_um: the size of the surface, in microns
/// @param size_int: size of the input height field, which might be slightly trimmed in order to obtain an FFT-friendly grid size
/// @param trim: number of rows or columns to trim from the height field 
/// @return a vector that contains the x or y coordinates of the 3D grid points
VectorXd Estimate::gridXY(double size_um, int size_int, int& trim) {
    double dxy = 1.6 * d;
    int expected = ceil(size_um / dxy);
    int upper, lower;
    if (expected % 2 == 0) {
        upper = expected / 2;
        lower = -upper + 1;
    } else {
        upper = (expected - 1) / 2;
        lower = -upper;
    }
    VectorXd range2d = VectorXd::LinSpaced(size_int + 1, -size_int * d / 2, size_int * d / 2);
    VectorXd range3d = VectorXd::LinSpaced(expected, lower * dxy, upper * dxy);
    
    // Adjust the height field domain
    int countL = 0;
    double minXY = (range2d(countL) + range2d(countL + 1)) / 2;
    int ptL = floor((minXY - range3d(0)) / dxy);
    while (ptL - 1 < 0) {
        countL = countL + 1;
        minXY = (range2d(countL) + range2d(countL + 1)) / 2;
        ptL = floor((minXY - range3d(0)) / dxy);
    }
    int countR = 0;
    double maxXY = (range2d(size_int - 1 - countR) + range2d(size_int - countR)) / 2;
    int ptR = floor((maxXY - range3d(0)) / dxy);
    while (ptR + 2 >= expected) {
        countR = countR + 1;
        maxXY = (range2d(size_int - 1 - countR) + range2d(size_int - countR)) / 2;
        ptR = floor((maxXY - range3d(0)) / dxy);
    }
    trim = countL > countR ? countL : countR;
    return range3d;
}

/// @brief Determines the number of points in the 3D grid created around the surface, along the z direction
/// @brief We try to make this number a product of small primes (2, 3, 5, 7)
/// @return a vector that contains the z coordinates of the 3D grid points
VectorXd Estimate::gridZ() {
    double dz = 6.4 * d / 3.0;
    int numH = ceil(zvals.maxCoeff() / dz) + 1;
    int numL = floor(zvals.minCoeff() / dz) - 1;
    int numZ = numH - numL + 1;
    if (numZ <= 120) {
        int oldZ = numZ;
        VectorXi vec(20);
        vec << 16, 20, 24, 30, 32, 36, 40, 45, 50, 54, 60, 64, 72, 75, 80, 90, 96, 100, 108, 120;
        for (int i = 0; i < 20; i++) {
            if (vec(i) >= oldZ) {
                numZ = vec(i);
                break;
            }
        }
        int extra1 = (numZ - oldZ) / 2;
        int extra2 = numZ - oldZ - extra1;
        numH = numH + extra1;
        numL = numL - extra2;
    }
    VectorXd zrange = VectorXd::LinSpaced(numZ, numL * dz, numH * dz);
    return zrange;
}

/// @brief Determine the row and column indices that specify the sparcity patterns in the sparse correction matrix blocks
/// @brief The code is run twice with different inputs (1 and 14).
/// @brief The A[0]--A[3] indices refer to matrix elements that require addressing singularities in integrals (overlapping or neighboring element pairs)
/// @brief The B[0]--B[3] indices specify the sparsity patterns for the full correction matrix blocks
/// @param thres: a distance threshold that determines which basis elements are considered close to each other
void Estimate::computeNearIndices(int thres) {
    int hori_num = (Nx - 1) * Ny;
    int vert_num = (Ny - 1) * Nx;

    // 1st quarter of indices
    int hh0 = 0;
    vector<int> mvec0, nvec0;
    for (int m = 0; m < hori_num; m++) {
        int my = m / (Nx - 1);
        int mx = m - (Nx - 1) * my;
        int min_nx = mx - thres - 1 >= 0 ? mx - thres - 1: 0;
        int max_nx = mx + thres + 1 < Nx - 1 ? mx + thres + 1: Nx - 2;
        int max_ny = my + thres + 1 < Ny ? my + thres + 1: Ny - 1;
        for (int ny = my; ny <= max_ny; ny++) {
            for (int nx = min_nx; nx <= max_nx; nx++) {
                if (abs(mx - nx) + abs(my - ny) > thres && abs(mx - nx - 1) + abs(my - ny) > thres && abs(mx - nx + 1) + abs(my - ny) > thres)
                    continue;
                int n = (Nx - 1) * ny + nx;
                if (n < m)
                    continue;
                mvec0.push_back(m);
                nvec0.push_back(n);
                hh0 += 1;
            }
        }
    }
    if (thres == 1) {
        A[0] = MatrixXi::Zero(hh0, 2);
        for (int i = 0; i < hh0; i++) {
            A[0](i, 0) = mvec0[i];
            A[0](i, 1) = nvec0[i];
        }
    } else {
        B[0] = MatrixXi::Zero(hh0, 2);
        for (int i = 0; i < hh0; i++) {
            B[0](i, 0) = mvec0[i];
            B[0](i, 1) = nvec0[i];
        }
    }
    
    // 2nd and 3rd quarter of indices
    int hv1 = 0, hv2 = 0;
    vector<int> mvec1, nvec1, mvec2, nvec2;
    for (int m = 0; m < hori_num; m++) {
        int my = m / (Nx - 1);
        int mx = m - (Nx - 1) * my;
        int min_nx = mx - thres - 1 >= 0 ? mx - thres - 1: 0;
        int max_nx = mx + thres + 1 < Nx ? mx + thres + 1: Nx - 1;
        int min_ny = my - thres - 1 >= 0 ? my - thres - 1: 0;
        int max_ny = my + thres + 1 < Ny - 1 ? my + thres + 1: Ny - 2;
        for (int nx = min_nx; nx <= max_nx; nx++) {
            for (int ny = min_ny; ny <= max_ny; ny++) {
                if (abs(mx - nx) + abs(my - ny) > thres && abs(mx - nx) + abs(my - ny - 1) > thres && abs(mx - nx + 1) + abs(my - ny) > thres && abs(mx - nx + 1) + abs(my - ny - 1) > thres)
                    continue;
                int n = (Ny - 1) * nx + ny;
                if (m > n) {
                    mvec1.push_back(m);
                    nvec1.push_back(n);
                    hv1 += 1;
                } else {
                    mvec2.push_back(m);
                    nvec2.push_back(n);
                    hv2 += 1;
                }
            }
        }
    }
    if (thres == 1) {
        A[1] = MatrixXi::Zero(hv1, 2);
        for (int i = 0; i < hv1; i++) {
            A[1](i, 0) = mvec1[i];
            A[1](i, 1) = nvec1[i];
        }
        A[2] = MatrixXi::Zero(hv2, 2);
        for (int i = 0; i < hv2; i++) {
            A[2](i, 0) = mvec2[i];
            A[2](i, 1) = nvec2[i];
        }
    } else {
        B[1] = MatrixXi::Zero(hv1, 2);
        for (int i = 0; i < hv1; i++) {
            B[1](i, 0) = mvec1[i];
            B[1](i, 1) = nvec1[i];
        }
        B[2] = MatrixXi::Zero(hv2, 2);
        for (int i = 0; i < hv2; i++) {
            B[2](i, 0) = mvec2[i];
            B[2](i, 1) = nvec2[i];
        }
    }
    
    // 4th quarter of indices
    int vv3 = 0;
    vector<int> mvec3, nvec3;
    for (int m = 0; m < vert_num; m++) {
        int mx = m / (Ny - 1);
        int my = m - (Ny - 1) * mx;
        int max_nx = mx + thres + 1 < Nx ? mx + thres + 1: Nx - 1;
        int min_ny = my - thres - 1 >= 0 ? my - thres - 1: 0;
        int max_ny = my + thres + 1 < Ny - 1 ? my + thres + 1: Ny - 2;
        for (int nx = mx; nx <= max_nx; nx++) {
            for (int ny = min_ny; ny <= max_ny; ny++) {
                if (abs(mx - nx) + abs(my - ny) > thres && abs(mx - nx) + abs(my - ny - 1) > thres && abs(mx - nx) + abs(my - ny + 1) > thres)
                    continue;
                int n = (Ny - 1) * nx + ny;
                if (n < m)
                    continue;
                mvec3.push_back(m);
                nvec3.push_back(n);
                vv3 += 1;
            }
        }
    }
    if (thres == 1) {
        A[3] = MatrixXi::Zero(vv3, 2);
        for (int i = 0; i < vv3; i++) {
            A[3](i, 0) = mvec3[i];
            A[3](i, 1) = nvec3[i];
        }
    } else {
        B[3] = MatrixXi::Zero(vv3, 2);
        for (int i = 0; i < vv3; i++) {
            B[3](i, 0) = mvec3[i];
            B[3](i, 1) = nvec3[i];
        }
    }
}

/// @brief Estimates the per GPU memory usage of the proposed simulation size
/// @param mem11: memory usage on the GPU for the single-GPU, memory-saving implementation, when the surface material is dielectric
/// @param mem12: memory usage on the GPU for the single-GPU, memory-saving implementation, when the surface material is lossy
/// @param mem21: memory usage on the GPU for the single-GPU, speed-oriented implementation, when the surface material is dielectric
/// @param mem22: memory usage on the GPU for the single-GPU, speed-oriented implementation, when the surface material is lossy
/// @param mem34: memory usage on each GPU for the 4-GPU implementations
void Estimate::memGB(double& mem11, double& mem12, double& mem21, double& mem22, double& mem34) {
    // Base approximation matrix data
    double fft_array = 64.0 * xrange.rows() * yrange.rows() * zrange.rows() / pow(1024.0, 3);
    double hcoef_array = 8.0 * 48.0 * (Nx - 1) * Ny / pow(1024.0, 3);
    double hind_array = 4.0 * 48.0 * (Nx - 1) * Ny / pow(1024.0, 3);
    double vcoef_array = 8.0 * 48.0 * (Ny - 1) * Nx / pow(1024.0, 3);
    double vind_array = 4.0 * 48.0 * (Ny - 1) * Nx / pow(1024.0, 3);
    
    // Sparse correction matrix data
    double ind0 = 4.0 * (A[0].rows() + B[0].rows()) / pow(1024.0, 3);
    double dataA0 = 8.0 * A[0].rows() / pow(1024.0, 3);
    double dataB0 = 8.0 * B[0].rows() / pow(1024.0, 3);
    double ind1 = 4.0 * (A[1].rows() + B[1].rows()) / pow(1024.0, 3);
    double dataA1 = 8.0 * A[1].rows() / pow(1024.0, 3);
    double dataB1 = 8.0 * B[1].rows() / pow(1024.0, 3);
    double ind2 = 4.0 * (A[2].rows() + B[2].rows()) / pow(1024.0, 3);
    double dataA2 = 8.0 * A[2].rows() / pow(1024.0, 3);
    double dataB2 = 8.0 * B[2].rows() / pow(1024.0, 3);
    double ind3 = 4.0 * (A[3].rows() + B[3].rows()) / pow(1024.0, 3);
    double dataA3 = 8.0 * A[3].rows() / pow(1024.0, 3);
    double dataB3 = 8.0 * B[3].rows() / pow(1024.0, 3);
    
    // GPU memory required when using MVProd1: dieletric
    mem11 = fft_array * 3 + hcoef_array * 6 + hind_array * 2 + vcoef_array * 6 + vind_array * 2;
    mem11 += ind0 * 2 + dataA0 * 5 + dataB0 * 3;
    mem11 += ind1 * 2 + dataA1 * 5 + dataB1 * 3;
    mem11 += ind2 * 2 + dataA2 * 5 + dataB2 * 3;
    mem11 += ind3 * 2 + dataA3 * 5 + dataB3 * 3;
    mem11 += 1.5;
    
    // GPU memory required when using MVProd1: lossy
    mem12 = fft_array * 3 + hcoef_array * 3 + hind_array * 2 + vcoef_array * 3 + vind_array * 2;
    mem12 += ind0 * 2 + dataA0 * 5 + dataB0 * 3;
    mem12 += ind1 * 2 + dataA1 * 5 + dataB1 * 3;
    mem12 += ind2 * 2 + dataA2 * 5 + dataB2 * 3;
    mem12 += ind3 * 2 + dataA3 * 5 + dataB3 * 3;
    mem12 += 1.5;
    
    // GPU memory required when using MVProd2: dieletric
    mem21 = fft_array * 11 + hcoef_array * 6 + hind_array * 2 + vcoef_array * 6 + vind_array * 2;
    mem21 += ind0 * 2 + dataA0 * 5 + dataB0 * 3;
    mem21 += ind1 * 2 + dataA1 * 5 + dataB1 * 3;
    mem21 += ind2 * 2 + dataA2 * 5 + dataB2 * 3;
    mem21 += ind3 * 2 + dataA3 * 5 + dataB3 * 3;
    mem21 += 1.5;
    
    // GPU memory required when using MVProd2: lossy
    mem22 = fft_array * 7 + hcoef_array * 3 + hind_array * 2 + vcoef_array * 3 + vind_array * 2;
    mem22 += ind0 * 2 + dataA0 * 5 + dataB0 * 3;
    mem22 += ind1 * 2 + dataA1 * 5 + dataB1 * 3;
    mem22 += ind2 * 2 + dataA2 * 5 + dataB2 * 3;
    mem22 += ind3 * 2 + dataA3 * 5 + dataB3 * 3;
    mem22 += 1.5;
    
    // GPU memory required when using MVProd3 or MVProd4
    mem34 = fft_array * 6 + hcoef_array * 3 + hind_array * 2 + vcoef_array * 3 + vind_array * 2;
    double mem34A = mem34 + ind0 * 2 + dataA0 * 5 + dataB0 * 3;
    double mem34B = mem34 + ind1 * 2 + dataA1 * 5 + dataB1 * 3;
    double mem34C = mem34 + ind2 * 2 + dataA2 * 5 + dataB2 * 3;
    double mem34D = mem34 + ind3 * 2 + dataA3 * 5 + dataB3 * 3;
    Vector4d mem34_vec;
    mem34_vec << mem34A, mem34B, mem34C, mem34D;
    mem34 = mem34_vec.maxCoeff() + 1.5;
}