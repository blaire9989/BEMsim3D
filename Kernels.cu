/* This file contains all the CUDA C++ kernels used in this simulator.
Users do not need to read or understand any kernel in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "Kernels.h"

// Setting a data array of type float to zero-valued
__global__
void clearData(float* data, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    data[i] = 0.0f;
}

// Setting a data array of type cuComplex to zero-valued
__global__
void clearData(cuComplex* data, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        data[i].x = 0.0f;
        data[i].y = 0.0f;
    }
}

// Setting a data array of type Tfcomp to zero-valued
__global__
void clearData(Tfcomp* data, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        data[i] = 0;
}

// Reset the singular matrix elements for each new simulation with given media parameters and wavelengths
__global__
void updateEEMM(int N, cuComplex* ee, cuComplex* mm, cuComplex* base1, cuComplex* base2, Tfcomp c1, Tfcomp c2, Tfcomp c3, Tfcomp c4) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    Tfcomp b1(base1[i].x, base1[i].y);
    Tfcomp b2(base2[i].x, base2[i].y);
    Tfcomp d1 = c1 * b1 + c2 * b2;
    Tfcomp d2 = c3 * b1 + c4 * b2;
    ee[i].x = d1.real();
    ee[i].y = d1.imag();
    mm[i].x = d2.real();
    mm[i].y = d2.imag();
}

// Compute a BEM matrix element using brute-force numerical integration (for nearby basis element pairs)
__global__
void individual(int N, int Nx, int Ny, float* zvals, float d, Tfcomp k1, Tfcomp k2, Tfcomp const1, Tfcomp const21, Tfcomp const22, float* xvech, float* wvech, float* xvecl, float* wvecl, int* m_inds, int* n_inds, int mx_add, int my_add, int nx_add, int ny_add, int type, int side1, int side2, cuComplex* store1, cuComplex* store2, cuComplex* store3) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    int m = m_inds[i];
    int n = n_inds[i];
    int mx, my, nx, ny;
    if (type == 1 || type == 2) {
        my = m / (Nx - 1);
        mx = m - (Nx - 1) * my;
    } else {
        mx = m / (Ny - 1);
        my = m - (Ny - 1) * mx;
    }
    if (type == 2 || type == 3) {
        nx = n / (Ny - 1);
        ny = n - (Ny - 1) * nx;
    } else {
        ny = n / (Nx - 1);
        nx = n - (Nx - 1) * ny;
    }
    mx = mx + mx_add;
    my = my + my_add;
    nx = nx + nx_add;
    ny = ny + ny_add;
    float x00_m = -Nx * d / 2 + (mx + 0.5f) * d;
    float y00_m = -Ny * d / 2 + (my + 0.5f) * d;
    float z1_m = zvals[my * (Nx + 1) + mx];
    float z2_m = zvals[my * (Nx + 1) + (mx + 1)];
    float z3_m = zvals[(my + 1) * (Nx + 1) + mx];
    float z4_m = zvals[(my + 1) * (Nx + 1) + (mx + 1)];
    float z11_m = (z1_m - z2_m - z3_m + z4_m) / 4;
    float z10_m = (-z1_m + z2_m - z3_m + z4_m) / 4;
    float z01_m = (-z1_m - z2_m + z3_m + z4_m) / 4;
    float z00_m = (z1_m + z2_m + z3_m + z4_m) / 4;
    float x00_n = -Nx * d / 2 + (nx + 0.5f) * d;
    float y00_n = -Ny * d / 2 + (ny + 0.5f) * d;
    float z1_n = zvals[ny * (Nx + 1) + nx];
    float z2_n = zvals[ny * (Nx + 1) + (nx + 1)];
    float z3_n = zvals[(ny + 1) * (Nx + 1) + nx];
    float z4_n = zvals[(ny + 1) * (Nx + 1) + (nx + 1)];
    float z11_n = (z1_n - z2_n - z3_n + z4_n) / 4;
    float z10_n = (-z1_n + z2_n - z3_n + z4_n) / 4;
    float z01_n = (-z1_n - z2_n + z3_n + z4_n) / 4;
    float z00_n = (z1_n + z2_n + z3_n + z4_n) / 4;
    float *xvec, *wvec;
    int order;
    if (abs(mx - nx) + abs(my - ny) <= 3) {
        order = 4;
        xvec = xvech;
        wvec = wvech;
    } else {
        order = 2;
        xvec = xvecl;
        wvec = wvecl;
    }
    Tfcomp ee1(0.0f, 0.0f), ee2(0.0f, 0.0f), em0(0.0f, 0.0f), cuFL(0.0f, 1.0f);
    for (int um = 0; um < order; um++) {
        float u1 = xvec[um];
        for (int vm = 0; vm < order; vm++) {
            float v1 = xvec[vm];
            for (int un = 0; un < order; un++) {
                float u2 = xvec[un];
                for (int vn = 0; vn < order; vn++) {
                    float v2 = xvec[vn];
                    float deltax = 0.5f * d * (u1 - u2) + x00_m - x00_n;
                    float deltay = 0.5f * d * (v1 - v2) + y00_m - y00_n;
                    float deltaz = z11_m * u1 * v1 + z10_m * u1 + z01_m * v1 + z00_m - z11_n * u2 * v2 - z10_n * u2 - z01_n * v2 - z00_n;
                    float r = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
                    float s1, s2, c1, c2;
                    sincosf(-k1.real() * r, &s1, &c1);
                    sincosf(-k2.real() * r, &s2, &c2);
                    float base1 = __expf(k1.imag() * r);
                    float base2 = __expf(k2.imag() * r);
                    Tfcomp val1(base1 * c1, base1 * s1), val2(base2 * c2, base2 * s2), green11, green12, green2;
                    if (abs(mx - nx) + abs(my - ny) <= 1) {
                        if (r < 1e-4) {
                            green11 = -cuFL * k1 / (4.0f * M_PI);
                            green12 = -cuFL * k2 / (4.0f * M_PI);
                        } else {
                            green11 = (val1 - 1.0f) / (4.0f * M_PI * r);
                            green12 = (val2 - 1.0f) / (4.0f * M_PI * r);
                            green2 = (val1 * (cuFL * k1 * r + 1.0f) + val2 * (cuFL * k2 * r + 1.0f) - 2.0f) / (4.0f * M_PI * r * r * r);
                        }
                    } else {
                        green11 = val1 / (4.0f * M_PI * r);
                        green12 = val2 / (4.0f * M_PI * r);
                        green2 = (val1 * (cuFL * k1 * r + 1.0f) + val2 * (cuFL * k2 * r + 1.0f)) / (4.0f * M_PI * r * r * r);
                    }
                    float weight = wvec[um] * wvec[vm] * wvec[un] * wvec[vn];
                    float common, dot;
                    if (type == 1) {
                        common = (1 - side1 * u1) * (1 - side2 * u2);
                        dot = 0.25f * d * d + (z11_m * v1 + z10_m) * (z11_n * v2 + z10_n);
                    } else if (type == 2) {
                        common = (1 - side1 * u1) * (1 - side2 * v2);
                        dot = (z11_m * v1 + z10_m) * (z11_n * u2 + z01_n);
                    } else {
                        common = (1 - side1 * v1) * (1 - side2 * v2);
                        dot = 0.25f * d * d + (z11_m * u1 + z01_m) * (z11_n * u2 + z01_n);
                    }
                    ee1 += weight * green11 * (const1 * common * dot - side1 * side2 * const21);
                    ee2 += weight * green12 * (const1 * common * dot - side1 * side2 * const22);
                    if (abs(mx - nx) + abs(my - ny) > 0) {
                        double cross;
                        if (type == 1)
                            cross = deltay * 0.5f * d * (z11_m * v1 + z10_m - z11_n * v2 - z10_n);
                        else if (type == 2)
                            cross = -deltax * 0.5f * d * (z11_m * v1 + z10_m) - deltay * 0.5f * d * (z11_n * u2 + z01_n) + deltaz * 0.25f * d * d;
                        else
                            cross = deltax * 0.5f * d * (z11_n * u2 + z01_n - z11_m * u1 - z01_m);
                        em0 += weight * green2 * common * cross;
                    }
                }
            }
        }
    }
    store1[i].x += ee1.real();
    store1[i].y += ee1.imag();
    store2[i].x += ee2.real();
    store2[i].y += ee2.imag();
    store3[i].x += em0.real();
    store3[i].y += em0.imag();
}

// Compute sparse correction matrix elements for matrix blocks that involve pairs of horizontally arranged basis functions
__global__
void correctionHH(int N, float dx, float dy, float dz, Tfcomp k, Tfcomp const1, Tfcomp const2, int totalY, int totalZ, int* m_inds, int* n_inds, int num_pts, int* hori_i, Tfcomp* hori_x, Tfcomp* hori_z, Tfcomp* hori_d, cuComplex* ee_old, cuComplex* em_old) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    int m = m_inds[i];
    int n = n_inds[i];
    Tfcomp ee(0.0f, 0.0f), em(0.0f, 0.0f);
    for (int pt1 = 0; pt1 < num_pts; pt1++) {
        int linear1 = hori_i[num_pts * m + pt1];
        int x1 = linear1 / (totalY * totalZ);
        int z1 = linear1 % totalZ;
        int y1 = (linear1 - z1 - totalY * totalZ * x1) / totalZ;
        for (int pt2 = 0; pt2 < num_pts; pt2++) {
            int linear2 = hori_i[num_pts * n + pt2];
            int x2 = linear2 / (totalY * totalZ);
            int z2 = linear2 % totalZ;
            int y2 = (linear2 - z2 - totalY * totalZ * x2) / totalZ;
            float distX = (x1 - x2) * dx;
            float distY = (y1 - y2) * dy;
            float distZ = (z1 - z2) * dz;
            float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
            Tfcomp g1(0.0f, 0.0f), g2(0.0f, 0.0f), cuFL(0.0f, 1.0f);
            if (dist > 0) {
                float fac = 4.0f * M_PI * dist, s, c;
                sincosf(-k.real() * dist, &s, &c);
                float base = __expf(k.imag() * dist) / fac;
                Tfcomp val(base * c, base * s);
                g1 = val;
                g2 = distY * val * (1.0f + cuFL * k * dist) / (dist * dist);
            }
            ee += g1 * (const1 * hori_x[num_pts * m + pt1] * hori_x[num_pts * n + pt2] + const1 * hori_z[num_pts * m + pt1] * hori_z[num_pts * n + pt2] - const2 * hori_d[num_pts * m + pt1] * hori_d[num_pts * n + pt2]);
            em += g2 * (hori_z[num_pts * m + pt1] * hori_x[num_pts * n + pt2] - hori_x[num_pts * m + pt1] * hori_z[num_pts * n + pt2]);
        }
    }
    ee_old[i].x -= ee.real();
    ee_old[i].y -= ee.imag();
    em_old[i].x -= em.real();
    em_old[i].y -= em.imag();
}

// Compute sparse correction matrix elements for matrix blocks that involve pairs of one horizontally arranged basis function and one vertically arranged basis function
__global__
void correctionHV(int N, float dx, float dy, float dz, Tfcomp k, Tfcomp const1, Tfcomp const2, int totalY, int totalZ, int* m_inds, int* n_inds, int num_pts, int* hori_i, int* vert_i, Tfcomp* hori_x, Tfcomp* hori_z, Tfcomp* hori_d, Tfcomp* vert_y, Tfcomp* vert_z, Tfcomp* vert_d, cuComplex* ee_old, cuComplex* em_old) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    int m = m_inds[i];
    int n = n_inds[i];
    Tfcomp ee(0.0f, 0.0f), em(0.0f, 0.0f);
    for (int pt1 = 0; pt1 < num_pts; pt1++) {
        int linear1 = hori_i[num_pts * m + pt1];
        int x1 = linear1 / (totalY * totalZ);
        int z1 = linear1 % totalZ;
        int y1 = (linear1 - z1 - totalY * totalZ * x1) / totalZ;
        for (int pt2 = 0; pt2 < num_pts; pt2++) {
            int linear2 = vert_i[num_pts * n + pt2];
            int x2 = linear2 / (totalY * totalZ);
            int z2 = linear2 % totalZ;
            int y2 = (linear2 - z2 - totalY * totalZ * x2) / totalZ;
            float distX = (x1 - x2) * dx;
            float distY = (y1 - y2) * dy;
            float distZ = (z1 - z2) * dz;
            float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
            Tfcomp g1(0.0f, 0.0f), g2(0.0f, 0.0f), g3(0.0f, 0.0f), g4(0.0f, 0.0f), cuFL(0.0f, 1.0f);
            if (dist > 0) {
                float fac = 4.0f * M_PI * dist, s, c;
                sincosf(-k.real() * dist, &s, &c);
                float base = __expf(k.imag() * dist) / fac;
                Tfcomp val(base * c, base * s);
                g1 = val;
                Tfcomp greens_temp = val * (1.0f + cuFL * k * dist) / (dist * dist);
                g2 = distX * greens_temp;
                g3 = distY * greens_temp;
                g4 = distZ * greens_temp;
            }
            ee += g1 * (const1 * hori_z[num_pts * m + pt1] * vert_z[num_pts * n + pt2] - const2 * hori_d[num_pts * m + pt1] * vert_d[num_pts * n + pt2]);
            em += g4 * hori_x[num_pts * m + pt1] * vert_y[num_pts * n + pt2] - g2 * hori_z[num_pts * m + pt1] * vert_y[num_pts * n + pt2] - g3 * hori_x[num_pts * m + pt1] * vert_z[num_pts * n + pt2];
        }
    }
    ee_old[i].x -= ee.real();
    ee_old[i].y -= ee.imag();
    em_old[i].x -= em.real();
    em_old[i].y -= em.imag();
}

// Compute sparse correction matrix elements for matrix blocks that involve pairs of vertically arranged basis functions
__global__
void correctionVV(int N, float dx, float dy, float dz, Tfcomp k, Tfcomp const1, Tfcomp const2, int totalY, int totalZ, int* m_inds, int* n_inds, int num_pts, int* vert_i, Tfcomp* vert_y, Tfcomp* vert_z, Tfcomp* vert_d, cuComplex* ee_old, cuComplex* em_old) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    int m = m_inds[i];
    int n = n_inds[i];
    Tfcomp ee(0.0f, 0.0f), em(0.0f, 0.0f);
    for (int pt1 = 0; pt1 < num_pts; pt1++) {
        int linear1 = vert_i[num_pts * m + pt1];
        int x1 = linear1 / (totalY * totalZ);
        int z1 = linear1 % totalZ;
        int y1 = (linear1 - z1 - totalY * totalZ * x1) / totalZ;
        for (int pt2 = 0; pt2 < num_pts; pt2++) {
            int linear2 = vert_i[num_pts * n + pt2];
            int x2 = linear2 / (totalY * totalZ);
            int z2 = linear2 % totalZ;
            int y2 = (linear2 - z2 - totalY * totalZ * x2) / totalZ;
            float distX = (x1 - x2) * dx;
            float distY = (y1 - y2) * dy;
            float distZ = (z1 - z2) * dz;
            float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
            Tfcomp g1(0.0f, 0.0f), g2(0.0f, 0.0f), cuFL(0.0f, 1.0f);
            if (dist > 0) {
                float fac = 4.0f * M_PI * dist, s, c;
                sincosf(-k.real() * dist, &s, &c);
                float base = __expf(k.imag() * dist) / fac;
                Tfcomp val(base * c, base * s);
                g1 = val;
                g2 = distX * val * (1.0f + cuFL * k * dist) / (dist * dist);
            }
            ee += g1 * (const1 * vert_y[num_pts * m + pt1] * vert_y[num_pts * n + pt2] + const1 * vert_z[num_pts * m + pt1] * vert_z[num_pts * n + pt2] - const2 * vert_d[num_pts * m + pt1] * vert_d[num_pts * n + pt2]);
            em += g2 * (vert_y[num_pts * m + pt1] * vert_z[num_pts * n + pt2] - vert_z[num_pts * m + pt1] * vert_y[num_pts * n + pt2]);
        }
    }
    ee_old[i].x -= ee.real();
    ee_old[i].y -= ee.imag();
    em_old[i].x -= em.real();
    em_old[i].y -= em.imag();
}

// Postprocess and finalize the sparse correct matrix data for each simulation
__global__
void postProcess(int N, bool hhvv, float eta0FL, Tfcomp eta1, Tfcomp eta2, int* m_inds, int* n_inds, cuComplex* ee, cuComplex* mm, cuComplex* em) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    Tfcomp data1(ee[i].x, ee[i].y), data2(mm[i].x, mm[i].y);
    Tfcomp ee_temp = data1 + data2;
    Tfcomp mm_temp = -eta1 * eta1 * data1 - eta2 * eta2 * data2;
    ee[i].x = ee_temp.real();
    ee[i].y = ee_temp.imag();
    mm[i].x = mm_temp.real();
    mm[i].y = mm_temp.imag();
    em[i].x = eta0FL * em[i].x;
    em[i].y = eta0FL * em[i].y;
    if (hhvv && m_inds[i] == n_inds[i]) {
        ee[i].x = 0.5f * ee[i].x;
        ee[i].y = 0.5f * ee[i].y;
        mm[i].x = 0.5f * mm[i].x;
        mm[i].y = 0.5f * mm[i].y;
        em[i].x = 0.5f * em[i].x;
        em[i].y = 0.5f * em[i].y;
    }
}

// Compute the shift-invarient g (Green's) function values for pairs of points in the 3D grid
__global__
void computeGreens(Tfcomp* greens, float dx, float dy, float dz, float eta0FL, Tfcomp eta, Tfcomp k, Tfcomp const1, Tfcomp const2, int N, int totalX, int totalY, int totalZ, int type) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    int countZ = i % totalZ;
    int countX = i / (totalY * totalZ);
    int countY = (i - countZ - totalY * totalZ * countX) / totalZ;
    if (countX == 0 || countY == 0 || countZ == 0) {
        greens[i] = 0;
        return;
    }
    float distX = (countX - totalX / 2) * dx;
    float distY = (countY - totalY / 2) * dy;
    float distZ = (countZ - totalZ / 2) * dz;
    float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
    if (dist == 0) {
        greens[i] = 0;
        return;
    }
    float fac = 4.0f * M_PI * dist;
    float s, c;
    sincosf(-k.real() * dist, &s, &c);
    float base = __expf(k.imag() * dist) / fac;
    Tfcomp cuFL(0.0f, 1.0f), val0(base * c, base * s), val;
    if (type == 0)
        val = const1 * val0;
    else if (type == 1)
        val = const2 * val0;
    else if (type == 2)
        val = const1 * eta * eta * val0;
    else if (type == 3)
        val = const2 * eta * eta * val0;
    else if (type == 4)
        val = eta0FL * distX * val0 * (1.0f + cuFL * k * dist) / (dist * dist);
    else if (type == 5)
        val = eta0FL * distY * val0 * (1.0f + cuFL * k * dist) / (dist * dist);
    else if (type == 6)
        val = eta0FL * distZ * val0 * (1.0f + cuFL * k * dist) / (dist * dist);
    else
        val = val0;
    greens[i] = val;
}

// Multiply the Lambda_2^T point source approximation coefficient matrix to a given vector
__global__
void scatter(Tfcomp* geometry, cuComplex* xvec, int num_row, int num_col, int num_pts, Tfcomp* coef, int* forward_inds) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_col)
        return;
    int linear = forward_inds[num_row * i];
    Tfcomp sum = 0;
    for (int pt = 1; pt < num_row; pt++) {
        int index = forward_inds[num_row * i + pt];
        if (index == -1)
            break;
        int c0 = index / num_pts;
        Tfcomp x0(xvec[c0].x, xvec[c0].y);
        sum += coef[index] * x0;
    }
    geometry[linear] += sum;
}

// Convolve two functions through element-wise multiplication in the Fourier domain
__global__
void convolve(Tfcomp* greens, Tfcomp* geometry, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        geometry[i] = greens[i] * geometry[i];
}

// Convolve two functions through element-wise multiplication in the Fourier domain; further apply a scale factor to the results
__global__
void convolveScale(Tfcomp* greens, Tfcomp* geometry, int N, Tfcomp scale) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        geometry[i] = scale * greens[i] * geometry[i];
}

// Convolve two functions through element-wise multiplication in the Fourier domain; further write the results into a designated array
__global__
void convolveTransfer(Tfcomp* target, Tfcomp* greens, Tfcomp* geometry, int N, bool negate) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    if (negate)
        target[i] -= greens[i] * geometry[i];
    else
        target[i] += greens[i] * geometry[i];
}

// Multiply the Lambda_1 point source approximation coefficient matrix to a given vector
__global__
void accumulate(cuComplex* yvec, Tfcomp* geometry, int total, int num_pts, int N, Tfcomp* coef, int* backward_inds, bool negate) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total)
        return;
    Tfcomp sum = 0;
    for (int pt = 0; pt < num_pts; pt++) {
        int linear = backward_inds[num_pts * i + pt];
        if (negate)
            sum -= coef[num_pts * i + pt] * geometry[linear];
        else
            sum += coef[num_pts * i + pt] * geometry[linear];
    }
    yvec[i].x += sum.real() / N;
    yvec[i].y += sum.imag() / N;
}

// Compute matrix elements in some BEM matrix blocks for overlapping basis element pairs
__global__
void computeSelfEE(int N, int offset, float* self, int Nx, int Ny, float* zvals, float d, int order1, float* p1, float* w1) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    int ind = i + offset;
    int num_x = ind / Ny;
    int num_y = ind - Ny * num_x;
    float z0 = zvals[num_y * (Nx + 1) + num_x];
    float z1 = zvals[num_y * (Nx + 1) + (num_x + 1)] - z0;
    float z2 = zvals[(num_y + 1) * (Nx + 1) + num_x] - z0;
    float z3 = zvals[(num_y + 1) * (Nx + 1) + (num_x + 1)] - z0;
    float s00 = 0, s01 = 0, s02 = 0, s03 = 0, s11 = 0, s12 = 0, s13 = 0, s22 = 0, s23 = 0, s33 = 0, div = 0;
    float z11 = (z3 - z1 - z2) / 4;
    float z10 = (z1 - z2 + z3) / 4;
    float z01 = (z2 + z3 - z1) / 4;
    for (int out_u = 0; out_u < order1; out_u++) {
        float u0 = p1[out_u];
        for (int out_v = 0; out_v < order1; out_v++) {
            float v0 = p1[out_v];
            float out_weight = w1[out_u] * w1[out_v];
            float A = z11 * v0 + z10;
            float B = z11 * u0 + z01;
            for (int in_1 = 0; in_1 < order1; in_1++) {
                for (int in_2 = 0; in_2 < order1; in_2++) {
                    float t1 = 0.5f * p1[in_1] + 0.5f;
                    float t2 = 0.5f * p1[in_2] + 0.5f;
                    float weight = out_weight * w1[in_1] * w1[in_2] / 4;
                    float temp1 = z11 * (2 * t1 - 1 - u0) * (-1 - v0) * (1 - t2) + A * (2 * t1 - 1 - u0) + B * (-1 - v0);
                    float base1 = weight / (2 * M_PI) * (v0 + 1) / sqrt(0.25f * d * d * (2 * t1 - 1 - u0) * (2 * t1 - 1 - u0) + 0.25f * d * d * (-1 - v0) * (-1 - v0) + temp1 * temp1);
                    float u1 = t2 * u0 + t2 - 1 + 2 * (1 - t2) * t1;
                    float v1 = t2 * (v0 + 1) - 1;
                    float common11 = 0.25f * d * d + (z11 * v0 + z10) * (z11 * v1 + z10);
                    float common12 = (z11 * v0 + z10) * (z11 * u1 + z01);
                    float common13 = 0.25f * d * d + (z11 * u0 + z01) * (z11 * u1 + z01);
                    s00 += (1 - u0) * (1 - u1) * common11 * base1;
                    s01 += (1 - u0) * (1 + u1) * common11 * base1;
                    s02 += (1 - u0) * (1 - v1) * common12 * base1;
                    s03 += (1 - u0) * (1 + v1) * common12 * base1;
                    s11 += (1 + u0) * (1 + u1) * common11 * base1;
                    s12 += (1 + u0) * (1 - v1) * common12 * base1;
                    s13 += (1 + u0) * (1 + v1) * common12 * base1;
                    s22 += (1 - v0) * (1 - v1) * common13 * base1;
                    s23 += (1 - v0) * (1 + v1) * common13 * base1;
                    s33 += (1 + v0) * (1 + v1) * common13 * base1;
                    float temp2 = z11 * (1 - u0) * (2 * t2 - v0 - 1) * t1 + A * (1 - u0) + B * (2 * t2 - 1 - v0);
                    float base2 = weight / (2 * M_PI) * (1 - u0) / sqrt(0.25f * d * d * (1 - u0) * (1 - u0) + 0.25f * d * d * (2 * t2 - 1 - v0) * (2 * t2 - 1 - v0) + temp2 * temp2);
                    float u2 = (1 - t1) * u0 + t1;
                    float v2 = (1 - t1) * v0 + 2 * t1 * t2 - t1;
                    float common21 = 0.25f * d * d + (z11 * v0 + z10) * (z11 * v2 + z10);
                    float common22 = (z11 * v0 + z10) * (z11 * u2 + z01);
                    float common23 = 0.25f * d * d + (z11 * u0 + z01) * (z11 * u2 + z01);
                    s00 += (1 - u0) * (1 - u2) * common21 * base2;
                    s01 += (1 - u0) * (1 + u2) * common21 * base2;
                    s02 += (1 - u0) * (1 - v2) * common22 * base2;
                    s03 += (1 - u0) * (1 + v2) * common22 * base2;
                    s11 += (1 + u0) * (1 + u2) * common21 * base2;
                    s12 += (1 + u0) * (1 - v2) * common22 * base2;
                    s13 += (1 + u0) * (1 + v2) * common22 * base2;
                    s22 += (1 - v0) * (1 - v2) * common23 * base2;
                    s23 += (1 - v0) * (1 + v2) * common23 * base2;
                    s33 += (1 + v0) * (1 + v2) * common23 * base2;
                    float temp3 = z11 * (2 * t1 - u0 - 1) * (1 - v0) * t2 + A * (2 * t1 - u0 - 1) + B * (1 - v0);
                    float base3 = weight / (2 * M_PI) * (1 - v0) / sqrt(0.25f * d * d * (2 * t1 - u0 - 1) * (2 * t1 - u0 - 1) + 0.25f * d * d * (1 - v0) * (1 - v0) + temp3 * temp3);
                    float u3 = (1 - t2) * u0 + 2 * t1 * t2 - t2;
                    float v3 = (1 - t2) * v0 + t2;
                    float common31 = 0.25f * d * d + (z11 * v0 + z10) * (z11 * v3 + z10);
                    float common32 = (z11 * v0 + z10) * (z11 * u3 + z01);
                    float common33 = 0.25f * d * d + (z11 * u0 + z01) * (z11 * u3 + z01);
                    s00 += (1 - u0) * (1 - u3) * common31 * base3;
                    s01 += (1 - u0) * (1 + u3) * common31 * base3;
                    s02 += (1 - u0) * (1 - v3) * common32 * base3;
                    s03 += (1 - u0) * (1 + v3) * common32 * base3;
                    s11 += (1 + u0) * (1 + u3) * common31 * base3;
                    s12 += (1 + u0) * (1 - v3) * common32 * base3;
                    s13 += (1 + u0) * (1 + v3) * common32 * base3;
                    s22 += (1 - v0) * (1 - v3) * common33 * base3;
                    s23 += (1 - v0) * (1 + v3) * common33 * base3;
                    s33 += (1 + v0) * (1 + v3) * common33 * base3;
                    float temp4 = z11 * (-1 - u0) * (2 * t2 - v0 - 1) * (1 - t1) + A * (-1 - u0) + B * (2 * t2 - v0 - 1);
                    float base4 = weight / (2 * M_PI) * (1 + u0) / sqrt(0.25f * d * d * (-1 - u0) * (-1 - u0) + 0.25f * d * d * (2 * t2 - v0 - 1) * (2 * t2 - v0 - 1) + temp4 * temp4);
                    float u4 = t1 * u0 - (1 - t1);
                    float v4 = t1 * v0 - 1 + t1 + 2 * t2 - 2 * t1 * t2;
                    float common41 = 0.25f * d * d + (z11 * v0 + z10) * (z11 * v4 + z10);
                    float common42 = (z11 * v0 + z10) * (z11 * u4 + z01);
                    float common43 = 0.25f * d * d + (z11 * u0 + z01) * (z11 * u4 + z01);
                    s00 += (1 - u0) * (1 - u4) * common41 * base4;
                    s01 += (1 - u0) * (1 + u4) * common41 * base4;
                    s02 += (1 - u0) * (1 - v4) * common42 * base4;
                    s03 += (1 - u0) * (1 + v4) * common42 * base4;
                    s11 += (1 + u0) * (1 + u4) * common41 * base4;
                    s12 += (1 + u0) * (1 - v4) * common42 * base4;
                    s13 += (1 + u0) * (1 + v4) * common42 * base4;
                    s22 += (1 - v0) * (1 - v4) * common43 * base4;
                    s23 += (1 - v0) * (1 + v4) * common43 * base4;
                    s33 += (1 + v0) * (1 + v4) * common43 * base4;
                    div += base1 + base2 + base3 + base4;
                }
            }
        }
    }
    self[11 * i + 0] = s00;
    self[11 * i + 1] = s01;
    self[11 * i + 2] = s02;
    self[11 * i + 3] = s03;
    self[11 * i + 4] = s11;
    self[11 * i + 5] = s12;
    self[11 * i + 6] = s13;
    self[11 * i + 7] = s22;
    self[11 * i + 8] = s23;
    self[11 * i + 9] = s33;
    self[11 * i + 10] = div;
}

// Compute matrix elements in some BEM matrix blocks for neighboring basis element pairs
__global__
void computeNeighborEE(int N, int offset, int orientation, float* neighbor, int Nx, int Ny, float* zvals, float d, int order2, float* p2, float* w2) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    int ind = i + offset;
    float z1, z2, z3, z4, z5;
    if (orientation == 0) {
        int num_y = ind / (Nx - 1);
        int num_x = ind - (Nx - 1) * num_y + 1;
        float z0 = zvals[num_y * (Nx + 1) + num_x];
        z1 = zvals[num_y * (Nx + 1) + (num_x + 1)] - z0;
        z2 = zvals[(num_y + 1) * (Nx + 1) + num_x] - z0;
        z3 = zvals[(num_y + 1) * (Nx + 1) + (num_x + 1)] - z0;
        z4 = zvals[num_y * (Nx + 1) + (num_x - 1)] - z0;
        z5 = zvals[(num_y + 1) * (Nx + 1) + (num_x - 1)] - z0;
    } else {
        int num_x = ind / (Ny - 1);
        int num_y = ind - (Ny - 1) * num_x + 1;
        float z0 = zvals[num_y * (Nx + 1) + num_x];
        z1 = zvals[(num_y + 1) * (Nx + 1) + num_x] - z0;
        z2 = zvals[num_y * (Nx + 1) + (num_x + 1)] - z0;
        z3 = zvals[(num_y + 1) * (Nx + 1) + (num_x + 1)] - z0;
        z4 = zvals[(num_y - 1) * (Nx + 1) + num_x] - z0;
        z5 = zvals[(num_y - 1) * (Nx + 1) + (num_x + 1)] - z0;
    }
    float z0 = 0;
    float n00 = 0, n01 = 0, n02 = 0, n03 = 0, n10 = 0, n11 = 0, n12 = 0, n13 = 0, n20 = 0, n21 = 0, n22 = 0, n23 = 0, n30 = 0, n31 = 0, n32 = 0, n33 = 0, div = 0;
    float z11_1 = (z4 - z0 - z5 + z2) / 4;
    float z10_1 = (-z4 + z0 - z5 + z2) / 4;
    float z01_1 = (-z4 - z0 + z5 + z2) / 4;
    float z11_2 = (z0 - z1 - z2 + z3) / 4;
    float z10_2 = (-z0 + z1 - z2 + z3) / 4;
    float z01_2 = (-z0 - z1 + z2 + z3) / 4;
    for (int v_out = 0; v_out < order2; v_out++) {
        float v0 = p2[v_out];
        float weight0 = w2[v_out];
        for (int ind_t = 0; ind_t < order2; ind_t++) {
            for (int ind_t1 = 0; ind_t1 < order2; ind_t1++) {
                for (int ind_t2 = 0; ind_t2 < order2; ind_t2++) {
                    float rt = p2[ind_t] * 0.5f + 0.5f;
                    float ct = rt + 1;
                    float rt1 = p2[ind_t1] * 0.5f + 0.5f;
                    float ct1 = p2[ind_t1] * (2 / ct - 1) / 2 + 0.5f;
                    float t2 = p2[ind_t2] * 0.5f + 0.5f;
                    float rweight = w2[ind_t] * w2[ind_t1] * w2[ind_t2] / 8;
                    float cweight = rweight * (2 / ct - 1);
                    float ru_out = 1 - 2 * rt * rt1;
                    float cu_out = 1 - 2 * ct * ct1;
                    float rdist_x1 = d * (rt1 * t2 - rt1 - t2);
                    float rdist_y1 = -d * (1 - v0) * (1 - rt1) / 2;
                    float rdist_z1 = -z0 / 2 * (v0 - 1) * ((1 - rt1) * t2 + (1 - 2 * rt1) - rt * (1 - rt1) * (1 - rt1) * t2) + z1 / 2 * t2 * (1 - rt1) * (v0 - 1) * (1 - rt + rt * rt1) + z2 / 2 * ((v0 + 1) * (t2 - rt1 - rt1 * t2) + (1 - v0) * (rt * (1 - rt1) * (1 - rt1) * t2 - 1 + rt1)) - z3 / 2 * t2 * (1 - rt1) * (v0 + 1 + (1 - v0) * (1 - rt1) * rt) - z4 / 2 * rt1 * (v0 - 1) + z5 / 2 * rt1 * (v0 + 1);
                    float rdist1 = sqrt(rdist_x1 * rdist_x1 + rdist_y1 * rdist_y1 + rdist_z1 * rdist_z1);
                    float cdist_x1 = d * (ct1 * t2 - ct1 - t2);
                    float cdist_y1 = -d * (1 - v0) * (1 - ct1) / 2;
                    float cdist_z1 = -z0 / 2 * (v0 - 1) * ((1 - ct1) * t2 + (1 - 2 * ct1) - ct * (1 - ct1) * (1 - ct1) * t2) + z1 / 2 * t2 * (1 - ct1) * (v0 - 1) * (1 - ct + ct * ct1) + z2 / 2 * ((v0 + 1) * (t2 - ct1 - ct1 * t2) + (1 - v0) * (ct * (1 - ct1) * (1 - ct1) * t2 - 1 + ct1)) - z3 / 2 * t2 * (1 - ct1) * (v0 + 1 + (1 - v0) * (1 - ct1) * ct) - z4 / 2 * ct1 * (v0 - 1) + z5 / 2 * ct1 * (v0 + 1);
                    float cdist1 = sqrt(cdist_x1 * cdist_x1 + cdist_y1 * cdist_y1 + cdist_z1 * cdist_z1);
                    float ru_in1 = 2 * rt * t2 * (1 - rt1) - 1;
                    float rv_in1 = v0 + (1 - v0) * rt * (1 - rt1);
                    float cu_in1 = 2 * ct * t2 * (1 - ct1) - 1;
                    float cv_in1 = v0 + (1 - v0) * ct * (1 - ct1);
                    float rL1 = weight0 * rweight * rt * (1 - rt1) * (1 - v0) / (M_PI * rdist1);
                    float rL11 = 0.25f * d * d + (z11_1 * v0 + z10_1) * (z11_2 * rv_in1 + z10_2);
                    float rL12 = (z11_1 * v0 + z10_1) * (z11_2 * ru_in1 + z01_2);
                    float rL13 = (z11_1 * ru_out + z01_1) * (z11_2 * rv_in1 + z10_2);
                    float rL14 = 0.25f * d * d + (z11_1 * ru_out + z01_1) * (z11_2 * ru_in1 + z01_2);
                    float cL1 = weight0 * cweight * ct * (1 - ct1) * (1 - v0) / (M_PI * cdist1);
                    float cL11 = 0.25f * d * d + (z11_1 * v0 + z10_1) * (z11_2 * cv_in1 + z10_2);
                    float cL12 = (z11_1 * v0 + z10_1) * (z11_2 * cu_in1 + z01_2);
                    float cL13 = (z11_1 * cu_out + z01_1) * (z11_2 * cv_in1 + z10_2);
                    float cL14 = 0.25f * d * d + (z11_1 * cu_out + z01_1) * (z11_2 * cu_in1 + z01_2);
                    n00 += (1 - ru_out) * (1 - ru_in1) * rL11 * rL1 + (1 - cu_out) * (1 - cu_in1) * cL11 * cL1;
                    n01 += (1 - ru_out) * (1 + ru_in1) * rL11 * rL1 + (1 - cu_out) * (1 + cu_in1) * cL11 * cL1;
                    n02 += (1 - ru_out) * (1 - rv_in1) * rL12 * rL1 + (1 - cu_out) * (1 - cv_in1) * cL12 * cL1;
                    n03 += (1 - ru_out) * (1 + rv_in1) * rL12 * rL1 + (1 - cu_out) * (1 + cv_in1) * cL12 * cL1;
                    n10 += (1 + ru_out) * (1 - ru_in1) * rL11 * rL1 + (1 + cu_out) * (1 - cu_in1) * cL11 * cL1;
                    n11 += (1 + ru_out) * (1 + ru_in1) * rL11 * rL1 + (1 + cu_out) * (1 + cu_in1) * cL11 * cL1;
                    n12 += (1 + ru_out) * (1 - rv_in1) * rL12 * rL1 + (1 + cu_out) * (1 - cv_in1) * cL12 * cL1;
                    n13 += (1 + ru_out) * (1 + rv_in1) * rL12 * rL1 + (1 + cu_out) * (1 + cv_in1) * cL12 * cL1;
                    n20 += (1 - v0) * (1 - ru_in1) * rL13 * rL1 + (1 - v0) * (1 - cu_in1) * cL13 * cL1;
                    n21 += (1 - v0) * (1 + ru_in1) * rL13 * rL1 + (1 - v0) * (1 + cu_in1) * cL13 * cL1;
                    n22 += (1 - v0) * (1 - rv_in1) * rL14 * rL1 + (1 - v0) * (1 - cv_in1) * cL14 * cL1;
                    n23 += (1 - v0) * (1 + rv_in1) * rL14 * rL1 + (1 - v0) * (1 + cv_in1) * cL14 * cL1;
                    n30 += (1 + v0) * (1 - ru_in1) * rL13 * rL1 + (1 + v0) * (1 - cu_in1) * cL13 * cL1;
                    n31 += (1 + v0) * (1 + ru_in1) * rL13 * rL1 + (1 + v0) * (1 + cu_in1) * cL13 * cL1;
                    n32 += (1 + v0) * (1 - rv_in1) * rL14 * rL1 + (1 + v0) * (1 - cv_in1) * cL14 * cL1;
                    n33 += (1 + v0) * (1 + rv_in1) * rL14 * rL1 + (1 + v0) * (1 + cv_in1) * cL14 * cL1;
                    float rdist_x2 = -d;
                    float rdist_y2 = d * (1 - rt1) * (v0 - 2 * t2 + 1) / 2;
                    float rdist_z2 = -z0 / 2 * (rt * (1 - rt1) * (1 - rt1) * (2 * t2 - 1 - v0) + (1 - rt1) * (2 * v0 - 2 * t2) + rt1 * (1 - v0)) + z1 / 2 * (1 - rt1) * ((1 - rt * (1 - rt1)) * v0 + rt * (1 - rt1) * (2 * t2 - 1) - 1) + z2 / 2 * (rt * (1 - rt1) * (1 - rt1) * (2 * t2 - v0 - 1) + (1 - rt1) * (2 * v0 - 2 * t2 + 2) - rt1 * (v0 + 1)) - z3 / 2 * (1 - rt1) * ((1 - rt * (1 - rt1)) * v0 + rt * (1 - rt1) * (2 * t2 - 1) + 1) - z4 / 2 * rt1 * (v0 - 1) + z5 / 2 * rt1 * (v0 + 1);
                    float rdist2 = sqrt(rdist_x2 * rdist_x2 + rdist_y2 * rdist_y2 + rdist_z2 * rdist_z2);
                    float cdist_x2 = -d;
                    float cdist_y2 = d * (1 - ct1) * (v0 - 2 * t2 + 1) / 2;
                    float cdist_z2 = -z0 / 2 * (ct * (1 - ct1) * (1 - ct1) * (2 * t2 - 1 - v0) + (1 - ct1) * (2 * v0 - 2 * t2) + ct1 * (1 - v0)) + z1 / 2 * (1 - ct1) * ((1 - ct * (1 - ct1)) * v0 + ct * (1 - ct1) * (2 * t2 - 1) - 1) + z2 / 2 * (ct * (1 - ct1) * (1 - ct1) * (2 * t2 - v0 - 1) + (1 - ct1) * (2 * v0 - 2 * t2 + 2) - ct1 * (v0 + 1)) - z3 / 2 * (1 - ct1) * ((1 - ct * (1 - ct1)) * v0 + ct * (1 - ct1) * (2 * t2 - 1) + 1) - z4 / 2 * ct1 * (v0 - 1) + z5 / 2 * ct1 * (v0 + 1);
                    float cdist2 = sqrt(cdist_x2 * cdist_x2 + cdist_y2 * cdist_y2 + cdist_z2 * cdist_z2);
                    float ru_in2 = -1 + 2 * rt * (1 - rt1);
                    float rv_in2 = (1 - rt * (1 - rt1)) * v0 + (2 * t2 - 1) * rt * (1 - rt1);
                    float cu_in2 = -1 + 2 * ct * (1 - ct1);
                    float cv_in2 = (1 - ct * (1 - ct1)) * v0 + (2 * t2 - 1) * ct * (1 - ct1);
                    float rL2 = 2 * weight0 * rweight * rt * (1 - rt1) / (M_PI * rdist2);
                    float rL21 = 0.25f * d * d + (z11_1 * v0 + z10_1) * (z11_2 * rv_in2 + z10_2);
                    float rL22 = (z11_1 * v0 + z10_1) * (z11_2 * ru_in2 + z01_2);
                    float rL23 = (z11_1 * ru_out + z01_1) * (z11_2 * rv_in2 + z10_2);
                    float rL24 = 0.25f * d * d + (z11_1 * ru_out + z01_1) * (z11_2 * ru_in2 + z01_2);
                    float cL2 = 2 * weight0 * cweight * ct * (1 - ct1) / (M_PI * cdist2);
                    float cL21 = 0.25f * d * d + (z11_1 * v0 + z10_1) * (z11_2 * cv_in2 + z10_2);
                    float cL22 = (z11_1 * v0 + z10_1) * (z11_2 * cu_in2 + z01_2);
                    float cL23 = (z11_1 * cu_out + z01_1) * (z11_2 * cv_in2 + z10_2);
                    float cL24 = 0.25f * d * d + (z11_1 * cu_out + z01_1) * (z11_2 * cu_in2 + z01_2);
                    n00 += (1 - ru_out) * (1 - ru_in2) * rL21 * rL2 + (1 - cu_out) * (1 - cu_in2) * cL21 * cL2;
                    n01 += (1 - ru_out) * (1 + ru_in2) * rL21 * rL2 + (1 - cu_out) * (1 + cu_in2) * cL21 * cL2;
                    n02 += (1 - ru_out) * (1 - rv_in2) * rL22 * rL2 + (1 - cu_out) * (1 - cv_in2) * cL22 * cL2;
                    n03 += (1 - ru_out) * (1 + rv_in2) * rL22 * rL2 + (1 - cu_out) * (1 + cv_in2) * cL22 * cL2;
                    n10 += (1 + ru_out) * (1 - ru_in2) * rL21 * rL2 + (1 + cu_out) * (1 - cu_in2) * cL21 * cL2;
                    n11 += (1 + ru_out) * (1 + ru_in2) * rL21 * rL2 + (1 + cu_out) * (1 + cu_in2) * cL21 * cL2;
                    n12 += (1 + ru_out) * (1 - rv_in2) * rL22 * rL2 + (1 + cu_out) * (1 - cv_in2) * cL22 * cL2;
                    n13 += (1 + ru_out) * (1 + rv_in2) * rL22 * rL2 + (1 + cu_out) * (1 + cv_in2) * cL22 * cL2;
                    n20 += (1 - v0) * (1 - ru_in2) * rL23 * rL2 + (1 - v0) * (1 - cu_in2) * cL23 * cL2;
                    n21 += (1 - v0) * (1 + ru_in2) * rL23 * rL2 + (1 - v0) * (1 + cu_in2) * cL23 * cL2;
                    n22 += (1 - v0) * (1 - rv_in2) * rL24 * rL2 + (1 - v0) * (1 - cv_in2) * cL24 * cL2;
                    n23 += (1 - v0) * (1 + rv_in2) * rL24 * rL2 + (1 - v0) * (1 + cv_in2) * cL24 * cL2;
                    n30 += (1 + v0) * (1 - ru_in2) * rL23 * rL2 + (1 + v0) * (1 - cu_in2) * cL23 * cL2;
                    n31 += (1 + v0) * (1 + ru_in2) * rL23 * rL2 + (1 + v0) * (1 + cu_in2) * cL23 * cL2;
                    n32 += (1 + v0) * (1 - rv_in2) * rL24 * rL2 + (1 + v0) * (1 - cv_in2) * cL24 * cL2;
                    n33 += (1 + v0) * (1 + rv_in2) * rL24 * rL2 + (1 + v0) * (1 + cv_in2) * cL24 * cL2;
                    float rdist_x3 = d * (rt1 * t2 - rt1 - t2);
                    float rdist_y3 = d * (1 + v0) * (1 - rt1) / 2;
                    float rdist_z3 = -z0 / 2 * ((1 - rt1) * (t2 * v0 - t2 + v0 + 1) - rt1 * (v0 - 1) - rt * (1 - rt1) * (1 - rt1) * t2 * (1 + v0)) + z1 / 2 * t2 * (1 - rt1) * (v0 - 1 - (1 + v0) * (1 - rt1) * rt) + z2 / 2 * (v0 + 1) * ((1 - rt1) * (1 + t2) - rt1 - rt * (1 - rt1) * (1 - rt1) * t2) - z3 / 2 * t2 * (1 - rt1) * (v0 + 1) * (1 - rt + rt * rt1) - z4 / 2 * rt1 * (v0 - 1) + z5 / 2 * rt1 * (v0 + 1);
                    float rdist3 = sqrt(rdist_x3 * rdist_x3 + rdist_y3 * rdist_y3 + rdist_z3 * rdist_z3);
                    float cdist_x3 = d * (ct1 * t2 - ct1 - t2);
                    float cdist_y3 = d * (1 + v0) * (1 - ct1) / 2;
                    float cdist_z3 = -z0 / 2 * ((1 - ct1) * (t2 * v0 - t2 + v0 + 1) - ct1 * (v0 - 1) - ct * (1 - ct1) * (1 - ct1) * t2 * (1 + v0)) + z1 / 2 * t2 * (1 - ct1) * (v0 - 1 - (1 + v0) * (1 - ct1) * ct) + z2 / 2 * (v0 + 1) * ((1 - ct1) * (1 + t2) - ct1 - ct * (1 - ct1) * (1 - ct1) * t2) - z3 / 2 * t2 * (1 - ct1) * (v0 + 1) * (1 - ct + ct * ct1) - z4 / 2 * ct1 * (v0 - 1) + z5 / 2 * ct1 * (v0 + 1);
                    float cdist3 = sqrt(cdist_x3 * cdist_x3 + cdist_y3 * cdist_y3 + cdist_z3 * cdist_z3);
                    float ru_in3 = -1 + 2 * rt * t2 * (1 - rt1);
                    float rv_in3 = v0 - rt * (1 - rt1) * (1 + v0);
                    float cu_in3 = -1 + 2 * ct * t2 * (1 - ct1);
                    float cv_in3 = v0 - ct * (1 - ct1) * (1 + v0);
                    float rL3 = weight0 * rweight * rt * (1 - rt1) * (1 + v0) / (M_PI * rdist3);
                    float rL31 = 0.25f * d * d + (z11_1 * v0 + z10_1) * (z11_2 * rv_in3 + z10_2);
                    float rL32 = (z11_1 * v0 + z10_1) * (z11_2 * ru_in3 + z01_2);
                    float rL33 = (z11_1 * ru_out + z01_1) * (z11_2 * rv_in3 + z10_2);
                    float rL34 = 0.25f * d * d + (z11_1 * ru_out + z01_1) * (z11_2 * ru_in3 + z01_2);
                    float cL3 = weight0 * cweight * ct * (1 - ct1) * (1 + v0) / (M_PI * cdist3);
                    float cL31 = 0.25f * d * d + (z11_1 * v0 + z10_1) * (z11_2 * cv_in3 + z10_2);
                    float cL32 = (z11_1 * v0 + z10_1) * (z11_2 * cu_in3 + z01_2);
                    float cL33 = (z11_1 * cu_out + z01_1) * (z11_2 * cv_in3 + z10_2);
                    float cL34 = 0.25f * d * d + (z11_1 * cu_out + z01_1) * (z11_2 * cu_in3 + z01_2);
                    n00 += (1 - ru_out) * (1 - ru_in3) * rL31 * rL3 + (1 - cu_out) * (1 - cu_in3) * cL31 * cL3;
                    n01 += (1 - ru_out) * (1 + ru_in3) * rL31 * rL3 + (1 - cu_out) * (1 + cu_in3) * cL31 * cL3;
                    n02 += (1 - ru_out) * (1 - rv_in3) * rL32 * rL3 + (1 - cu_out) * (1 - cv_in3) * cL32 * cL3;
                    n03 += (1 - ru_out) * (1 + rv_in3) * rL32 * rL3 + (1 - cu_out) * (1 + cv_in3) * cL32 * cL3;
                    n10 += (1 + ru_out) * (1 - ru_in3) * rL31 * rL3 + (1 + cu_out) * (1 - cu_in3) * cL31 * cL3;
                    n11 += (1 + ru_out) * (1 + ru_in3) * rL31 * rL3 + (1 + cu_out) * (1 + cu_in3) * cL31 * cL3;
                    n12 += (1 + ru_out) * (1 - rv_in3) * rL32 * rL3 + (1 + cu_out) * (1 - cv_in3) * cL32 * cL3;
                    n13 += (1 + ru_out) * (1 + rv_in3) * rL32 * rL3 + (1 + cu_out) * (1 + cv_in3) * cL32 * cL3;
                    n20 += (1 - v0) * (1 - ru_in3) * rL33 * rL3 + (1 - v0) * (1 - cu_in3) * cL33 * cL3;
                    n21 += (1 - v0) * (1 + ru_in3) * rL33 * rL3 + (1 - v0) * (1 + cu_in3) * cL33 * cL3;
                    n22 += (1 - v0) * (1 - rv_in3) * rL34 * rL3 + (1 - v0) * (1 - cv_in3) * cL34 * cL3;
                    n23 += (1 - v0) * (1 + rv_in3) * rL34 * rL3 + (1 - v0) * (1 + cv_in3) * cL34 * cL3;
                    n30 += (1 + v0) * (1 - ru_in3) * rL33 * rL3 + (1 + v0) * (1 - cu_in3) * cL33 * cL3;
                    n31 += (1 + v0) * (1 + ru_in3) * rL33 * rL3 + (1 + v0) * (1 + cu_in3) * cL33 * cL3;
                    n32 += (1 + v0) * (1 - rv_in3) * rL34 * rL3 + (1 + v0) * (1 - cv_in3) * cL34 * cL3;
                    n33 += (1 + v0) * (1 + rv_in3) * rL34 * rL3 + (1 + v0) * (1 + cv_in3) * cL34 * cL3;
                    div += rL1 + cL1 + rL2 + cL2 + rL3 + cL3;
                }
            }
        }
    }
    neighbor[17 * i + 0] = n00;
    neighbor[17 * i + 1] = n01;
    neighbor[17 * i + 2] = n02;
    neighbor[17 * i + 3] = n03;
    neighbor[17 * i + 4] = n10;
    neighbor[17 * i + 5] = n11;
    neighbor[17 * i + 6] = n12;
    neighbor[17 * i + 7] = n13;
    neighbor[17 * i + 8] = n20;
    neighbor[17 * i + 9] = n21;
    neighbor[17 * i + 10] = n22;
    neighbor[17 * i + 11] = n23;
    neighbor[17 * i + 12] = n30;
    neighbor[17 * i + 13] = n31;
    neighbor[17 * i + 14] = n32;
    neighbor[17 * i + 15] = n33;
    neighbor[17 * i + 16] = div;
}