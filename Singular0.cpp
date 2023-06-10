/* CPU implementation for computing elements in the BEM matrix with singularities in the underlying integrals.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "Singular0.h"

/// @brief CPU implementation for computing elements in the BEM matrix with singularities in the underlying integrals
/// @param est: an Estimate object
Singular0::Singular0(Estimate* est): Singular(est) {
    computeInvariants();
}

/// @brief Driver code for computing all the singular matrix elements
void Singular0::computeInvariants() {
    selfEE();
    neighborEE();
    neighborEM();
}

/// @brief Compute elements in the diagonal blocks of the BEM matrix that involve overlapping basis element pairs
void Singular0::selfEE() {
    Matrix4f div_pattern;
    div_pattern << -1.0, 1.0, -1.0, 1.0,
                    0.0, -1.0, 1.0, -1.0,
                    0.0, 0.0, -1.0, 1.0,
                    0.0, 0.0, 0.0, -1.0;
    LS1 = MatrixXf::Zero(4 * Nx, 4 * Ny);
    LS2 = MatrixXf::Zero(4 * Nx, 4 * Ny);

    // Loop through all the basis elements and integrate over each element twice
    parallel_for(Nx, [&](int start, int end) {
    for (int num_x = start; num_x < end; num_x++) {
        for (int num_y = 0; num_y < Ny; num_y++) {
            float z1 = zvals(num_x + 1, num_y) - zvals(num_x, num_y);
            float z2 = zvals(num_x, num_y + 1) - zvals(num_x, num_y);
            float z3 = zvals(num_x + 1, num_y + 1) - zvals(num_x, num_y);
            float div = 0;
            LS1.block(4 * num_x, 4 * num_y, 4, 4) = computeSelfEE(z1, z2, z3, div);
            LS2.block(4 * num_x, 4 * num_y, 4, 4) = div * div_pattern;
        }
    }
    } );
}

/// @brief Compute elements in the diagonal blocks of the BEM matrix that involve neighboring basis element pairs
void Singular0::neighborEE() {
    Matrix4f div_pattern;
    div_pattern << -1.0, 1.0, -1.0, 1.0,
                    1.0, -1.0, 1.0, -1.0,
                    -1.0, 1.0, -1.0, 1.0,
                    1.0, -1.0, 1.0, -1.0;
    
    // Matrix elements involving horizontally neighboring basis elements
    LH1 = MatrixXf::Zero(4 * (Nx - 1), 4 * Ny);
    LH2 = MatrixXf::Zero(4 * (Nx - 1), 4 * Ny);
    parallel_for(Ny, [&](int start, int end) {
    for (int num_y = start; num_y < end; num_y++) {
        for (int num_x = 1; num_x < Nx; num_x++) {
            float z1 = zvals(num_x + 1, num_y) - zvals(num_x, num_y);
            float z2 = zvals(num_x, num_y + 1) - zvals(num_x, num_y);
            float z3 = zvals(num_x + 1, num_y + 1) - zvals(num_x, num_y);
            float z4 = zvals(num_x - 1, num_y) - zvals(num_x, num_y);
            float z5 = zvals(num_x - 1, num_y + 1) - zvals(num_x, num_y);
            float div = 0;
            LH1.block(4 * (num_x - 1), 4 * num_y, 4, 4) = computeNeighborEE(z1, z2, z3, z4, z5, div);
            LH2.block(4 * (num_x - 1), 4 * num_y, 4, 4) = div * div_pattern;
        }
    }
    } );
    
    // Matrix elements involving vertically neighboring basis elements
    LV1 = MatrixXf::Zero(4 * Nx, 4 * (Ny - 1));
    LV2 = MatrixXf::Zero(4 * Nx, 4 * (Ny - 1));
    parallel_for(Nx, [&](int start, int end) {
    for (int num_x = start; num_x < end; num_x++) {
        for (int num_y = 1; num_y < Ny; num_y++) {
            float z1 = zvals(num_x, num_y + 1) - zvals(num_x, num_y);
            float z2 = zvals(num_x + 1, num_y) - zvals(num_x, num_y);
            float z3 = zvals(num_x + 1, num_y + 1) - zvals(num_x, num_y);
            float z4 = zvals(num_x, num_y - 1) - zvals(num_x, num_y);
            float z5 = zvals(num_x + 1, num_y - 1) - zvals(num_x, num_y);
            float div = 0;
            LV1.block(4 * num_x, 4 * (num_y - 1), 4, 4) = computeNeighborEE(z1, z2, z3, z4, z5, div);
            LV2.block(4 * num_x, 4 * (num_y - 1), 4, 4) = div * div_pattern;
        }
    }
    } );
}

/// @brief Compute individual matrix elements through numerical integration after change of coordinates
/// @brief Nontrivial change of coordinates computations are involved; users should not try to understand or modify the code
/// @param z1, z2, z3: height values from the considered basis element
/// @param div: stores a computed (result) value associated with the computed matrix elements
/// @return computed matrix elements
MatrixXf Singular0::computeSelfEE(float z1, float z2, float z3, float& div) {
    MatrixXf block = MatrixXf::Zero(4, 4);
    div = 0;
    float z11 = (z3 - z1 - z2) / 4;
    float z10 = (z1 - z2 + z3) / 4;
    float z01 = (z2 + z3 - z1) / 4;
    for (int out_u = 0; out_u < order1; out_u++) {
        float u0 = p1(out_u);
        for (int out_v = 0; out_v < order1; out_v++) {
            float v0 = p1(out_v);
            float out_weight = w1(out_u) * w1(out_v);
            float A = z11 * v0 + z10;
            float B = z11 * u0 + z01;
            for (int in_1 = 0; in_1 < order1; in_1++) {
                for (int in_2 = 0; in_2 < order1; in_2++) {
                    float t1 = 0.5 * p1(in_1) + 0.5;
                    float t2 = 0.5 * p1(in_2) + 0.5;
                    float weight = out_weight * w1(in_1) * w1(in_2) / 4;
                    float base1 = weight / (2 * M_PI) * (v0 + 1) / sqrt(0.25 * d * d * pow(2 * t1 - 1 - u0, 2) + 0.25 * d * d * pow(-1 - v0, 2) + pow(z11 * (2 * t1 - 1 - u0) * (-1 - v0) * (1 - t2) + A * (2 * t1 - 1 - u0) + B * (-1 - v0), 2));
                    float u1 = t2 * u0 + t2 - 1 + 2 * (1 - t2) * t1;
                    float v1 = t2 * (v0 + 1) - 1;
                    float common11 = 0.25 * d * d + (z11 * v0 + z10) * (z11 * v1 + z10);
                    float common12 = (z11 * v0 + z10) * (z11 * u1 + z01);
                    float common13 = 0.25 * d * d + (z11 * u0 + z01) * (z11 * u1 + z01);
                    block(0, 0) += (1 - u0) * (1 - u1) * common11 * base1;
                    block(0, 1) += (1 - u0) * (1 + u1) * common11 * base1;
                    block(0, 2) += (1 - u0) * (1 - v1) * common12 * base1;
                    block(0, 3) += (1 - u0) * (1 + v1) * common12 * base1;
                    block(1, 1) += (1 + u0) * (1 + u1) * common11 * base1;
                    block(1, 2) += (1 + u0) * (1 - v1) * common12 * base1;
                    block(1, 3) += (1 + u0) * (1 + v1) * common12 * base1;
                    block(2, 2) += (1 - v0) * (1 - v1) * common13 * base1;
                    block(2, 3) += (1 - v0) * (1 + v1) * common13 * base1;
                    block(3, 3) += (1 + v0) * (1 + v1) * common13 * base1;
                    float base2 = weight / (2 * M_PI) * (1 - u0) / sqrt(0.25 * d * d * pow(1 - u0, 2) + 0.25 * d * d * pow(2 * t2 - 1 - v0, 2) + pow(z11 * (1 - u0) * (2 * t2 - v0 - 1) * t1 + A * (1 - u0) + B * (2 * t2 - 1 - v0), 2));
                    float u2 = (1 - t1) * u0 + t1;
                    float v2 = (1 - t1) * v0 + 2 * t1 * t2 - t1;
                    float common21 = 0.25 * d * d + (z11 * v0 + z10) * (z11 * v2 + z10);
                    float common22 = (z11 * v0 + z10) * (z11 * u2 + z01);
                    float common23 = 0.25 * d * d + (z11 * u0 + z01) * (z11 * u2 + z01);
                    block(0, 0) += (1 - u0) * (1 - u2) * common21 * base2;
                    block(0, 1) += (1 - u0) * (1 + u2) * common21 * base2;
                    block(0, 2) += (1 - u0) * (1 - v2) * common22 * base2;
                    block(0, 3) += (1 - u0) * (1 + v2) * common22 * base2;
                    block(1, 1) += (1 + u0) * (1 + u2) * common21 * base2;
                    block(1, 2) += (1 + u0) * (1 - v2) * common22 * base2;
                    block(1, 3) += (1 + u0) * (1 + v2) * common22 * base2;
                    block(2, 2) += (1 - v0) * (1 - v2) * common23 * base2;
                    block(2, 3) += (1 - v0) * (1 + v2) * common23 * base2;
                    block(3, 3) += (1 + v0) * (1 + v2) * common23 * base2;
                    float base3 = weight / (2 * M_PI) * (1 - v0) / sqrt(0.25 * d * d * pow(2 * t1 - u0 - 1, 2) + 0.25 * d * d * pow(1 - v0, 2) + pow(z11 * (2 * t1 - u0 - 1) * (1 - v0) * t2 + A * (2 * t1 - u0 - 1) + B * (1 - v0), 2));
                    float u3 = (1 - t2) * u0 + 2 * t1 * t2 - t2;
                    float v3 = (1 - t2) * v0 + t2;
                    float common31 = 0.25 * d * d + (z11 * v0 + z10) * (z11 * v3 + z10);
                    float common32 = (z11 * v0 + z10) * (z11 * u3 + z01);
                    float common33 = 0.25 * d * d + (z11 * u0 + z01) * (z11 * u3 + z01);
                    block(0, 0) += (1 - u0) * (1 - u3) * common31 * base3;
                    block(0, 1) += (1 - u0) * (1 + u3) * common31 * base3;
                    block(0, 2) += (1 - u0) * (1 - v3) * common32 * base3;
                    block(0, 3) += (1 - u0) * (1 + v3) * common32 * base3;
                    block(1, 1) += (1 + u0) * (1 + u3) * common31 * base3;
                    block(1, 2) += (1 + u0) * (1 - v3) * common32 * base3;
                    block(1, 3) += (1 + u0) * (1 + v3) * common32 * base3;
                    block(2, 2) += (1 - v0) * (1 - v3) * common33 * base3;
                    block(2, 3) += (1 - v0) * (1 + v3) * common33 * base3;
                    block(3, 3) += (1 + v0) * (1 + v3) * common33 * base3;
                    float base4 = weight / (2 * M_PI) * (1 + u0) / sqrt(0.25 * d * d * pow(-1 - u0, 2) + 0.25 * d * d * pow(2 * t2 - v0 - 1, 2) + pow(z11 * (-1 - u0) * (2 * t2 - v0 - 1) * (1 - t1) + A * (-1 - u0) + B * (2 * t2 - v0 - 1), 2));
                    float u4 = t1 * u0 - (1 - t1);
                    float v4 = t1 * v0 - 1 + t1 + 2 * t2 - 2 * t1 * t2;
                    float common41 = 0.25 * d * d + (z11 * v0 + z10) * (z11 * v4 + z10);
                    float common42 = (z11 * v0 + z10) * (z11 * u4 + z01);
                    float common43 = 0.25 * d * d + (z11 * u0 + z01) * (z11 * u4 + z01);
                    block(0, 0) += (1 - u0) * (1 - u4) * common41 * base4;
                    block(0, 1) += (1 - u0) * (1 + u4) * common41 * base4;
                    block(0, 2) += (1 - u0) * (1 - v4) * common42 * base4;
                    block(0, 3) += (1 - u0) * (1 + v4) * common42 * base4;
                    block(1, 1) += (1 + u0) * (1 + u4) * common41 * base4;
                    block(1, 2) += (1 + u0) * (1 - v4) * common42 * base4;
                    block(1, 3) += (1 + u0) * (1 + v4) * common42 * base4;
                    block(2, 2) += (1 - v0) * (1 - v4) * common43 * base4;
                    block(2, 3) += (1 - v0) * (1 + v4) * common43 * base4;
                    block(3, 3) += (1 + v0) * (1 + v4) * common43 * base4;
                    div += base1 + base2 + base3 + base4;
                }
            }
        }
    }
    return block;
}

/// @brief Compute individual matrix elements through numerical integration after change of coordinates
/// @brief Nontrivial change of coordinates computations are involved; users should not try to understand or modify the code
/// @param z1, z2, z3, z4, z5: height values from the considered basis element pair
/// @param div: stores a computed (result) value associated with the computed matrix elements
/// @return computed matrix elements
MatrixXf Singular0::computeNeighborEE(float z1, float z2, float z3, float z4, float z5, float& div) {
    MatrixXf block = MatrixXf::Zero(4, 4);
    div = 0;
    float z0 = 0;
    float z11_1 = (z4 - z0 - z5 + z2) / 4;
    float z10_1 = (-z4 + z0 - z5 + z2) / 4;
    float z01_1 = (-z4 - z0 + z5 + z2) / 4;
    float z11_2 = (z0 - z1 - z2 + z3) / 4;
    float z10_2 = (-z0 + z1 - z2 + z3) / 4;
    float z01_2 = (-z0 - z1 + z2 + z3) / 4;
    for (int v_out = 0; v_out < order2; v_out++) {
        float v0 = p2(v_out);
        float weight0 = w2(v_out);
        for (int ind_t = 0; ind_t < order2; ind_t++) {
            for (int ind_t1 = 0; ind_t1 < order2; ind_t1++) {
                for (int ind_t2 = 0; ind_t2 < order2; ind_t2++) {
                    float rt = p2(ind_t) * 0.5 + 0.5;
                    float ct = rt + 1;
                    float rt1 = p2(ind_t1) * 0.5 + 0.5;
                    float ct1 = p2(ind_t1) * (2 / ct - 1) / 2 + 0.5;
                    float t2 = p2(ind_t2) * 0.5 + 0.5;
                    float rweight = w2(ind_t) * w2(ind_t1) * w2(ind_t2) / 8;
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
                    float rL11 = 0.25 * d * d + (z11_1 * v0 + z10_1) * (z11_2 * rv_in1 + z10_2);
                    float rL12 = (z11_1 * v0 + z10_1) * (z11_2 * ru_in1 + z01_2);
                    float rL13 = (z11_1 * ru_out + z01_1) * (z11_2 * rv_in1 + z10_2);
                    float rL14 = 0.25 * d * d + (z11_1 * ru_out + z01_1) * (z11_2 * ru_in1 + z01_2);
                    float cL1 = weight0 * cweight * ct * (1 - ct1) * (1 - v0) / (M_PI * cdist1);
                    float cL11 = 0.25 * d * d + (z11_1 * v0 + z10_1) * (z11_2 * cv_in1 + z10_2);
                    float cL12 = (z11_1 * v0 + z10_1) * (z11_2 * cu_in1 + z01_2);
                    float cL13 = (z11_1 * cu_out + z01_1) * (z11_2 * cv_in1 + z10_2);
                    float cL14 = 0.25 * d * d + (z11_1 * cu_out + z01_1) * (z11_2 * cu_in1 + z01_2);
                    block(0, 0) += (1 - ru_out) * (1 - ru_in1) * rL11 * rL1 + (1 - cu_out) * (1 - cu_in1) * cL11 * cL1;
                    block(0, 1) += (1 - ru_out) * (1 + ru_in1) * rL11 * rL1 + (1 - cu_out) * (1 + cu_in1) * cL11 * cL1;
                    block(0, 2) += (1 - ru_out) * (1 - rv_in1) * rL12 * rL1 + (1 - cu_out) * (1 - cv_in1) * cL12 * cL1;
                    block(0, 3) += (1 - ru_out) * (1 + rv_in1) * rL12 * rL1 + (1 - cu_out) * (1 + cv_in1) * cL12 * cL1;
                    block(1, 0) += (1 + ru_out) * (1 - ru_in1) * rL11 * rL1 + (1 + cu_out) * (1 - cu_in1) * cL11 * cL1;
                    block(1, 1) += (1 + ru_out) * (1 + ru_in1) * rL11 * rL1 + (1 + cu_out) * (1 + cu_in1) * cL11 * cL1;
                    block(1, 2) += (1 + ru_out) * (1 - rv_in1) * rL12 * rL1 + (1 + cu_out) * (1 - cv_in1) * cL12 * cL1;
                    block(1, 3) += (1 + ru_out) * (1 + rv_in1) * rL12 * rL1 + (1 + cu_out) * (1 + cv_in1) * cL12 * cL1;
                    block(2, 0) += (1 - v0) * (1 - ru_in1) * rL13 * rL1 + (1 - v0) * (1 - cu_in1) * cL13 * cL1;
                    block(2, 1) += (1 - v0) * (1 + ru_in1) * rL13 * rL1 + (1 - v0) * (1 + cu_in1) * cL13 * cL1;
                    block(2, 2) += (1 - v0) * (1 - rv_in1) * rL14 * rL1 + (1 - v0) * (1 - cv_in1) * cL14 * cL1;
                    block(2, 3) += (1 - v0) * (1 + rv_in1) * rL14 * rL1 + (1 - v0) * (1 + cv_in1) * cL14 * cL1;
                    block(3, 0) += (1 + v0) * (1 - ru_in1) * rL13 * rL1 + (1 + v0) * (1 - cu_in1) * cL13 * cL1;
                    block(3, 1) += (1 + v0) * (1 + ru_in1) * rL13 * rL1 + (1 + v0) * (1 + cu_in1) * cL13 * cL1;
                    block(3, 2) += (1 + v0) * (1 - rv_in1) * rL14 * rL1 + (1 + v0) * (1 - cv_in1) * cL14 * cL1;
                    block(3, 3) += (1 + v0) * (1 + rv_in1) * rL14 * rL1 + (1 + v0) * (1 + cv_in1) * cL14 * cL1;
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
                    float rL21 = 0.25 * d * d + (z11_1 * v0 + z10_1) * (z11_2 * rv_in2 + z10_2);
                    float rL22 = (z11_1 * v0 + z10_1) * (z11_2 * ru_in2 + z01_2);
                    float rL23 = (z11_1 * ru_out + z01_1) * (z11_2 * rv_in2 + z10_2);
                    float rL24 = 0.25 * d * d + (z11_1 * ru_out + z01_1) * (z11_2 * ru_in2 + z01_2);
                    float cL2 = 2 * weight0 * cweight * ct * (1 - ct1) / (M_PI * cdist2);
                    float cL21 = 0.25 * d * d + (z11_1 * v0 + z10_1) * (z11_2 * cv_in2 + z10_2);
                    float cL22 = (z11_1 * v0 + z10_1) * (z11_2 * cu_in2 + z01_2);
                    float cL23 = (z11_1 * cu_out + z01_1) * (z11_2 * cv_in2 + z10_2);
                    float cL24 = 0.25 * d * d + (z11_1 * cu_out + z01_1) * (z11_2 * cu_in2 + z01_2);
                    block(0, 0) += (1 - ru_out) * (1 - ru_in2) * rL21 * rL2 + (1 - cu_out) * (1 - cu_in2) * cL21 * cL2;
                    block(0, 1) += (1 - ru_out) * (1 + ru_in2) * rL21 * rL2 + (1 - cu_out) * (1 + cu_in2) * cL21 * cL2;
                    block(0, 2) += (1 - ru_out) * (1 - rv_in2) * rL22 * rL2 + (1 - cu_out) * (1 - cv_in2) * cL22 * cL2;
                    block(0, 3) += (1 - ru_out) * (1 + rv_in2) * rL22 * rL2 + (1 - cu_out) * (1 + cv_in2) * cL22 * cL2;
                    block(1, 0) += (1 + ru_out) * (1 - ru_in2) * rL21 * rL2 + (1 + cu_out) * (1 - cu_in2) * cL21 * cL2;
                    block(1, 1) += (1 + ru_out) * (1 + ru_in2) * rL21 * rL2 + (1 + cu_out) * (1 + cu_in2) * cL21 * cL2;
                    block(1, 2) += (1 + ru_out) * (1 - rv_in2) * rL22 * rL2 + (1 + cu_out) * (1 - cv_in2) * cL22 * cL2;
                    block(1, 3) += (1 + ru_out) * (1 + rv_in2) * rL22 * rL2 + (1 + cu_out) * (1 + cv_in2) * cL22 * cL2;
                    block(2, 0) += (1 - v0) * (1 - ru_in2) * rL23 * rL2 + (1 - v0) * (1 - cu_in2) * cL23 * cL2;
                    block(2, 1) += (1 - v0) * (1 + ru_in2) * rL23 * rL2 + (1 - v0) * (1 + cu_in2) * cL23 * cL2;
                    block(2, 2) += (1 - v0) * (1 - rv_in2) * rL24 * rL2 + (1 - v0) * (1 - cv_in2) * cL24 * cL2;
                    block(2, 3) += (1 - v0) * (1 + rv_in2) * rL24 * rL2 + (1 - v0) * (1 + cv_in2) * cL24 * cL2;
                    block(3, 0) += (1 + v0) * (1 - ru_in2) * rL23 * rL2 + (1 + v0) * (1 - cu_in2) * cL23 * cL2;
                    block(3, 1) += (1 + v0) * (1 + ru_in2) * rL23 * rL2 + (1 + v0) * (1 + cu_in2) * cL23 * cL2;
                    block(3, 2) += (1 + v0) * (1 - rv_in2) * rL24 * rL2 + (1 + v0) * (1 - cv_in2) * cL24 * cL2;
                    block(3, 3) += (1 + v0) * (1 + rv_in2) * rL24 * rL2 + (1 + v0) * (1 + cv_in2) * cL24 * cL2;
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
                    float rL31 = 0.25 * d * d + (z11_1 * v0 + z10_1) * (z11_2 * rv_in3 + z10_2);
                    float rL32 = (z11_1 * v0 + z10_1) * (z11_2 * ru_in3 + z01_2);
                    float rL33 = (z11_1 * ru_out + z01_1) * (z11_2 * rv_in3 + z10_2);
                    float rL34 = 0.25 * d * d + (z11_1 * ru_out + z01_1) * (z11_2 * ru_in3 + z01_2);
                    float cL3 = weight0 * cweight * ct * (1 - ct1) * (1 + v0) / (M_PI * cdist3);
                    float cL31 = 0.25 * d * d + (z11_1 * v0 + z10_1) * (z11_2 * cv_in3 + z10_2);
                    float cL32 = (z11_1 * v0 + z10_1) * (z11_2 * cu_in3 + z01_2);
                    float cL33 = (z11_1 * cu_out + z01_1) * (z11_2 * cv_in3 + z10_2);
                    float cL34 = 0.25 * d * d + (z11_1 * cu_out + z01_1) * (z11_2 * cu_in3 + z01_2);
                    block(0, 0) += (1 - ru_out) * (1 - ru_in3) * rL31 * rL3 + (1 - cu_out) * (1 - cu_in3) * cL31 * cL3;
                    block(0, 1) += (1 - ru_out) * (1 + ru_in3) * rL31 * rL3 + (1 - cu_out) * (1 + cu_in3) * cL31 * cL3;
                    block(0, 2) += (1 - ru_out) * (1 - rv_in3) * rL32 * rL3 + (1 - cu_out) * (1 - cv_in3) * cL32 * cL3;
                    block(0, 3) += (1 - ru_out) * (1 + rv_in3) * rL32 * rL3 + (1 - cu_out) * (1 + cv_in3) * cL32 * cL3;
                    block(1, 0) += (1 + ru_out) * (1 - ru_in3) * rL31 * rL3 + (1 + cu_out) * (1 - cu_in3) * cL31 * cL3;
                    block(1, 1) += (1 + ru_out) * (1 + ru_in3) * rL31 * rL3 + (1 + cu_out) * (1 + cu_in3) * cL31 * cL3;
                    block(1, 2) += (1 + ru_out) * (1 - rv_in3) * rL32 * rL3 + (1 + cu_out) * (1 - cv_in3) * cL32 * cL3;
                    block(1, 3) += (1 + ru_out) * (1 + rv_in3) * rL32 * rL3 + (1 + cu_out) * (1 + cv_in3) * cL32 * cL3;
                    block(2, 0) += (1 - v0) * (1 - ru_in3) * rL33 * rL3 + (1 - v0) * (1 - cu_in3) * cL33 * cL3;
                    block(2, 1) += (1 - v0) * (1 + ru_in3) * rL33 * rL3 + (1 - v0) * (1 + cu_in3) * cL33 * cL3;
                    block(2, 2) += (1 - v0) * (1 - rv_in3) * rL34 * rL3 + (1 - v0) * (1 - cv_in3) * cL34 * cL3;
                    block(2, 3) += (1 - v0) * (1 + rv_in3) * rL34 * rL3 + (1 - v0) * (1 + cv_in3) * cL34 * cL3;
                    block(3, 0) += (1 + v0) * (1 - ru_in3) * rL33 * rL3 + (1 + v0) * (1 - cu_in3) * cL33 * cL3;
                    block(3, 1) += (1 + v0) * (1 + ru_in3) * rL33 * rL3 + (1 + v0) * (1 + cu_in3) * cL33 * cL3;
                    block(3, 2) += (1 + v0) * (1 - rv_in3) * rL34 * rL3 + (1 + v0) * (1 - cv_in3) * cL34 * cL3;
                    block(3, 3) += (1 + v0) * (1 + rv_in3) * rL34 * rL3 + (1 + v0) * (1 + cv_in3) * cL34 * cL3;
                    div += rL1 + cL1 + rL2 + cL2 + rL3 + cL3;
                }
            }
        }
    }
    return block;
}