/* Super class module for computing elements in the BEM matrix with singularities in the underlying integrals.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "Singular.h"

/// @brief Super class module for computing elements in the BEM matrix with singularities in the underlying integrals
/// @param est: an Estimate object
Singular::Singular(Estimate* est) {
    this->Nx = est->Nx;
    this->Ny = est->Ny;
    this->d = (float)(est->d);
    this->zvals = est->zvals.cast<float>();
    hori_num = (Nx - 1) * Ny;
    vert_num = (Ny - 1) * Nx;
    this->est = est;

    // Different quadrature orders are used for different singularity terms
    order1 = 25;
    order2 = 8;
    order3 = 6;
    p1 = quadrature_points.block(order1 - 1, 0, 1, order1).transpose().cast<float>();
    w1 = quadrature_weights.block(order1 - 1, 0, 1, order1).transpose().cast<float>();
    p2 = quadrature_points.block(order2 - 1, 0, 1, order2).transpose().cast<float>();
    w2 = quadrature_weights.block(order2 - 1, 0, 1, order2).transpose().cast<float>();
    p3 = quadrature_points.block(order3 - 1, 0, 1, order3).transpose().cast<float>();
    w3 = quadrature_weights.block(order3 - 1, 0, 1, order3).transpose().cast<float>();
}

void Singular::computeInvariants() {
    printf("Should not reach this virtual method.\n");
}

/// @brief Compute elements in the off-diagonal blocks of the BEM matrix that involve neighboring basis element pairs
void Singular::neighborEM() {
    KH0 = MatrixXf::Zero(4 * (Nx - 1), 4 * Ny);
    KV0 = MatrixXf::Zero(4 * Nx, 4 * (Ny - 1));

    // Matrix elements involving horizontally neighboring basis elements
    parallel_for(Ny, [&](int start, int end) {
    for (int num_y = start; num_y < end; num_y++) {
        for (int num_x = 1; num_x < Nx; num_x++) {
            float z1 = (float)(zvals(num_x + 1, num_y) - zvals(num_x, num_y));
            float z2 = (float)(zvals(num_x, num_y + 1) - zvals(num_x, num_y));
            float z3 = (float)(zvals(num_x + 1, num_y + 1) - zvals(num_x, num_y));
            float z4 = (float)(zvals(num_x - 1, num_y) - zvals(num_x, num_y));
            float z5 = (float)(zvals(num_x - 1, num_y + 1) - zvals(num_x, num_y));
            KH0.block(4 * (num_x - 1), 4 * num_y, 4, 4) = computeNeighborEM(z1, z2, z3, z4, z5);
        }
    }
    } );

    // Matrix elements involving vertically neighboring basis elements
    parallel_for(Nx, [&](int start, int end) {
    for (int num_x = start; num_x < end; num_x++) {
        for (int num_y = 1; num_y < Ny; num_y++) {
            float z1 = (float)(zvals(num_x, num_y + 1) - zvals(num_x, num_y));
            float z2 = (float)(zvals(num_x + 1, num_y) - zvals(num_x, num_y));
            float z3 = (float)(zvals(num_x + 1, num_y + 1) - zvals(num_x, num_y));
            float z4 = (float)(zvals(num_x, num_y - 1) - zvals(num_x, num_y));
            float z5 = (float)(zvals(num_x + 1, num_y - 1) - zvals(num_x, num_y));
            KV0.block(4 * num_x, 4 * (num_y - 1), 4, 4) = -computeNeighborEM(z1, z2, z3, z4, z5);
        }
    }
    } );
}

/// @brief Compute individual matrix elements through numerical integration after change of coordinates
/// @brief Nontrivial change of coordinates computations are involved; users should not try to understand or modify the code
/// @param z1, z2, z3, z4, z5: height values from the considered basis element pair
/// @return computed matrix elements
MatrixXf Singular::computeNeighborEM(float z1, float z2, float z3, float z4, float z5) {
    MatrixXf block = MatrixXf::Zero(4, 4);
    float z0 = 0;
    float z11_1 = (z4 - z0 - z5 + z2) / 4;
    float z10_1 = (-z4 + z0 - z5 + z2) / 4;
    float z01_1 = (-z4 - z0 + z5 + z2) / 4;
    float z11_2 = (z0 - z1 - z2 + z3) / 4;
    float z10_2 = (-z0 + z1 - z2 + z3) / 4;
    float z01_2 = (-z0 - z1 + z2 + z3) / 4;
    for (int v_out = 0; v_out < order3; v_out++) {
        float v0 = p3(v_out);
        float weight0 = w3(v_out);
        for (int ind_t = 0; ind_t < order3; ind_t++) {
            for (int ind_t1 = 0; ind_t1 < order3; ind_t1++) {
                for (int ind_t2 = 0; ind_t2 < order3; ind_t2++) {
                    float rt = p3(ind_t) * 0.5 + 0.5;
                    float ct = rt + 1;
                    float rt1 = p3(ind_t1) * 0.5 + 0.5;
                    float ct1 = p3(ind_t1) * (2 / ct - 1) / 2 + 0.5;
                    float t2 = p3(ind_t2) * 0.5 + 0.5;
                    float rweight = w3(ind_t) * w3(ind_t1) * w3(ind_t2) / 8;
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
                    float rK1 = weight0 * rweight * (1 - rt1) * (1 - v0) / (M_PI * rdist1 * rdist1 * rdist1);
                    float rK11 = rdist_y1 * 0.5 * d * (z11_1 * v0 + z10_1 - z11_2 * rv_in1 - z10_2);
                    float rK12 = -rdist_x1 * 0.5 * d * (z11_1 * v0 + z10_1) - rdist_y1 * 0.5 * d * (z11_2 * ru_in1 + z01_2) + rdist_z1 * 0.25 * d * d;
                    float rK13 = rdist_x1 * 0.5 * d * (z11_2 * rv_in1 + z10_2) + rdist_y1 * 0.5 * d * (z11_1 * ru_out + z01_1) - rdist_z1 * 0.25 * d * d;
                    float rK14 = rdist_x1 * 0.5 * d * (z11_2 * ru_in1 + z01_2 - z11_1 * ru_out - z01_1);
                    float cK1 = weight0 * cweight * (1 - ct1) * (1 - v0) / (M_PI * cdist1 * cdist1 * cdist1);
                    float cK11 = cdist_y1 * 0.5 * d * (z11_1 * v0 + z10_1 - z11_2 * cv_in1 - z10_2);
                    float cK12 = -cdist_x1 * 0.5 * d * (z11_1 * v0 + z10_1) - cdist_y1 * 0.5 * d * (z11_2 * cu_in1 + z01_2) + cdist_z1 * 0.25 * d * d;
                    float cK13 = cdist_x1 * 0.5 * d * (z11_2 * cv_in1 + z10_2) + cdist_y1 * 0.5 * d * (z11_1 * cu_out + z01_1) - cdist_z1 * 0.25 * d * d;
                    float cK14 = cdist_x1 * 0.5 * d * (z11_2 * cu_in1 + z01_2 - z11_1 * cu_out - z01_1);
                    block(0, 0) += (1 - ru_out) * (1 - ru_in1) * rK11 * rK1 + (1 - cu_out) * (1 - cu_in1) * cK11 * cK1;
                    block(0, 1) += (1 - ru_out) * (1 + ru_in1) * rK11 * rK1 + (1 - cu_out) * (1 + cu_in1) * cK11 * cK1;
                    block(0, 2) += (1 - ru_out) * (1 - rv_in1) * rK12 * rK1 + (1 - cu_out) * (1 - cv_in1) * cK12 * cK1;
                    block(0, 3) += (1 - ru_out) * (1 + rv_in1) * rK12 * rK1 + (1 - cu_out) * (1 + cv_in1) * cK12 * cK1;
                    block(1, 0) += (1 + ru_out) * (1 - ru_in1) * rK11 * rK1 + (1 + cu_out) * (1 - cu_in1) * cK11 * cK1;
                    block(1, 1) += (1 + ru_out) * (1 + ru_in1) * rK11 * rK1 + (1 + cu_out) * (1 + cu_in1) * cK11 * cK1;
                    block(1, 2) += (1 + ru_out) * (1 - rv_in1) * rK12 * rK1 + (1 + cu_out) * (1 - cv_in1) * cK12 * cK1;
                    block(1, 3) += (1 + ru_out) * (1 + rv_in1) * rK12 * rK1 + (1 + cu_out) * (1 + cv_in1) * cK12 * cK1;
                    block(2, 0) += (1 - v0) * (1 - ru_in1) * rK13 * rK1 + (1 - v0) * (1 - cu_in1) * cK13 * cK1;
                    block(2, 1) += (1 - v0) * (1 + ru_in1) * rK13 * rK1 + (1 - v0) * (1 + cu_in1) * cK13 * cK1;
                    block(2, 2) += (1 - v0) * (1 - rv_in1) * rK14 * rK1 + (1 - v0) * (1 - cv_in1) * cK14 * cK1;
                    block(2, 3) += (1 - v0) * (1 + rv_in1) * rK14 * rK1 + (1 - v0) * (1 + cv_in1) * cK14 * cK1;
                    block(3, 0) += (1 + v0) * (1 - ru_in1) * rK13 * rK1 + (1 + v0) * (1 - cu_in1) * cK13 * cK1;
                    block(3, 1) += (1 + v0) * (1 + ru_in1) * rK13 * rK1 + (1 + v0) * (1 + cu_in1) * cK13 * cK1;
                    block(3, 2) += (1 + v0) * (1 - rv_in1) * rK14 * rK1 + (1 + v0) * (1 - cv_in1) * cK14 * cK1;
                    block(3, 3) += (1 + v0) * (1 + rv_in1) * rK14 * rK1 + (1 + v0) * (1 + cv_in1) * cK14 * cK1;
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
                    float rK2 = 2 * weight0 * rweight * (1 - rt1) / (M_PI * rdist2 * rdist2 * rdist2);
                    float rK21 = rdist_y2 * 0.5 * d * (z11_1 * v0 + z10_1 - z11_2 * rv_in2 - z10_2);
                    float rK22 = -rdist_x2 * 0.5 * d * (z11_1 * v0 + z10_1) - rdist_y2 * 0.5 * d * (z11_2 * ru_in2 + z01_2) + rdist_z2 * 0.25 * d * d;
                    float rK23 = rdist_x2 * 0.5 * d * (z11_2 * rv_in2 + z10_2) + rdist_y2 * 0.5 * d * (z11_1 * ru_out + z01_1) - rdist_z2 * 0.25 * d * d;
                    float rK24 = rdist_x2 * 0.5 * d * (z11_2 * ru_in2 + z01_2 - z11_1 * ru_out - z01_1);
                    float cK2 = 2 * weight0 * cweight * (1 - ct1) / (M_PI * cdist2 * cdist2 * cdist2);
                    float cK21 = cdist_y2 * 0.5 * d * (z11_1 * v0 + z10_1 - z11_2 * cv_in2 - z10_2);
                    float cK22 = -cdist_x2 * 0.5 * d * (z11_1 * v0 + z10_1) - cdist_y2 * 0.5 * d * (z11_2 * cu_in2 + z01_2) + cdist_z2 * 0.25 * d * d;
                    float cK23 = cdist_x2 * 0.5 * d * (z11_2 * cv_in2 + z10_2) + cdist_y2 * 0.5 * d * (z11_1 * cu_out + z01_1) - cdist_z2 * 0.25 * d * d;
                    float cK24 = cdist_x2 * 0.5 * d * (z11_2 * cu_in2 + z01_2 - z11_1 * cu_out - z01_1);
                    block(0, 0) += (1 - ru_out) * (1 - ru_in2) * rK21 * rK2 + (1 - cu_out) * (1 - cu_in2) * cK21 * cK2;
                    block(0, 1) += (1 - ru_out) * (1 + ru_in2) * rK21 * rK2 + (1 - cu_out) * (1 + cu_in2) * cK21 * cK2;
                    block(0, 2) += (1 - ru_out) * (1 - rv_in2) * rK22 * rK2 + (1 - cu_out) * (1 - cv_in2) * cK22 * cK2;
                    block(0, 3) += (1 - ru_out) * (1 + rv_in2) * rK22 * rK2 + (1 - cu_out) * (1 + cv_in2) * cK22 * cK2;
                    block(1, 0) += (1 + ru_out) * (1 - ru_in2) * rK21 * rK2 + (1 + cu_out) * (1 - cu_in2) * cK21 * cK2;
                    block(1, 1) += (1 + ru_out) * (1 + ru_in2) * rK21 * rK2 + (1 + cu_out) * (1 + cu_in2) * cK21 * cK2;
                    block(1, 2) += (1 + ru_out) * (1 - rv_in2) * rK22 * rK2 + (1 + cu_out) * (1 - cv_in2) * cK22 * cK2;
                    block(1, 3) += (1 + ru_out) * (1 + rv_in2) * rK22 * rK2 + (1 + cu_out) * (1 + cv_in2) * cK22 * cK2;
                    block(2, 0) += (1 - v0) * (1 - ru_in2) * rK23 * rK2 + (1 - v0) * (1 - cu_in2) * cK23 * cK2;
                    block(2, 1) += (1 - v0) * (1 + ru_in2) * rK23 * rK2 + (1 - v0) * (1 + cu_in2) * cK23 * cK2;
                    block(2, 2) += (1 - v0) * (1 - rv_in2) * rK24 * rK2 + (1 - v0) * (1 - cv_in2) * cK24 * cK2;
                    block(2, 3) += (1 - v0) * (1 + rv_in2) * rK24 * rK2 + (1 - v0) * (1 + cv_in2) * cK24 * cK2;
                    block(3, 0) += (1 + v0) * (1 - ru_in2) * rK23 * rK2 + (1 + v0) * (1 - cu_in2) * cK23 * cK2;
                    block(3, 1) += (1 + v0) * (1 + ru_in2) * rK23 * rK2 + (1 + v0) * (1 + cu_in2) * cK23 * cK2;
                    block(3, 2) += (1 + v0) * (1 - rv_in2) * rK24 * rK2 + (1 + v0) * (1 - cv_in2) * cK24 * cK2;
                    block(3, 3) += (1 + v0) * (1 + rv_in2) * rK24 * rK2 + (1 + v0) * (1 + cv_in2) * cK24 * cK2;
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
                    float rK3 = weight0 * rweight * (1 - rt1) * (1 + v0) / (M_PI * rdist3 * rdist3 * rdist3);
                    float rK31 = rdist_y3 * 0.5 * d * (z11_1 * v0 + z10_1 - z11_2 * rv_in3 - z10_2);
                    float rK32 = -rdist_x3 * 0.5 * d * (z11_1 * v0 + z10_1) - rdist_y3 * 0.5 * d * (z11_2 * ru_in3 + z01_2) + rdist_z3 * 0.25 * d * d;
                    float rK33 = rdist_x3 * 0.5 * d * (z11_2 * rv_in3 + z10_2) + rdist_y3 * 0.5 * d * (z11_1 * ru_out + z01_1) - rdist_z3 * 0.25 * d * d;
                    float rK34 = rdist_x3 * 0.5 * d * (z11_2 * ru_in3 + z01_2 - z11_1 * ru_out - z01_1);
                    float cK3 = weight0 * cweight * (1 - ct1) * (1 + v0) / (M_PI * cdist3 * cdist3 * cdist3);
                    float cK31 = cdist_y3 * 0.5 * d * (z11_1 * v0 + z10_1 - z11_2 * cv_in3 - z10_2);
                    float cK32 = -cdist_x3 * 0.5 * d * (z11_1 * v0 + z10_1) - cdist_y3 * 0.5 * d * (z11_2 * cu_in3 + z01_2) + cdist_z3 * 0.25 * d * d;
                    float cK33 = cdist_x3 * 0.5 * d * (z11_2 * cv_in3 + z10_2) + cdist_y3 * 0.5 * d * (z11_1 * cu_out + z01_1) - cdist_z3 * 0.25 * d * d;
                    float cK34 = cdist_x3 * 0.5 * d * (z11_2 * cu_in3 + z01_2 - z11_1 * cu_out - z01_1);
                    block(0, 0) += (1 - ru_out) * (1 - ru_in3) * rK31 * rK3 + (1 - cu_out) * (1 - cu_in3) * cK31 * cK3;
                    block(0, 1) += (1 - ru_out) * (1 + ru_in3) * rK31 * rK3 + (1 - cu_out) * (1 + cu_in3) * cK31 * cK3;
                    block(0, 2) += (1 - ru_out) * (1 - rv_in3) * rK32 * rK3 + (1 - cu_out) * (1 - cv_in3) * cK32 * cK3;
                    block(0, 3) += (1 - ru_out) * (1 + rv_in3) * rK32 * rK3 + (1 - cu_out) * (1 + cv_in3) * cK32 * cK3;
                    block(1, 0) += (1 + ru_out) * (1 - ru_in3) * rK31 * rK3 + (1 + cu_out) * (1 - cu_in3) * cK31 * cK3;
                    block(1, 1) += (1 + ru_out) * (1 + ru_in3) * rK31 * rK3 + (1 + cu_out) * (1 + cu_in3) * cK31 * cK3;
                    block(1, 2) += (1 + ru_out) * (1 - rv_in3) * rK32 * rK3 + (1 + cu_out) * (1 - cv_in3) * cK32 * cK3;
                    block(1, 3) += (1 + ru_out) * (1 + rv_in3) * rK32 * rK3 + (1 + cu_out) * (1 + cv_in3) * cK32 * cK3;
                    block(2, 0) += (1 - v0) * (1 - ru_in3) * rK33 * rK3 + (1 - v0) * (1 - cu_in3) * cK33 * cK3;
                    block(2, 1) += (1 - v0) * (1 + ru_in3) * rK33 * rK3 + (1 - v0) * (1 + cu_in3) * cK33 * cK3;
                    block(2, 2) += (1 - v0) * (1 - rv_in3) * rK34 * rK3 + (1 - v0) * (1 - cv_in3) * cK34 * cK3;
                    block(2, 3) += (1 - v0) * (1 + rv_in3) * rK34 * rK3 + (1 - v0) * (1 + cv_in3) * cK34 * cK3;
                    block(3, 0) += (1 + v0) * (1 - ru_in3) * rK33 * rK3 + (1 + v0) * (1 - cu_in3) * cK33 * cK3;
                    block(3, 1) += (1 + v0) * (1 + ru_in3) * rK33 * rK3 + (1 + v0) * (1 + cu_in3) * cK33 * cK3;
                    block(3, 2) += (1 + v0) * (1 - rv_in3) * rK34 * rK3 + (1 + v0) * (1 - cv_in3) * cK34 * cK3;
                    block(3, 3) += (1 + v0) * (1 + rv_in3) * rK34 * rK3 + (1 + v0) * (1 + cv_in3) * cK34 * cK3;
                }
            }
        }
    }
    block = 2.0f * eta0FL * block;
    return block;
}

/// @brief Singular matrix elements are computed in four groups; this method specifies which group of elements to compute and performs computations
/// @param dev: an integer with the value of 0, 1, 2, or 3 that specifies the singular matrix element group to compute
void Singular::computeQuarter(int dev) {
    if (dev == 0)
        computeHH();
    else if (dev == 3)
        computeVV();
    else
        computeHV(dev);
}

/// @brief Compute singular matrix element group 0, which involves pairs of horizontally arranged basis functions
void Singular::computeHH() {
    int nnz = est->A[0].rows();
    quarter = MatrixXf::Zero(nnz, 3);
    parallel_for(nnz, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        int m = est->A[0](i, 0);
        int my = m / (Nx - 1);
        int mx = m - (Nx - 1) * my;
        int n = est->A[0](i, 1);
        int ny = n / (Nx - 1);
        int nx = n - (Nx - 1) * ny;
        float eemm1 = 0, eemm2 = 0, emme0 = 0;
        if (abs(mx - nx) + abs(my - ny) <= 1) {
            if (mx == nx && my == ny) {
                eemm1 += LS1(4 * mx + 1, 4 * my + 1) + LS1(4 * (mx + 1) + 0, 4 * my + 0);
                eemm2 += LS2(4 * mx + 1, 4 * my + 1) + LS2(4 * (mx + 1) + 0, 4 * my + 0);
            } else if (mx == nx) {
                eemm1 += LV1(4 * mx + 3, 4 * my + 3) + LV1(4 * (mx + 1) + 2, 4 * my + 2);
                eemm2 += LV2(4 * mx + 3, 4 * my + 3) + LV2(4 * (mx + 1) + 2, 4 * my + 2);
                emme0 += KV0(4 * mx + 3, 4 * my + 3) + KV0(4 * (mx + 1) + 2, 4 * my + 2);
            } else {
                eemm1 += LH1(4 * mx + 1, 4 * my + 1) + LH1(4 * nx + 0, 4 * my + 0);
                eemm2 += LH2(4 * mx + 1, 4 * my + 1) + LH2(4 * nx + 0, 4 * my + 0);
                emme0 += KH0(4 * mx + 1, 4 * my + 1) + KH0(4 * nx + 0, 4 * my + 0);
            }
        }
        if (abs(mx - nx - 1) + abs(my - ny) <= 1) {
            if (mx == nx + 1) {
                eemm1 += LV1(4 * mx + 3, 4 * my + 2);
                eemm2 += LV2(4 * mx + 3, 4 * my + 2);
                emme0 += KV0(4 * mx + 3, 4 * my + 2);
            } else {
                eemm1 += LH1(4 * mx + 1, 4 * my + 0);
                eemm2 += LH2(4 * mx + 1, 4 * my + 0);
                emme0 += KH0(4 * mx + 1, 4 * my + 0);
            }
        }
        if (abs(mx - nx + 1) + abs(my - ny) <= 1) {
            if (mx + 1 == nx && my == ny) {
                eemm1 += LS1(4 * nx + 0, 4 * ny + 1);
                eemm2 += LS2(4 * nx + 0, 4 * ny + 1);
            } else if (mx + 1 == nx) {
                eemm1 += LV1(4 * nx + 2, 4 * my + 3);
                eemm2 += LV2(4 * nx + 2, 4 * my + 3);
                emme0 += KV0(4 * nx + 2, 4 * my + 3);
            } else if (mx == nx) {
                eemm1 += LH1(4 * nx + 1, 4 * ny + 0);
                eemm2 += LH2(4 * nx + 1, 4 * ny + 0);
                emme0 += KH0(4 * nx + 1, 4 * ny + 0);
            } else {
                eemm1 += LH1(4 * (mx + 1) + 0, 4 * my + 1);
                eemm2 += LH2(4 * (mx + 1) + 0, 4 * my + 1);
                emme0 += KH0(4 * (mx + 1) + 0, 4 * my + 1);
            }
        }
        if (m == n) {
            eemm1 = 0.5f * eemm1;
            eemm2 = 0.5f * eemm2;
            emme0 = 0.5f * emme0;
        }
        quarter(i, 0) = eemm1;
        quarter(i, 1) = eemm2;
        quarter(i, 2) = emme0;
    }
    } );
}

/// @brief Compute singular matrix element group 1 or 2, which involves pairs of one horizontally arranged basis function and one vertically arranged basis function
void Singular::computeHV(int dev) {
    int nnz = est->A[dev].rows();
    quarter = MatrixXf::Zero(nnz, 3);
    parallel_for(nnz, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        int m = est->A[dev](i, 0);
        int my = m / (Nx - 1);
        int mx = m - (Nx - 1) * my;
        int n = est->A[dev](i, 1);
        int nx = n / (Ny - 1);
        int ny = n - (Ny - 1) * nx;
        float eemm1 = 0, eemm2 = 0, emme0 = 0;
        if (abs(mx - nx) + abs(my - ny) <= 1) {
            if (mx == nx && my == ny) {
                eemm1 += LS1(4 * mx + 1, 4 * my + 3);
                eemm2 += LS2(4 * mx + 1, 4 * my + 3);
            } else if (mx + 1 == nx) {
                eemm1 += LH1(4 * mx + 1, 4 * my + 3);
                eemm2 += LH2(4 * mx + 1, 4 * my + 3);
                emme0 += KH0(4 * mx + 1, 4 * my + 3);
            } else if (nx + 1 == mx) {
                eemm1 += LH1(4 * nx + 3, 4 * ny + 1);
                eemm2 += LH2(4 * nx + 3, 4 * ny + 1);
                emme0 += KH0(4 * nx + 3, 4 * ny + 1);
            } else if (my + 1 == ny) {
                eemm1 += LV1(4 * mx + 3, 4 * my + 1);
                eemm2 += LV2(4 * mx + 3, 4 * my + 1);
                emme0 += KV0(4 * mx + 3, 4 * my + 1);
            } else {
                eemm1 += LV1(4 * nx + 1, 4 * ny + 3);
                eemm2 += LV2(4 * nx + 1, 4 * ny + 3);
                emme0 += KV0(4 * nx + 1, 4 * ny + 3);
            }
        }
        if (abs(mx - nx) + abs(my - ny - 1) <= 1) {
            if (mx == nx && my == ny + 1) {
                eemm1 += LS1(4 * mx + 1, 4 * my + 2);
                eemm2 += LS2(4 * mx + 1, 4 * my + 2);
            } else if (mx + 1 == nx) {
                eemm1 += LH1(4 * mx + 1, 4 * my + 2);
                eemm2 += LH2(4 * mx + 1, 4 * my + 2);
                emme0 += KH0(4 * mx + 1, 4 * my + 2);
            } else if (nx + 1 == mx) {
                eemm1 += LH1(4 * nx + 2, 4 * my + 1);
                eemm2 += LH2(4 * nx + 2, 4 * my + 1);
                emme0 += KH0(4 * nx + 2, 4 * my + 1);
            } else if (my == ny) {
                eemm1 += LV1(4 * mx + 3, 4 * my + 0);
                eemm2 += LV2(4 * mx + 3, 4 * my + 0);
                emme0 += KV0(4 * mx + 3, 4 * my + 0);
            } else {
                eemm1 += LV1(4 * nx + 0, 4 * (ny + 1) + 3);
                eemm2 += LV2(4 * nx + 0, 4 * (ny + 1) + 3);
                emme0 += KV0(4 * nx + 0, 4 * (ny + 1) + 3);
            }
        }
        if (abs(mx - nx + 1) + abs(my - ny) <= 1) {
            if (mx + 1 == nx && my == ny) {
                eemm1 += LS1(4 * nx + 0, 4 * ny + 3);
                eemm2 += LS2(4 * nx + 0, 4 * ny + 3);
            } else if (mx + 2 == nx) {
                eemm1 += LH1(4 * (mx + 1) + 0, 4 * my + 3);
                eemm2 += LH2(4 * (mx + 1) + 0, 4 * my + 3);
                emme0 += KH0(4 * (mx + 1) + 0, 4 * my + 3);
            } else if (mx == nx) {
                eemm1 += LH1(4 * nx + 3, 4 * ny + 0);
                eemm2 += LH2(4 * nx + 3, 4 * ny + 0);
                emme0 += KH0(4 * nx + 3, 4 * ny + 0);
            } else if (my + 1 == ny) {
                eemm1 += LV1(4 * nx + 2, 4 * my + 1);
                eemm2 += LV2(4 * nx + 2, 4 * my + 1);
                emme0 += KV0(4 * nx + 2, 4 * my + 1);
            } else {
                eemm1 += LV1(4 * nx + 1, 4 * ny + 2);
                eemm2 += LV2(4 * nx + 1, 4 * ny + 2);
                emme0 += KV0(4 * nx + 1, 4 * ny + 2);
            }
        }
        if (abs(mx - nx + 1) + abs(my - ny - 1) <= 1) {
            if (mx + 1 == nx && my == ny + 1) {
                eemm1 += LS1(4 * nx + 0, 4 * my + 2);
                eemm2 += LS2(4 * nx + 0, 4 * my + 2);
            } else if (mx + 2 == nx) {
                eemm1 += LH1(4 * (mx + 1) + 0, 4 * my + 2);
                eemm2 += LH2(4 * (mx + 1) + 0, 4 * my + 2);
                emme0 += KH0(4 * (mx + 1) + 0, 4 * my + 2);
            } else if (mx == nx) {
                eemm1 += LH1(4 * nx + 2, 4 * my + 0);
                eemm2 += LH2(4 * nx + 2, 4 * my + 0);
                emme0 += KH0(4 * nx + 2, 4 * my + 0);
            } else if (my == ny) {
                eemm1 += LV1(4 * nx + 2, 4 * my + 0);
                eemm2 += LV2(4 * nx + 2, 4 * my + 0);
                emme0 += KV0(4 * nx + 2, 4 * my + 0);
            } else {
                eemm1 += LV1(4 * nx + 0, 4 * (ny + 1) + 2);
                eemm2 += LV2(4 * nx + 0, 4 * (ny + 1) + 2);
                emme0 += KV0(4 * nx + 0, 4 * (ny + 1) + 2);
            }
        }
        quarter(i, 0) = eemm1;
        quarter(i, 1) = eemm2;
        quarter(i, 2) = emme0;
    }
    } );
}

/// @brief Compute singular matrix element group 3, which involves pairs of vertically arranged basis functions
void Singular::computeVV() {
    int nnz = est->A[3].rows();
    quarter = MatrixXf::Zero(nnz, 3);
    parallel_for(nnz, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        int m = est->A[3](i, 0);
        int mx = m / (Ny - 1);
        int my = m - (Ny - 1) * mx;
        int n = est->A[3](i, 1);
        int nx = n / (Ny - 1);
        int ny = n - (Ny - 1) * nx;
        float eemm1 = 0, eemm2 = 0, emme0 = 0;
        if (abs(mx - nx) + abs(my - ny) <= 1) {
            if (mx == nx && my == ny) {
                eemm1 += LS1(4 * mx + 3, 4 * my + 3) + LS1(4 * mx + 2, 4 * (my + 1) + 2);
                eemm2 += LS2(4 * mx + 3, 4 * my + 3) + LS2(4 * mx + 2, 4 * (my + 1) + 2);
            } else if (mx == nx) {
                eemm1 += LV1(4 * mx + 1, 4 * my + 1) + LV1(4 * mx + 0, 4 * ny + 0);
                eemm2 += LV2(4 * mx + 1, 4 * my + 1) + LV2(4 * mx + 0, 4 * ny + 0);
                emme0 += KV0(4 * mx + 1, 4 * my + 1) + KV0(4 * mx + 0, 4 * ny + 0);
            } else {
                eemm1 += LH1(4 * mx + 3, 4 * my + 3) + LH1(4 * mx + 2, 4 * (my + 1) + 2);
                eemm2 += LH2(4 * mx + 3, 4 * my + 3) + LH2(4 * mx + 2, 4 * (my + 1) + 2);
                emme0 += KH0(4 * mx + 3, 4 * my + 3) + KH0(4 * mx + 2, 4 * (my + 1) + 2);
            }
        }
        if (abs(mx - nx) + abs(my - ny - 1) <= 1) {
            if (mx == nx) {
                eemm1 += LV1(4 * mx + 1, 4 * my + 0);
                eemm2 += LV2(4 * mx + 1, 4 * my + 0);
                emme0 += KV0(4 * mx + 1, 4 * my + 0);
            } else {
                eemm1 += LH1(4 * mx + 3, 4 * my + 2);
                eemm2 += LH2(4 * mx + 3, 4 * my + 2);
                emme0 += KH0(4 * mx + 3, 4 * my + 2);
            }
        }
        if (abs(mx - nx) + abs(my - ny + 1) <= 1) {
            if (mx == nx && my + 1 == ny) {
                eemm1 += LS1(4 * nx + 2, 4 * ny + 3);
                eemm2 += LS2(4 * nx + 2, 4 * ny + 3);
            } else if (my == ny) {
                eemm1 += LV1(4 * mx + 1, 4 * my + 0);
                eemm2 += LV2(4 * mx + 1, 4 * my + 0);
                emme0 += KV0(4 * mx + 1, 4 * my + 0);
            } else if (my + 1 == ny) {
                eemm1 += LH1(4 * mx + 2, 4 * ny + 3);
                eemm2 += LH2(4 * mx + 2, 4 * ny + 3);
                emme0 += KH0(4 * mx + 2, 4 * ny + 3);
            } else {
                eemm1 += LV1(4 * mx + 0, 4 * (my + 1) + 1);
                eemm2 += LV2(4 * mx + 0, 4 * (my + 1) + 1);
                emme0 += KV0(4 * mx + 0, 4 * (my + 1) + 1);
            }
        }
        if (m == n) {
            eemm1 = 0.5f * eemm1;
            eemm2 = 0.5f * eemm2;
            emme0 = 0.5f * emme0;
        }
        quarter(i, 0) = eemm1;
        quarter(i, 1) = eemm2;
        quarter(i, 2) = emme0;
    }
    } );
}