/* Super class module for computing matrix-vector products in BEM.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "MVProd.h"

/// @brief A super class module for computing matrix-vector products for solving the BEM linear system
/// @param est: an Estimate object with information on the simulated surface
/// @param singular: a Singular object that computes the matrix elements with singularties in the underlying integrals
/// @param grid: a Grid object with information on the 3D grid of point sources
MVProd::MVProd(Estimate* est, Singular* singular, Grid* grid) {
    this->Nx = est->Nx;
    this->Ny = est->Ny;
    this->d = est->d;
    this->est = est;
    this->singular = singular;
    this->grid = grid;
    hori_num = (Nx - 1) * Ny;
    vert_num = (Ny - 1) * Nx;
    totalX = grid->totalX;
    totalY = grid->totalY;
    totalZ = grid->totalZ;
    N = totalX * totalY * totalZ;
    num_pts = grid->num_pts;
    dx = grid->dx;
    dy = grid->dy;
    dz = grid->dz;
    hori_row = grid->hori_f.rows();
    hori_col = grid->hori_f.cols();
    vert_row = grid->vert_f.rows();
    vert_col = grid->vert_f.cols();
}

void MVProd::setParameters(double eta1, dcomp eta2, double lambda) {
    printf("Should not reach this virtual method.\n");
}

VectorXcf MVProd::multiply(VectorXcf x) {
    printf("Should not reach this virtual method.\n");
    VectorXcf y = VectorXcf::Zero(2 * hori_num + 2 * vert_num);
    return y;
}

void MVProd::cleanAll() {
    printf("Should not reach this virtual method.\n");
}