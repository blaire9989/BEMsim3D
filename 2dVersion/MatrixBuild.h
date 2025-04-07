#pragma once
#include "constants.h"
#include "parallel.h"
#include "HankelTable.h"

class MatrixBuild {
    public:
        int num_components, N, order_q, order_r, order_s, order_t;
        double lambda, omega;
        VectorXd pvals_q, wvals_q, pvals_r, wvals_r, pvals_s, wvals_s, pvals_t, wvals_t;
        MatrixXd xyvals;
        MatrixXi geo_info, mat_info;
        MatrixXcf Z_TE, Z_TM;
        HankelTable* hankel;
    
        MatrixBuild(MatrixXd xyvals, MatrixXi info, HankelTable* hankel);
        void assemble(double lambda, double n1);
        void computeDiagonalBlock(int index, dcomp ior_ext, dcomp ior_int);
        void computeOffDiagonalBlock(int index1, int index2, dcomp ior);
        void selectOrders(Vector2d pt1, Vector2d pt2, int& order1, int& order2, VectorXd& pvals1, VectorXd& pvals2, VectorXd& wvals1, VectorXd& wvals2);
        void Ltt(int m, int n, MatrixXd XY1, MatrixXd XY2, dcomp ior1, dcomp ior2, fcomp& tt1, fcomp& tt2);
        void Lzz(int m, int n, MatrixXd XY1, MatrixXd XY2, dcomp ior1, dcomp ior2, fcomp& zz1, fcomp& zz2);
        void Ktz(int m, int n, MatrixXd XY1, MatrixXd XY2, dcomp ior1, dcomp ior2, fcomp& tz1, fcomp& tz2);
};