#pragma once
#include <Eigen/Sparse>
#include <fftw3.h>
#include "constants.h"
#include "MVProd.h"
#include "Singular0.h"

typedef Eigen::SparseMatrix<fcomp> SpMatfc;
typedef Eigen::Triplet<fcomp> T;

class MVProd0: public MVProd {
    public:
        int orderH, orderL, numX, numY, numZ;
        fcomp k1, k2, const1, const21, const22, c1, c2, c3, c4, eta1, eta2;
        bool isDielectric;
        VectorXf pH, pL, wH, wL, xvals, yvals;
        MatrixXf zvals;
        MatrixXcf qData, data1, data3, data41, data42;
        vector<T> ee_fill, mm_fill, em_fill;
        SpMatfc Zeehh, Zmmhh, Zemhh, Zeehv, Zmmhv, Zemhv, Zeevv, Zmmvv, Zemvv;
        fftwf_complex *array1, *array3, *array41, *array42;
        fftwf_plan forward2, forward3, forward41, forward42, backward1, backward2, backward3;
    
        MVProd0(Estimate* est, Singular* singular, Grid* grid);
        void setParameters(double eta1, dcomp eta2, double lambda);
        void computeGreens();
        void sparseHH();
        void sparseHV();
        void sparseVV();
        VectorXcf multiply(VectorXcf x);
        VectorXcf near(VectorXcf x);
        VectorXcf far(VectorXcf x, fcomp eta, fcomp const2, MatrixXcf& data4, MatrixXcf& hori_x, MatrixXcf& hori_z, MatrixXcf& hori_d, MatrixXcf& vert_y, MatrixXcf& vert_z, MatrixXcf& vert_d);
        void computeHH();
        void computeHV(int dev);
        void computeVV();
        void individual(int mx, int my, int nx, int ny, int type, int side1, int side2, fcomp& store1, fcomp& store2, fcomp& store3);
        void cleanAll();
};