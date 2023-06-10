#pragma once
#include "MVProd.h"
#include "Singular12.h"

class MVProd2: public MVProd {
    public:
        bool isDielectric;
        int nnzA[4], nnzB[4], *d_Arows[4], *d_Acols[4], *d_Brows[4], *d_Bcols[4], *d_hori_f, *d_hori_b, *d_vert_f, *d_vert_b;
        float *zvals, *xvech, *wvech, *xvecl, *wvecl;
        Tfcomp e1, e2, const1, const21, const22;
        cuComplex alpha, beta, *d_base1[4], *d_base2[4], *d_Aee[4], *d_Aem[4], *d_Amm[4], *d_Bee[4], *d_Bem[4], *d_Bmm[4], *d_x1, *d_x2, *d_x3, *d_x4, *d_y1, *d_y2, *d_y3, *d_y4;
        Tfcomp *d_hori_x, *d_hori_z, *d_hori_d, *d_vert_y, *d_vert_z, *d_vert_d, *d_hori_X, *d_hori_Z, *d_hori_D, *d_vert_Y, *d_vert_Z, *d_vert_D, *g0_data, *g1_data, *g2_data, *g3_data, *g4_data, *g5_data, *g6_data, *g7_data, *geo0_data, *geo1_data;
        VectorXcf h_y1, h_y2, h_y3, h_y4;
        cusparseHandle_t handle;
        cusparseSpMatDescr_t Aee[4], Aem[4], Amm[4], Bee[4], Bem[4], Bmm[4];
        cusparseDnVecDescr_t x1, x2, x3, x4, y1, y2, y3, y4;
        void* d_work;
        cufftHandle plan;
    
        MVProd2(Estimate* est, Singular* singular, Grid* grid, bool isDielectric, int ind);
        void initializeNear();
        void initializeFar();
        void setParameters(double eta1, dcomp eta2, double lambda);
        VectorXcf multiply(VectorXcf x);
        void near();
        void far0();
        void far1();
        void far2();
        void far3();
        void cleanAll();
};