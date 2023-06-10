#pragma once
#include "MVProd.h"
#include "Singular34.h"

class MVProd4: public MVProd {
    public:
        int devNumber[4], nnzA[4], nnzB[4], *d_Arows[4], *d_Acols[4], *d_Brows[4], *d_Bcols[4], *d_hori_f[4], *d_hori_b[4], *d_vert_f[4], *d_vert_b[4];
        float *zvals[4], *xvech[4], *wvech[4], *xvecl[4], *wvecl[4];
        cuComplex alpha, beta, *d_base1[4], *d_base2[4], *d_Aee[4], *d_Aem[4], *d_Amm[4], *d_Bee[4], *d_Bem[4], *d_Bmm[4], *d_x1[4], *d_x2[4], *d_x3[4], *d_x4[4], *d_y1[4], *d_y2[4], *d_y3[4], *d_y4[4];
        Tfcomp *d_hori_x[4], *d_hori_z[4], *d_hori_d[4], *d_vert_y[4], *d_vert_z[4], *d_vert_d[4], *g0_data[2], *g1_data[2], *g2_data[2], *g3_data[2], *g4_data[2], *geo0_data[2], *geo1_data[2], *geo2_data[2];
        VectorXcf h_y1, h_y2, h_y3, h_y4;
        cusparseHandle_t handle0, handle1, handle2, handle3;
        cusparseSpMatDescr_t Aee[4], Aem[4], Amm[4], Bee[4], Bem[4], Bmm[4];
        cusparseDnVecDescr_t x1[4], x2[4], x3[4], x4[4], y1[4], y2[4], y3[4], y4[4];
        void* d_work[4];
        cufftHandle plan[4];
    
        MVProd4(Estimate* est, Singular* singular, Grid* grid, int ind0, int ind1, int ind2, int ind3);
        void initializeNear();
        void initializeFar();
        void setParameters(double eta1, dcomp eta2, double lambda);
        VectorXcf multiply(VectorXcf x);
        void gpu0();
        void gpu1();
        void gpu2();
        void gpu3();
        void cleanAll();
};