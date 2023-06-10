#pragma once
#include "constants.h"
#include "Estimate.h"
#include "MVProd0.h"
#include "MVProd1.h"
#include "MVProd2.h"
#include "MVProd3.h"
#include "MVProd4.h"

class Solver {
    public:
        int hori_num, vert_num;
        MVProd* mv;
        VectorXcf x;
        
        Solver(Estimate* est, MVProd* mv);
        int csMINRES(VectorXcf b, int maxit, float rtol, bool show = false);
        VectorXcf symOrtho(fcomp a, fcomp b);
};