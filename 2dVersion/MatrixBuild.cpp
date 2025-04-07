#include "MatrixBuild.h"

MatrixBuild::MatrixBuild(MatrixXd xyvals, MatrixXi info, HankelTable* hankel) {
    this->xyvals = xyvals;
    this->geo_info = info;
    num_components = geo_info.rows();
    mat_info = MatrixXi::Zero(num_components, 2);
    mat_info(0, 0) = 0;
    mat_info(0, 1) = geo_info(0, 1) - 2;
    for (int i = 1; i < num_components; i++) {
        mat_info(i, 0) = mat_info(i - 1, 0) + mat_info(i - 1, 1);
        mat_info(i, 1) = geo_info(i, 1) - 2;
    }
    N = mat_info(num_components - 1, 0) + mat_info(num_components - 1, 1);
    this->hankel = hankel;
    order_q = 3;
    pvals_q = quadrature_points.block(order_q - 1, 0, 1, order_q).transpose();
    wvals_q = quadrature_weights.block(order_q - 1, 0, 1, order_q).transpose();
    order_r = 6;
    pvals_r = quadrature_points.block(order_r - 1, 0, 1, order_r).transpose();
    wvals_r = quadrature_weights.block(order_r - 1, 0, 1, order_r).transpose();
    order_s = 14;
    pvals_s = quadrature_points.block(order_s - 1, 0, 1, order_s).transpose();
    wvals_s = quadrature_weights.block(order_s - 1, 0, 1, order_s).transpose();
    order_t = 15;
    pvals_t = quadrature_points.block(order_t - 1, 0, 1, order_t).transpose();
    wvals_t = quadrature_weights.block(order_t - 1, 0, 1, order_t).transpose();
}

void MatrixBuild::assemble(double lambda, double n1) {
    this->lambda = lambda;
    omega = c / lambda * 2 * M_PI;
    double n0 = 1.0;
    Z_TE = MatrixXcf::Zero(2 * N, 2 * N);
    Z_TM = MatrixXcf::Zero(2 * N, 2 * N);

    // Assemble the BEM matrix (for Nate's project, assuming there are 2 boundary layers)
    computeDiagonalBlock(0, n0, n1);
    computeDiagonalBlock(1, n1, n0);
    computeOffDiagonalBlock(0, 1, n1);
}

void MatrixBuild::computeDiagonalBlock(int index, dcomp ior_ext, dcomp ior_int) {
    MatrixXd XY = xyvals.block(geo_info(index, 0), 0, geo_info(index, 1), 2);
    dcomp eps_ext = 1 / (mu * c * c) * ior_ext * ior_ext;
    dcomp eps_int = 1 / (mu * c * c) * ior_int * ior_int;
    int init = mat_info(index, 0);
    int size = mat_info(index, 1);

    // Compute symmetrical EJ and HM blocks
    int total = 0;
    vector<int> mvec, nvec;
    for (int m = 0; m < size; m++) {
        for (int n = m; n < size; n++) {
            total = total + 1;
            mvec.push_back(m);
            nvec.push_back(n);
        }
    }
    fcomp c0 = (fcomp)(cunit * omega * mu);
    fcomp c1 = (fcomp)(cunit * omega * eps_ext * eta0 * eta0);
    fcomp c2 = (fcomp)(cunit * omega * eps_int * eta0 * eta0);
    parallel_for(total, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        int m = mvec[i];
        int n = nvec[i];
        fcomp tt_ext, tt_int;
        Ltt(m + 1, n + 1, XY, XY, ior_ext, ior_int, tt_ext, tt_int);
        // EJ block for TM polarization
        Z_TM(init + m, init + n) = c0 * tt_ext + c0 * tt_int;
        Z_TM(init + n, init + m) = c0 * tt_ext + c0 * tt_int;
        // HM block for TE polarization
        Z_TE(N + init + m, N + init + n) = -c1 * tt_ext - c2 * tt_int;
        Z_TE(N + init + n, N + init + m) = -c1 * tt_ext - c2 * tt_int;
        fcomp zz_ext, zz_int;
        Lzz(m + 1, n + 1, XY, XY, ior_ext, ior_int, zz_ext, zz_int);
        // EJ block for TE polarization
        Z_TE(init + m, init + n) = c0 * zz_ext + c0 * zz_int;
        Z_TE(init + n, init + m) = c0 * zz_ext + c0 * zz_int;
        // HM block for TM polarization
        Z_TM(N + init + m, N + init + n) = -c1 * zz_ext - c2 * zz_int;
        Z_TM(N + init + n, N + init + m) = -c1 * zz_ext - c2 * zz_int;
    }
    } );

    // Compute EM and HJ blocks: one is the transpose of the other
    parallel_for(size, [&](int start, int end) {
    for (int m = start; m < end; m++) {
        for (int n = 0; n < size; n++) {
            fcomp tz_ext, tz_int;
            Ktz(m + 1, n + 1, XY, XY, ior_ext, ior_int, tz_ext, tz_int);
            fcomp element = eta0f * (tz_ext + tz_int);
            // EM block for TM polarization
            Z_TM(init + m, N + init + n) = element;
            // HJ block for TE polarization
            Z_TE(N + init + m, init + n) = element;
            // HJ block for TM polarization
            Z_TM(N + init + n, init + m) = element;
            // EM block for TE polarization
            Z_TE(init + n, N + init + m) = element;
        }
    }
    } );
}

void MatrixBuild::computeOffDiagonalBlock(int index1, int index2, dcomp ior) {
    MatrixXd XY1 = xyvals.block(geo_info(index1, 0), 0, geo_info(index1, 1), 2);
    MatrixXd XY2 = xyvals.block(geo_info(index2, 0), 0, geo_info(index2, 1), 2);
    dcomp eps = 1 / (mu * c * c) * ior * ior;
    int init1 = mat_info(index1, 0);
    int size1 = mat_info(index1, 1);
    int init2 = mat_info(index2, 0);
    int size2 = mat_info(index2, 1);

    // Compute EJ and HM blocks
    fcomp cEJ = (fcomp)(cunit * omega * mu);
    fcomp cHM = (fcomp)(cunit * omega * eps * eta0 * eta0);
    parallel_for(size1, [&](int start, int end) {
    for (int m = start; m < end; m++) {
        for (int n = 0; n < size2; n++) {
            fcomp tt, tt_null;
            Ltt(m + 1, n + 1, XY1, XY2, ior, 0, tt, tt_null);
            // EJ blocks for TM polarization
            Z_TM(init1 + m, init2 + n) = cEJ * tt;
            Z_TM(init2 + n, init1 + m) = cEJ * tt;
            // HM blocks for TE polarization
            Z_TE(N + init1 + m, N + init2 + n) = -cHM * tt;
            Z_TE(N + init2 + n, N + init1 + m) = -cHM * tt;
            fcomp zz, zz_null;
            Lzz(m + 1, n + 1, XY1, XY2, ior, 0, zz, zz_null);
            // EJ blocks for TE polarization
            Z_TE(init1 + m, init2 + n) = cEJ * zz;
            Z_TE(init2 + n, init1 + m) = cEJ * zz;
            // HM blocks for TM polarization
            Z_TM(N + init1 + m, N + init2 + n) = -cHM * zz;
            Z_TM(N + init2 + n, N + init1 + m) = -cHM * zz;
        }
    }
    } );

    // Compute EM and HJ blocks
    parallel_for(size1, [&](int start, int end) {
    for (int m = start; m < end; m++) {
        for (int n = 0; n < size2; n++) {
            fcomp tz, tz_null;
            Ktz(m + 1, n + 1, XY1, XY2, ior, 0, tz, tz_null);
            fcomp element = eta0f * tz;
            // EM block for TM polarization
            Z_TM(init1 + m, N + init2 + n) = element;
            // HJ block for TE polarization
            Z_TE(N + init1 + m, init2 + n) = element;
            // HJ block for TM polarization
            Z_TM(N + init2 + n, init1 + m) = element;
            // EM block for TE polarization
            Z_TE(init2 + n, N + init1 + m) = element;
        }
    }
    } );
    parallel_for(size2, [&](int start, int end) {
    for (int n = start; n < end; n++) {
        for (int m = 0; m < size1; m++) {
            fcomp tz, tz_null;
            Ktz(n + 1, m + 1, XY2, XY1, ior, 0, tz, tz_null);
            fcomp element = eta0f * tz;
            // EM block for TM polarization
            Z_TM(init2 + n, N + init1 + m) = element;
            // HJ block for TE polarization
            Z_TE(N + init2 + n, init1 + m) = element;
            // HJ block for TM polarization
            Z_TM(N + init1 + m, init2 + n) = element;
            // EM block for TE polarization
            Z_TE(init1 + m, N + init2 + n) = element;
        }
    }
    } );
}

void MatrixBuild::selectOrders(Vector2d pt1, Vector2d pt2, int& order1, int& order2, VectorXd& pvals1, VectorXd& pvals2, VectorXd& wvals1, VectorXd& wvals2) {
    double dist = abs(pt1(0) - pt2(0)) + abs(pt1(1) - pt2(1));
    if (dist > 0.04) {
        order1 = order_q;
        order2 = order_q;
        pvals1 = pvals_q;
        pvals2 = pvals_q;
        wvals1 = wvals_q;
        wvals2 = wvals_q;
    } else if (dist != 0) {
        order1 = order_r;
        order2 = order_r;
        pvals1 = pvals_r;
        pvals2 = pvals_r;
        wvals1 = wvals_r;
        wvals2 = wvals_r;
    } else {
        order1 = order_s;
        order2 = order_t;
        pvals1 = pvals_s;
        pvals2 = pvals_t;
        wvals1 = wvals_s;
        wvals2 = wvals_t;
    }
}

void MatrixBuild::Ltt(int m, int n, MatrixXd XY1, MatrixXd XY2, dcomp ior1, dcomp ior2, fcomp& tt1, fcomp& tt2) {
    dcomp k1 = 2.0 * M_PI * ior1 / lambda;
    dcomp k2 = 2.0 * M_PI * ior2 / lambda;
    dcomp b1 = 1.781 * k1 / 2.0;
    dcomp b2 = 1.781 * k2 / 2.0;

    // Locate relevant basis elements
    int N1 = XY1.rows(), N2 = XY2.rows();
    Vector2d prev1(XY1((m - 1 + N1) % N1, 0), XY1((m - 1 + N1) % N1, 1));
    Vector2d curr1(XY1(m, 0), XY1(m, 1));
    Vector2d next1(XY1((m + 1) % N1, 0), XY1((m + 1) % N1, 1));
    Vector2d prev2(XY2((n - 1 + N2) % N2, 0), XY2((n - 1 + N2) % N2, 1));
    Vector2d curr2(XY2(n, 0), XY2(n, 1));
    Vector2d next2(XY2((n + 1) % N2, 0), XY2((n + 1) % N2, 1));
    dcomp val1 = 0, val2 = 0, val3 = 0, val4 = 0;
    int order1, order2;
    VectorXd pvals1, pvals2, wvals1, wvals2;

    // Segment (1, 1)
    double dot11 = (curr1 - prev1).dot(curr2 - prev2);
    selectOrders(curr1, curr2, order1, order2, pvals1, pvals2, wvals1, wvals2);
    for (int num1 = 0; num1 < order1; num1++) {
        double r1 = 0.5 * pvals1(num1) + 0.5;
        for (int num2 = 0; num2 < order2; num2++) {
            double r2 = 0.5 * pvals2(num2) + 0.5;
            Vector2d p1 = (1 - r1) * prev1 + r1 * curr1;
            Vector2d p2 = (1 - r2) * prev2 + r2 * curr2;
            double dist = (p1 - p2).norm();
            dcomp g = -cunit / 4.0 * hankel->lookUp(k1, dist, 0);
            val1 += wvals1(num1) * wvals2(num2) / 4.0 * r1 * r2 * dot11 * g;
            val2 += wvals1(num1) * wvals2(num2) / 4.0 * g;
            if (ior2.real() != 0) {
                g = -cunit / 4.0 * hankel->lookUp(k2, dist, 0);
                val3 += wvals1(num1) * wvals2(num2) / 4.0 * r1 * r2 * dot11 * g;
                val4 += wvals1(num1) * wvals2(num2) / 4.0 * g;
            }
        }
    }

    // Segment (1, 2)
    double dot12 = (curr1 - prev1).dot(next2 - curr2);
    selectOrders(curr1, next2, order1, order2, pvals1, pvals2, wvals1, wvals2);
    for (int num1 = 0; num1 < order1; num1++) {
        double r1 = 0.5 * pvals1(num1) + 0.5;
        for (int num2 = 0; num2 < order2; num2++) {
            double r2 = 0.5 * pvals2(num2) + 0.5;
            Vector2d p1 = (1 - r1) * prev1 + r1 * curr1;
            Vector2d p2 = (1 - r2) * curr2 + r2 * next2;
            double dist = (p1 - p2).norm();
            dcomp g = -cunit / 4.0 * hankel->lookUp(k1, dist, 0);
            val1 += wvals1(num1) * wvals2(num2) / 4.0 * r1 * (1 - r2) * dot12 * g;
            val2 -= wvals1(num1) * wvals2(num2) / 4.0 * g;
            if (ior2.real() != 0) {
                g = -cunit / 4.0 * hankel->lookUp(k2, dist, 0);
                val3 += wvals1(num1) * wvals2(num2) / 4.0 * r1 * (1 - r2) * dot12 * g;
                val4 -= wvals1(num1) * wvals2(num2) / 4.0 * g;
            }
        }
    }

    // Segment (2, 1)
    double dot21 = (next1 - curr1).dot(curr2 - prev2);
    selectOrders(next1, curr2, order1, order2, pvals1, pvals2, wvals1, wvals2);
    for (int num1 = 0; num1 < order1; num1++) {
        double r1 = 0.5 * pvals1(num1) + 0.5;
        for (int num2 = 0; num2 < order2; num2++) {
            double r2 = 0.5 * pvals2(num2) + 0.5;
            Vector2d p1 = (1 - r1) * curr1 + r1 * next1;
            Vector2d p2 = (1 - r2) * prev2 + r2 * curr2;
            double dist = (p1 - p2).norm();
            dcomp g = -cunit / 4.0 * hankel->lookUp(k1, dist, 0);
            val1 += wvals1(num1) * wvals2(num2) / 4.0 * (1 - r1) * r2 * dot21 * g;
            val2 -= wvals1(num1) * wvals2(num2) / 4.0 * g;
            if (ior2.real() != 0) {
                g = -cunit / 4.0 * hankel->lookUp(k2, dist, 0);
                val3 += wvals1(num1) * wvals2(num2) / 4.0 * (1 - r1) * r2 * dot21 * g;
                val4 -= wvals1(num1) * wvals2(num2) / 4.0 * g;
            }
        }
    }

    // Segment (2, 2)
    double dot22 = (next1 - curr1).dot(next2 - curr2);
    selectOrders(curr1, curr2, order1, order2, pvals1, pvals2, wvals1, wvals2);
    for (int num1 = 0; num1 < order1; num1++) {
        double r1 = 0.5 * pvals1(num1) + 0.5;
        for (int num2 = 0; num2 < order2; num2++) {
            double r2 = 0.5 * pvals2(num2) + 0.5;
            Vector2d p1 = (1 - r1) * curr1 + r1 * next1;
            Vector2d p2 = (1 - r2) * curr2 + r2 * next2;
            double dist = (p1 - p2).norm();
            dcomp g = -cunit / 4.0 * hankel->lookUp(k1, dist, 0);
            val1 += wvals1(num1) * wvals2(num2) / 4.0 * (1 - r1) * (1 - r2) * dot22 * g;
            val2 += wvals1(num1) * wvals2(num2) / 4.0 * g;
            if (ior2.real() != 0) {
                g = -cunit / 4.0 * hankel->lookUp(k2, dist, 0);
                val3 += wvals1(num1) * wvals2(num2) / 4.0 * (1 - r1) * (1 - r2) * dot22 * g;
                val4 += wvals1(num1) * wvals2(num2) / 4.0 * g;
            }
        }
    }

    // Matrix elements
    tt1 = (fcomp)(val1 - val2 / (k1 * k1));
    if (ior2.real() != 0)
        tt2 = (fcomp)(val3 - val4 / (k2 * k2));
    else
        tt2 = 0;
}

void MatrixBuild::Lzz(int m, int n, MatrixXd XY1, MatrixXd XY2, dcomp ior1, dcomp ior2, fcomp& zz1, fcomp& zz2) {
    dcomp k1 = 2.0 * M_PI * ior1 / lambda;
    dcomp k2 = 2.0 * M_PI * ior2 / lambda;
    dcomp b1 = 1.781 * k1 / 2.0;
    dcomp b2 = 1.781 * k2 / 2.0;

    // Locate relevant basis elements
    int N1 = XY1.rows(), N2 = XY2.rows();
    Vector2d prev1(XY1((m - 1 + N1) % N1, 0), XY1((m - 1 + N1) % N1, 1));
    Vector2d curr1(XY1(m, 0), XY1(m, 1));
    Vector2d next1(XY1((m + 1) % N1, 0), XY1((m + 1) % N1, 1));
    Vector2d prev2(XY2((n - 1 + N2) % N2, 0), XY2((n - 1 + N2) % N2, 1));
    Vector2d curr2(XY2(n, 0), XY2(n, 1));
    Vector2d next2(XY2((n + 1) % N2, 0), XY2((n + 1) % N2, 1));
    dcomp val1 = 0, val2 = 0;
    double delta1, delta2;
    int order1, order2;
    VectorXd pvals1, pvals2, wvals1, wvals2;

    // Segment (1, 1)
    delta1 = (curr1 - prev1).norm();
    delta2 = (curr2 - prev2).norm();
    selectOrders(curr1, curr2, order1, order2, pvals1, pvals2, wvals1, wvals2);
    for (int num1 = 0; num1 < order1; num1++) {
        double r1 = 0.5 * pvals1(num1) + 0.5;
        for (int num2 = 0; num2 < order2; num2++) {
            double r2 = 0.5 * pvals2(num2) + 0.5;
            Vector2d p1 = (1 - r1) * prev1 + r1 * curr1;
            Vector2d p2 = (1 - r2) * prev2 + r2 * curr2;
            double dist = (p1 - p2).norm();
            dcomp g = -cunit / 4.0 * hankel->lookUp(k1, dist, 0);
            val1 += wvals1(num1) * wvals2(num2) * delta1 * delta2 / 4.0 * r1 * r2 * g;
            if (ior2.real() != 0) {
                g = -cunit / 4.0 * hankel->lookUp(k2, dist, 0);
                val2 += wvals1(num1) * wvals2(num2) * delta1 * delta2 / 4.0 * r1 * r2 * g;
            }
        }
    }

    // Segment (1, 2)
    delta1 = (curr1 - prev1).norm();
    delta2 = (next2 - curr2).norm();
    selectOrders(curr1, next2, order1, order2, pvals1, pvals2, wvals1, wvals2);
    for (int num1 = 0; num1 < order1; num1++) {
        double r1 = 0.5 * pvals1(num1) + 0.5;
        for (int num2 = 0; num2 < order2; num2++) {
            double r2 = 0.5 * pvals2(num2) + 0.5;
            Vector2d p1 = (1 - r1) * prev1 + r1 * curr1;
            Vector2d p2 = (1 - r2) * curr2 + r2 * next2;
            double dist = (p1 - p2).norm();
            dcomp g = -cunit / 4.0 * hankel->lookUp(k1, dist, 0);
            val1 += wvals1(num1) * wvals2(num2) * delta1 * delta2 / 4.0 * r1 * (1 - r2) * g;
            if (ior2.real() != 0) {
                g = -cunit / 4.0 * hankel->lookUp(k2, dist, 0);
                val2 += wvals1(num1) * wvals2(num2) * delta1 * delta2 / 4.0 * r1 * (1 - r2) * g;
            }
        }
    }

    // Segment (2, 1)
    delta1 = (next1 - curr1).norm();
    delta2 = (curr2 - prev2).norm();
    selectOrders(next1, curr2, order1, order2, pvals1, pvals2, wvals1, wvals2);
    for (int num1 = 0; num1 < order1; num1++) {
        double r1 = 0.5 * pvals1(num1) + 0.5;
        for (int num2 = 0; num2 < order2; num2++) {
            double r2 = 0.5 * pvals2(num2) + 0.5;
            Vector2d p1 = (1 - r1) * curr1 + r1 * next1;
            Vector2d p2 = (1 - r2) * prev2 + r2 * curr2;
            double dist = (p1 - p2).norm();
            dcomp g = -cunit / 4.0 * hankel->lookUp(k1, dist, 0);
            val1 += wvals1(num1) * wvals2(num2) * delta1 * delta2 / 4.0 * (1 - r1) * r2 * g;
            if (ior2.real() != 0) {
                g = -cunit / 4.0 * hankel->lookUp(k2, dist, 0);
                val2 += wvals1(num1) * wvals2(num2) * delta1 * delta2 / 4.0 * (1 - r1) * r2 * g;
            }
        }
    }

    // Segment (2, 2)
    delta1 = (next1 - curr1).norm();
    delta2 = (next2 - curr2).norm();
    selectOrders(curr1, curr2, order1, order2, pvals1, pvals2, wvals1, wvals2);
    for (int num1 = 0; num1 < order1; num1++) {
        double r1 = 0.5 * pvals1(num1) + 0.5;
        for (int num2 = 0; num2 < order2; num2++) {
            double r2 = 0.5 * pvals2(num2) + 0.5;
            Vector2d p1 = (1 - r1) * curr1 + r1 * next1;
            Vector2d p2 = (1 - r2) * curr2 + r2 * next2;
            double dist = (p1 - p2).norm();
            dcomp g = -cunit / 4.0 * hankel->lookUp(k1, dist, 0);
            val1 += wvals1(num1) * wvals2(num2) * delta1 * delta2 / 4.0 * (1 - r1) * (1 - r2) * g;
            if (ior2.real() != 0) {
                g = -cunit / 4.0 * hankel->lookUp(k2, dist, 0);
                val2 += wvals1(num1) * wvals2(num2) * delta1 * delta2 / 4.0 * (1 - r1) * (1 - r2) * g;
            }
        }
    }

    // Matrix elements
    zz1 = (fcomp)val1;
    if (ior2.real() != 0)
        zz2 = (fcomp)val2;
    else
        zz2 = 0;
}

void MatrixBuild::Ktz(int m, int n, MatrixXd XY1, MatrixXd XY2, dcomp ior1, dcomp ior2, fcomp& tz1, fcomp& tz2) {
    dcomp k1 = 2.0 * M_PI * ior1 / lambda;
    dcomp k2 = 2.0 * M_PI * ior2 / lambda;
    dcomp b1 = 1.781 * k1 / 2.0;
    dcomp b2 = 1.781 * k2 / 2.0;

    // Locate relevant basis elements
    int N1 = XY1.rows(), N2 = XY2.rows();
    Vector2d prev1(XY1((m - 1 + N1) % N1, 0), XY1((m - 1 + N1) % N1, 1));
    Vector2d curr1(XY1(m, 0), XY1(m, 1));
    Vector2d next1(XY1((m + 1) % N1, 0), XY1((m + 1) % N1, 1));
    Vector2d prev2(XY2((n - 1 + N2) % N2, 0), XY2((n - 1 + N2) % N2, 1));
    Vector2d curr2(XY2(n, 0), XY2(n, 1));
    Vector2d next2(XY2((n + 1) % N2, 0), XY2((n + 1) % N2, 1));
    dcomp val1 = 0, val2 = 0;
    Vector3d z(0, 0, 1);
    int order1, order2;
    VectorXd pvals1, pvals2, wvals1, wvals2;

    // Segment (1, 1)
    if (curr1(0) != curr2(0) || curr1(1) != curr2(1)) {
        selectOrders(curr1, curr2, order1, order2, pvals1, pvals2, wvals1, wvals2);
        double delta = (curr2 - prev2).norm();
        Vector3d vec(curr1(0) - prev1(0), curr1(1) - prev1(1), 0);
        for (int num1 = 0; num1 < order1; num1++) {
            double r1 = 0.5 * pvals1(num1) + 0.5;
            for (int num2 = 0; num2 < order2; num2++) {
                double r2 = 0.5 * pvals2(num2) + 0.5;
                Vector2d p1 = (1 - r1) * prev1 + r1 * curr1;
                Vector2d p2 = (1 - r2) * prev2 + r2 * curr2;
                Vector3d diff(p1(0) - p2(0), p1(1) - p2(1), 0);
                double dist = (p1 - p2).norm();
                dcomp g = cunit * k1 / 4.0 * hankel->lookUp(k1, dist, 1);
                val1 += wvals1(num1) * wvals2(num2) * delta / 4.0 * r1 * r2 * g * vec.dot(diff.cross(z)) / dist;
                if (ior2.real() != 0) {
                    g = cunit * k2 / 4.0 * hankel->lookUp(k2, dist, 1);
                    val2 += wvals1(num1) * wvals2(num2) * delta / 4.0 * r1 * r2 * g * vec.dot(diff.cross(z)) / dist;
                }
            }
        }
    }

    // Segment (1, 2)
    if (curr1(0) != next2(0) || curr1(1) != next2(1)) {
        selectOrders(curr1, next2, order1, order2, pvals1, pvals2, wvals1, wvals2);
        double delta = (next2 - curr2).norm();
        Vector3d vec(curr1(0) - prev1(0), curr1(1) - prev1(1), 0);
        for (int num1 = 0; num1 < order1; num1++) {
            double r1 = 0.5 * pvals1(num1) + 0.5;
            for (int num2 = 0; num2 < order2; num2++) {
                double r2 = 0.5 * pvals2(num2) + 0.5;
                Vector2d p1 = (1 - r1) * prev1 + r1 * curr1;
                Vector2d p2 = (1 - r2) * curr2 + r2 * next2;
                Vector3d diff(p1(0) - p2(0), p1(1) - p2(1), 0);
                double dist = (p1 - p2).norm();
                dcomp g = cunit * k1 / 4.0 * hankel->lookUp(k1, dist, 1);
                val1 += wvals1(num1) * wvals2(num2) * delta / 4.0 * r1 * (1 - r2) * g * vec.dot(diff.cross(z)) / dist;
                if (ior2.real() != 0) {
                    g = cunit * k2 / 4.0 * hankel->lookUp(k2, dist, 1);
                    val2 += wvals1(num1) * wvals2(num2) * delta / 4.0 * r1 * (1 - r2) * g * vec.dot(diff.cross(z)) / dist;
                }
            }
        }
    }

    // Segment (2, 1)
    if (next1(0) != curr2(0) || next1(1) != curr2(1)) {
        selectOrders(next1, curr2, order1, order2, pvals1, pvals2, wvals1, wvals2);
        double delta = (curr2 - prev2).norm();
        Vector3d vec(next1(0) - curr1(0), next1(1) - curr1(1), 0);
        for (int num1 = 0; num1 < order1; num1++) {
            double r1 = 0.5 * pvals1(num1) + 0.5;
            for (int num2 = 0; num2 < order2; num2++) {
                double r2 = 0.5 * pvals2(num2) + 0.5;
                Vector2d p1 = (1 - r1) * curr1 + r1 * next1;
                Vector2d p2 = (1 - r2) * prev2 + r2 * curr2;
                Vector3d diff(p1(0) - p2(0), p1(1) - p2(1), 0);
                double dist = (p1 - p2).norm();
                dcomp g = cunit * k1 / 4.0 * hankel->lookUp(k1, dist, 1);
                val1 += wvals1(num1) * wvals2(num2) * delta / 4.0 * (1 - r1) * r2 * g * vec.dot(diff.cross(z)) / dist;
                if (ior2.real() != 0) {
                    g = cunit * k2 / 4.0 * hankel->lookUp(k2, dist, 1);
                    val2 += wvals1(num1) * wvals2(num2) * delta / 4.0 * (1 - r1) * r2 * g * vec.dot(diff.cross(z)) / dist;
                }
            }
        }
    }

    // Segment (2, 2)
    if (curr1(0) != curr2(0) || curr1(1) != curr2(1)) {
        selectOrders(curr1, curr2, order1, order2, pvals1, pvals2, wvals1, wvals2);
        double delta = (next2 - curr2).norm();
        Vector3d vec(next1(0) - curr1(0), next1(1) - curr1(1), 0);
        for (int num1 = 0; num1 < order1; num1++) {
            double r1 = 0.5 * pvals1(num1) + 0.5;
            for (int num2 = 0; num2 < order2; num2++) {
                double r2 = 0.5 * pvals2(num2) + 0.5;
                Vector2d p1 = (1 - r1) * curr1 + r1 * next1;
                Vector2d p2 = (1 - r2) * curr2 + r2 * next2;
                Vector3d diff(p1(0) - p2(0), p1(1) - p2(1), 0);
                double dist = (p1 - p2).norm();
                dcomp g = cunit * k1 / 4.0 * hankel->lookUp(k1, dist, 1);
                val1 += wvals1(num1) * wvals2(num2) * delta / 4.0 * (1 - r1) * (1 - r2) * g * vec.dot(diff.cross(z)) / dist;
                if (ior2.real() != 0) {
                    g = cunit * k2 / 4.0 * hankel->lookUp(k2, dist, 1);
                    val2 += wvals1(num1) * wvals2(num2) * delta / 4.0 * (1 - r1) * (1 - r2) * g * vec.dot(diff.cross(z)) / dist;
                }
            }
        }
    }

    // Matrix elements
    tz1 = (fcomp)val1;
    if (ior2.real() != 0)
        tz2 = (fcomp)val2;
    else
        tz2 = 0;
}