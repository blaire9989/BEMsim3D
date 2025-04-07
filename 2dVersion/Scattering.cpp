#include "Scattering.h"

Scattering::Scattering(MatrixXd xyvals, MatrixXi geo_info, MatrixXi mat_info, HankelTable* hankel, Incidence* inci) {
    this->xyvals = xyvals;
    this->geo_info = geo_info;
    this->mat_info = mat_info;
    this->hankel = hankel;
    this->inci = inci;
    order = 3;
    pvals = quadrature_points.block(order - 1, 0, 1, order).transpose();
    wvals = quadrature_weights.block(order - 1, 0, 1, order).transpose();
}

void Scattering::computeFarField(double lambda, VectorXcf x_TE, VectorXcf x_TM, int resolution) {
    double omega = 2.0 * M_PI * c / lambda;
    double k = 2.0 * M_PI / lambda;
    double eps = 1.0 / (mu * c * c);
    VectorXd directions = VectorXd::LinSpaced(resolution, -1.0, 1.0);
    int N = mat_info(mat_info.rows() - 1, 0) + mat_info(mat_info.rows() - 1, 1);
    VectorXd xvals = xyvals.block(geo_info(0, 0), 0, geo_info(0, 1), 1);
    VectorXd yvals = xyvals.block(geo_info(0, 0), 1, geo_info(0, 1), 1);
    VectorXcf j_TE = x_TE.block(mat_info(0, 0), 0, mat_info(0, 1), 1);
    VectorXcf m_TE = x_TE.block(N + mat_info(0, 0), 0, mat_info(0, 1), 1);
    VectorXcf j_TM = x_TM.block(mat_info(0, 0), 0, mat_info(0, 1), 1);
    VectorXcf m_TM = x_TM.block(N + mat_info(0, 0), 0, mat_info(0, 1), 1);

    // Compute far field scattering for each scattering direction
    far = MatrixXf::Zero(resolution, 2);
    parallel_for(resolution, [&](int start, int end) {
    for (int count = start; count < end; count++) {
        double phi = acos(directions(count));
        fcomp Ex, Ey, Ez, Hx, Hy, Hz;
        MatrixXcf field1 = farFieldValues(phi, k, 1, xvals, yvals, j_TE, m_TE);
        MatrixXcf field2 = farFieldValues(phi, k, 2, xvals, yvals, j_TM, m_TM);
        // TE polarization
        Ez = -(fcomp)(k * k / (4.0 * omega * eps)) * field1(2, 1) - (fcomp)(cunit * k / 4.0) * field1(2, 0);
        Hx = (fcomp)(cunit * k / 4.0) * field1(0, 1) + (fcomp)(k * k / (8.0 * omega * mu)) * field1(0, 0);
        Hy = (fcomp)(cunit * k / 4.0) * field1(1, 1) + (fcomp)(k * k / (8.0 * omega * mu)) * field1(1, 0);
        // TM polarization
        Hz = -(fcomp)(k * k / (4.0 * omega * mu)) * field2(2, 1) + (fcomp)(cunit * k / 4.0) * field2(2, 0);
        Ex = -(fcomp)(cunit * k / 4.0) * field2(0, 1) + (fcomp)(k * k / (8.0 * omega * eps)) * field2(0, 0);
        Ey = -(fcomp)(cunit * k / 4.0) * field2(1, 1) + (fcomp)(k * k / (8.0 * omega * eps)) * field2(1, 0);
        // Poynting vectors
        Vector2f p_TE, p_TM;
        p_TE(0) = 0.5f * real(-Ez * conj(Hy));
        p_TE(1) = 0.5f * real(Ez * conj(Hx));
        far(count, 0) = p_TE.norm();
        p_TM(0) = 0.5f * real(Ey * conj(Hz));
        p_TM(1) = 0.5f * real(-Ex * conj(Hz));
        far(count, 1) = p_TM.norm();
    }
    } );
}

MatrixXcf Scattering::farFieldValues(double phi, double k, int polarization, VectorXd X, VectorXd Y, VectorXcf J0, VectorXcf M0) {
    double phi_x = cos(phi);
    double phi_y = sin(phi);
    MatrixXcf field = MatrixXcf::Zero(3, 2);
    int size = X.rows();
    for (int num = 1; num < size - 1; num++) {
        Vector2d prev(X(num - 1), Y(num - 1)), curr(X(num), Y(num)), next(X(num + 1), Y(num + 1));
        double delta1 = (curr - prev).norm();
        double delta2 = (next - curr).norm();
        Vector2d t1 = (curr - prev) / delta1;
        Vector2d t2 = (next - curr) / delta2;
        fcomp jmz, jmt;
        if (polarization == 1) {
            jmz = J0(num - 1);
            jmt = M0(num - 1) * eta0f;
        } else {
            jmt = J0(num - 1);
            jmz = M0(num - 1) * eta0f;
        }
        for (int count = 0; count < order; count++) {
            double r = 0.5 * pvals(count) + 0.5;
            Vector2d pt = (1 - r) * prev + r * curr;
            dcomp g0 = sqrt(2.0 * cunit / (M_PI * k)) * exp(cunit * k * (phi_x * pt(0) + phi_y * pt(1)));
            dcomp g1 = g0 * cunit;
            dcomp g2 = -g0;
            field(0, 0) += (fcomp)(delta1 * wvals(count) / 2.0 * (((phi_y * phi_y - phi_x * phi_x) * t1(0) - 2.0 * phi_x * phi_y * t1(1)) * g2 - t1(0) * g0) * r) * jmt;
            field(1, 0) += (fcomp)(delta1 * wvals(count) / 2.0 * (((phi_x * phi_x - phi_y * phi_y) * t1(1) - 2.0 * phi_x * phi_y * t1(0)) * g2 - t1(1) * g0) * r) * jmt;
            field(2, 0) += (fcomp)(delta1 * wvals(count) / 2.0 * (phi_x * t1(1) - phi_y * t1(0)) * g1 * r) * jmt;
            field(0, 1) += (fcomp)(delta1 * wvals(count) / 2.0 * phi_y * g1 * r) * jmz;
            field(1, 1) += (fcomp)(-delta1 * wvals(count) / 2.0 * phi_x * g1 * r) * jmz;
            field(2, 1) += (fcomp)(delta1 * wvals(count) / 2.0 * g0 * r) * jmz;
        }
        for (int count = 0; count < order; count++) {
            double r = 0.5 * pvals(count) + 0.5;
            Vector2d pt = (1 - r) * curr + r * next;
            dcomp g0 = sqrt(2.0 * cunit / (M_PI * k)) * exp(cunit * k * (phi_x * pt(0) + phi_y * pt(1)));
            dcomp g1 = g0 * cunit;
            dcomp g2 = -g0;
            field(0, 0) += (fcomp)(delta2 * wvals(count) / 2.0 * (((phi_y * phi_y - phi_x * phi_x) * t2(0) - 2.0 * phi_x * phi_y * t2(1)) * g2 - t2(0) * g0) * (1 - r)) * jmt;
            field(1, 0) += (fcomp)(delta2 * wvals(count) / 2.0 * (((phi_x * phi_x - phi_y * phi_y) * t2(1) - 2.0 * phi_x * phi_y * t2(0)) * g2 - t2(1) * g0) * (1 - r)) * jmt;
            field(2, 0) += (fcomp)(delta2 * wvals(count) / 2.0 * (phi_x * t2(1) - phi_y * t2(0)) * g1 * (1 - r)) * jmt;
            field(0, 1) += (fcomp)(delta2 * wvals(count) / 2.0 * phi_y * g1 * (1 - r)) * jmz;
            field(1, 1) += (fcomp)(-delta2 * wvals(count) / 2.0 * phi_x * g1 * (1 - r)) * jmz;
            field(2, 1) += (fcomp)(delta2 * wvals(count) / 2.0 * g0 * (1 - r)) * jmz;
        }
    }
    return field;
}

void Scattering::computeNearField(double lambda, double ior, VectorXcf x_TE, VectorXcf x_TM, int x_res, double x_min, double x_max, int y_res, double y_min, double y_max, bool add_incident) {
    double omega = 2.0 * M_PI * c / lambda;
    double k1 = 2.0 * M_PI / lambda;
    double k2 = 2.0 * M_PI * ior / lambda;
    double eps1 = 1.0 / (mu * c * c);
    double eps2 = 1.0 / (mu * c * c) * ior * ior;
    near_TE = MatrixXcf::Zero(y_res, x_res);
    near_TM = MatrixXcf::Zero(y_res, x_res);
    VectorXd x_range = VectorXd::LinSpaced(x_res, x_min, x_max);
    VectorXd y_range = VectorXd::LinSpaced(y_res, y_min, y_max);

    // Separate the top and bottom surfaces
    int N = mat_info(mat_info.rows() - 1, 0) + mat_info(mat_info.rows() - 1, 1);
    VectorXd x1 = xyvals.block(geo_info(0, 0), 0, geo_info(0, 1), 1);
    VectorXd y1 = xyvals.block(geo_info(0, 0), 1, geo_info(0, 1), 1);
    VectorXd x2 = xyvals.block(geo_info(1, 0), 0, geo_info(1, 1), 1);
    VectorXd y2 = xyvals.block(geo_info(1, 0), 1, geo_info(1, 1), 1);
    VectorXcf j1_TE = x_TE.block(mat_info(0, 0), 0, mat_info(0, 1), 1);
    VectorXcf m1_TE = x_TE.block(N + mat_info(0, 0), 0, mat_info(0, 1), 1);
    VectorXcf j1_TM = x_TM.block(mat_info(0, 0), 0, mat_info(0, 1), 1);
    VectorXcf m1_TM = x_TM.block(N + mat_info(0, 0), 0, mat_info(0, 1), 1);
    VectorXcf j2_TE = x_TE.block(mat_info(1, 0), 0, mat_info(1, 1), 1);
    VectorXcf m2_TE = x_TE.block(N + mat_info(1, 0), 0, mat_info(1, 1), 1);
    VectorXcf j2_TM = x_TM.block(mat_info(1, 0), 0, mat_info(1, 1), 1);
    VectorXcf m2_TM = x_TM.block(N + mat_info(1, 0), 0, mat_info(1, 1), 1);
    
    // Compute field values at each point outside the structure
    parallel_for(x_res, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        for (int j = 0; j < y_res; j++) {
            double x0 = x_range(i), y0 = y_range(j);
            int result = locatePoint(x0, y0);
            Vector2d p0(x0, y0);
            if (result == 0) {
                Vector2cf field1 = nearFieldEHz(p0, k1, 1, x1, y1, j1_TE, m1_TE);
                Vector2cf field2 = nearFieldEHz(p0, k1, 2, x1, y1, j1_TM, m1_TM);
                near_TE(y_res - 1 - j, i) = -(fcomp)(k1 * k1 / (4.0 * omega * eps1)) * field1(1) - (fcomp)(cunit * k1 / 4.0) * field1(0);
                near_TM(y_res - 1 - j, i) = (fcomp)(cunit * k1 / 4.0) * field2(0) - (fcomp)(k1 * k1 / (4.0 * omega * mu)) * field2(1);
                if (add_incident) {
                    MatrixXcd field = MatrixXcd::Zero(3, 2);
                    if (inci->w0 == 0)
                        field = inci->plane(x0, y0);
                    else {
                        for (int count = 0; count < inci->center.rows(); count++)
                            field = field + inci->phase(count) * inci->gaussian(x0 - inci->center(count), y0 - inci->yc);
                    }
                    near_TE(y_res - 1 - j, i) += field(2, 0);
                    near_TM(y_res - 1 - j, i) += field(2, 1);
                }
            } else if (result == 1) {
                Vector2cf field1 = -nearFieldEHz(p0, k2, 1, x1, y1, j1_TE, m1_TE);
                Vector2cf field2 = -nearFieldEHz(p0, k2, 2, x1, y1, j1_TM, m1_TM);
                near_TE(y_res - 1 - j, i) += -(fcomp)(k2 * k2 / (4.0 * omega * eps2)) * field1(1) - (fcomp)(cunit * k2 / 4.0) * field1(0);
                near_TM(y_res - 1 - j, i) += (fcomp)(cunit * k2 / 4.0) * field2(0) - (fcomp)(k2 * k2 / (4.0 * omega * mu)) * field2(1);
                Vector2cf field3 = -nearFieldEHz(p0, k2, 1, x2, y2, j2_TE, m2_TE);
                Vector2cf field4 = -nearFieldEHz(p0, k2, 2, x2, y2, j2_TM, m2_TM);
                near_TE(y_res - 1 - j, i) += -(fcomp)(k2 * k2 / (4.0 * omega * eps2)) * field3(1) - (fcomp)(cunit * k2 / 4.0) * field3(0);
                near_TM(y_res - 1 - j, i) += (fcomp)(cunit * k2 / 4.0) * field4(0) - (fcomp)(k2 * k2 / (4.0 * omega * mu)) * field4(1);
            } else if (result == 2) {
                Vector2cf field1 = nearFieldEHz(p0, k1, 1, x2, y2, j2_TE, m2_TE);
                Vector2cf field2 = nearFieldEHz(p0, k1, 2, x2, y2, j2_TM, m2_TM);
                near_TE(y_res - 1 - j, i) = -(fcomp)(k1 * k1 / (4.0 * omega * eps1)) * field1(1) - (fcomp)(cunit * k1 / 4.0) * field1(0);
                near_TM(y_res - 1 - j, i) = (fcomp)(cunit * k1 / 4.0) * field2(0) - (fcomp)(k1 * k1 / (4.0 * omega * mu)) * field2(1);
            }
        }
    }
    } );
}

int Scattering::locatePoint(double x0, double y0) {
    VectorXd x1 = xyvals.block(geo_info(0, 0), 0, geo_info(0, 1), 1);
    VectorXd y1 = xyvals.block(geo_info(0, 0), 1, geo_info(0, 1), 1);
    VectorXd x2 = xyvals.block(geo_info(1, 0), 0, geo_info(1, 1), 1);
    VectorXd y2 = xyvals.block(geo_info(1, 0), 1, geo_info(1, 1), 1);
    if (x0 <= x1.minCoeff() || x0 >= x1.maxCoeff())
        return -1;
    
    // Find threshold point on the top surface
    int index1 = 0;
    while (x1(index1) <= x0)
        index1 += 1;
    double r1 = (x0 - x1(index1 - 1)) / (x1(index1) - x1(index1 - 1));
    double ytop = (1 - r1) * y1(index1 - 1) + r1 * y1(index1);

    // Find threshold point on the bottom surface
    int index2 = 0;
    while (x2(index2) <= x0)
        index2 += 1;
    double r2 = (x0 - x2(index2 - 1)) / (x2(index2) - x2(index2 - 1));
    double ybot = (1 - r2) * y2(index2 - 1) + r2 * y2(index2);

    // Locate the given point
    if (y0 >= ytop + 0.001)
        return 0;
    if (y0 <= ytop - 0.001 && y0 >= ybot + 0.001)
        return 1;
    if (y0 <= ybot - 0.001)
        return 2;
    return -1;
}

Vector2cf Scattering::nearFieldEHz(Vector2d p0, double k, int polarization, VectorXd X, VectorXd Y, VectorXcf J0, VectorXcf M0) {
    fcomp tZ = 0, zZ = 0;
    int size = J0.rows();
    for (int num = 0; num < size; num++) {
        Vector2d prev(X(num), Y(num)), curr(X(num + 1), Y(num + 1)), next(X(num + 2), Y(num + 2));
        double delta1 = (curr - prev).norm();
        double delta2 = (next - curr).norm();
        Vector2d t1 = (curr - prev) / delta1;
        Vector2d t2 = (next - curr) / delta2;
        fcomp jmz, jmt;
        if (polarization == 1) {
            jmz = J0(num);
            jmt = M0(num) * eta0f;
        } else {
            jmt = J0(num);
            jmz = M0(num) * eta0f;
        }
        for (int count = 0; count < order; count++) {
            double r = 0.5 * pvals(count) + 0.5;
            Vector2d pt = (1 - r) * prev + r * curr;
            double dist = (pt - p0).norm();
            dcomp g0 = hankel->lookUp(k, dist, 0);
            zZ += (fcomp)(delta1 * wvals(count) / 2.0 * g0 * r) * jmz;
            double rx = (p0(0) - pt(0)) / dist;
            double ry = (p0(1) - pt(1)) / dist;
            dcomp g1 = hankel->lookUp(k, dist, 1);
            tZ += (fcomp)(delta1 * wvals(count) / 2.0 * (rx * t1(1) - ry * t1(0)) * g1 * r) * jmt;
        }
        for (int count = 0; count < order; count++) {
            double r = 0.5 * pvals(count) + 0.5;
            Vector2d pt = (1 - r) * curr + r * next;
            double dist = (pt - p0).norm();
            dcomp g0 = hankel->lookUp(k, dist, 0);
            zZ += (fcomp)(delta2 * wvals(count) / 2.0 * g0 * (1 - r)) * jmz;
            double rx = (p0(0) - pt(0)) / dist;
            double ry = (p0(1) - pt(1)) / dist;
            dcomp g1 = hankel->lookUp(k, dist, 1);
            tZ += (fcomp)(delta2 * wvals(count) / 2.0 * (rx * t2(1) - ry * t2(0)) * g1 * (1 - r)) * jmt;
        }
    }
    Vector2cf field(tZ, zZ);
    return field;
}
