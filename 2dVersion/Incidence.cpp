#include "Incidence.h"

Incidence::Incidence(MatrixXd xyvals, MatrixXi geo_info, MatrixXi mat_info) {
    this->xyvals = xyvals;
    this->geo_info = geo_info;
    this->mat_info = mat_info;
    N = mat_info(mat_info.rows() - 1, 0) + mat_info(mat_info.rows() - 1, 1);
    air = 1.0;
    eps = 1 / (mu * c * c) * air * air;
    order = 3;
    pvals = quadrature_points.block(order - 1, 0, 1, order).transpose();
    wvals = quadrature_weights.block(order - 1, 0, 1, order).transpose();
}

void Incidence::computeIncidence(double lambda, double phi, double w0, double xc, double yc, int nBeams) {
    this->phi = phi;
    omega = 2 * M_PI * c / lambda;
    k = 2 * M_PI / lambda * air;
    double kx = k * cos(phi);
    double ky = k * sin(phi);
    this->yc = yc;
    center = VectorXd::LinSpaced(nBeams, xc - 0.5 * w0 * (nBeams - 1), xc + 0.5 * w0 * (nBeams - 1));
    phase = VectorXcd::Zero(nBeams);
    for (int i = 0; i < nBeams; i++)
        phase(i) = exp(cunit * (kx * center(i) + ky * yc));
    this->w0 = w0 * sin(phi);
    computeVectors();
    computeIrradiance();
}

void Incidence::computeVectors() {
    v_TE = VectorXcf::Zero(2 * N);
    v_TM = VectorXcf::Zero(2 * N);
    VectorXcd v1 = VectorXcd::Zero(mat_info(0, 1));
    VectorXcd v2 = VectorXcd::Zero(mat_info(0, 1));
    VectorXcd v3 = VectorXcd::Zero(mat_info(0, 1));
    VectorXcd v4 = VectorXcd::Zero(mat_info(0, 1));
    MatrixXd XY = xyvals.block(geo_info(0, 0), 0, geo_info(0, 1), 2);
    parallel_for(mat_info(0, 1), [&](int start, int end) {
    for (int i = start; i < end; i++) {
        Vector2d prev(XY(i, 0), XY(i, 1)), curr(XY(i + 1, 0), XY(i + 1, 1)), next(XY(i + 2, 0), XY(i + 2, 1));
        double delta1 = (curr - prev).norm();
        double delta2 = (next - curr).norm();
        Vector2d t1 = (curr - prev) / delta1;
        Vector2d t2 = (next - curr) / delta2;
        // First segment
        for (int count = 0; count < order; count++) {
            double r = 0.5 * pvals(count) + 0.5;
            Vector2d pt = (1 - r) * prev + r * curr;
            MatrixXcd field = MatrixXcd::Zero(3, 2);
            if (w0 == 0)
                field = plane(pt(0), pt(1));
            else {
                for (int j = 0; j < center.rows(); j++)
                    field = field + phase(j) * gaussian(pt(0) - center(j), pt(1) - yc);
            }
            v1(i) += delta1 / 2.0 * wvals(count) * r * (t1(0) * field(0, 0) + t1(1) * field(1, 0));
            v2(i) += delta1 / 2.0 * wvals(count) * r * field(2, 0);
            v3(i) += delta1 / 2.0 * wvals(count) * r * (t1(0) * field(0, 1) + t1(1) * field(1, 1));
            v4(i) += delta1 / 2.0 * wvals(count) * r * field(2, 1);
        }
        // Second segment
        for (int count = 0; count < order; count++) {
            double r = 0.5 * pvals(count) + 0.5;
            Vector2d pt = (1 - r) * curr + r * next;
            MatrixXcd field = MatrixXcd::Zero(3, 2);
            if (w0 == 0)
                field = plane(pt(0), pt(1));
            else {
                for (int j = 0; j < center.rows(); j++)
                    field = field + phase(j) * gaussian(pt(0) - center(j), pt(1) - yc);
            }
            v1(i) += delta2 / 2.0 * wvals(count) * (1 - r) * (t2(0) * field(0, 0) + t2(1) * field(1, 0));
            v2(i) += delta2 / 2.0 * wvals(count) * (1 - r) * field(2, 0);
            v3(i) += delta2 / 2.0 * wvals(count) * (1 - r) * (t2(0) * field(0, 1) + t2(1) * field(1, 1));
            v4(i) += delta2 / 2.0 * wvals(count) * (1 - r) * field(2, 1);
        }
    }
    } );
    v_TE.block(mat_info(0, 0), 0, mat_info(0, 1), 1) = v2.cast<fcomp>(); // Ez
    v_TE.block(N + mat_info(0, 0), 0, mat_info(0, 1), 1) = -(eta0 * v3).cast<fcomp>(); // Hx, Hy
    v_TM.block(mat_info(0, 0), 0, mat_info(0, 1), 1) = v1.cast<fcomp>(); // Ex, Ey
    v_TM.block(N + mat_info(0, 0), 0, mat_info(0, 1), 1) = -(eta0 * v4).cast<fcomp>(); // Hz
}

void Incidence::computeIrradiance() {
    double irra1 = 0, irra2 = 0;
    MatrixXd XY = xyvals.block(geo_info(0, 0), 0, geo_info(0, 1), 2);
    for (int i = 0; i < XY.rows() - 2; i++) {
        Vector2d prev(XY(i, 0), XY(i, 1)), curr(XY(i + 1, 0), XY(i + 1, 1)), next(XY(i + 2, 0), XY(i + 2, 1));
        Vector2d t = (next - prev) / (next - prev).norm();
        Vector2d n(-t(1), t(0));
        double d = 0.5 * (curr - prev).norm() + 0.5 * (next - curr).norm();
        MatrixXcd field = MatrixXcd::Zero(3, 2);
        if (w0 == 0)
            field = plane(curr(0), curr(1));
        else {
            for (int j = 0; j < center.rows(); j++)
                field = field + phase(j) * gaussian(curr(0) - center(j), curr(1) - yc);
        }
        Vector3cd Ei = field.block(0, 0, 3, 1);
        Vector3cd Hi = (field.block(0, 1, 3, 1)).conjugate();
        Vector2d poynting_TE, poynting_TM;
        poynting_TE(0) = 0.5 * (-Ei(2) * Hi(1)).real();
        poynting_TE(1) = 0.5 * (Ei(2) * Hi(0)).real();
        poynting_TM(0) = 0.5 * (Ei(1) * Hi(2)).real();
        poynting_TM(1) = 0.5 * (-Ei(0) * Hi(2)).real();
        if (poynting_TE(0) * n(0) + poynting_TE(1) * n(1) < 0)
            irra1 += d * abs(poynting_TE(0) * n(0) + poynting_TE(1) * n(1));
        if (poynting_TM(0) * n(0) + poynting_TM(1) * n(1) < 0)
            irra2 += d * abs(poynting_TM(0) * n(0) + poynting_TM(1) * n(1));
    }
    irra_TE = (float)irra1;
    irra_TM = (float)irra2;
}

MatrixXcd Incidence::plane(double x0, double y0) {
    MatrixXcd field = MatrixXcd::Zero(3, 2);
    dcomp plane = exp(cunit * k * (cos(phi) * x0 + sin(phi) * y0));

    // Fill in field values for TE polarization
    field(2, 0) = plane;
    field(0, 1) = -omega * eps / k * sin(phi) * plane;
    field(1, 1) = omega * eps / k * cos(phi) * plane;
    
    // Fill in field values for TM polarization
    field(0, 0) = -sin(phi) * plane;
    field(1, 0) = cos(phi) * plane;
    field(2, 1) = -k / (omega * mu) * plane;
    return field;
}

MatrixXcd Incidence::gaussian(double x0, double y0) {
    dcomp Ex = 0, Ey = 0, Ez = 0, Hx = 0, Hy = 0, Hz = 0;
    MatrixXcd fields(3, 2);
    for (int num = 0; num < 1440; num++) {
        double phi0 = phi - M_PI / 2 + num * M_PI / 1440;
        double coef = M_PI / 1440 * exp(-k * k * sin(phi0 - phi) * sin(phi0 - phi) * w0 * w0 / 4);
        dcomp plane = exp(cunit * k * (x0 * cos(phi0) + y0 * sin(phi0)));
        Ex += coef * sin(phi0) * plane;
        Ey += -coef * cos(phi0) * plane;
        Ez += coef * k * plane;
        Hx += -coef * sin(phi0) * plane;
        Hy += coef * cos(phi0) * plane;
        Hz += coef * k * plane;
    }

    // Fill in field values for TE polarization
    fields(2, 0) = w0 / (2 * sqrt(M_PI)) * Ez;
    fields(0, 1) = w0 / (2 * sqrt(M_PI)) * omega * eps * Hx;
    fields(1, 1) = w0 / (2 * sqrt(M_PI)) * omega * eps * Hy;

    // Fill in field values for TM polarization
    fields(0, 0) = w0 / (2 * sqrt(M_PI)) * omega * mu * air / eta0 * Ex;
    fields(1, 0) = w0 / (2 * sqrt(M_PI)) * omega * mu * air / eta0 * Ey;
    fields(2, 1) = w0 / (2 * sqrt(M_PI)) * air / eta0 * Hz;
    return fields;
}
