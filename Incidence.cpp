/* This module computes the b vector in the BEM linear system Ax = b.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "Incidence.h"

/// @brief Compute and construct the right-hand-side vector in the BEM linear system
/// @param est: an Estimate object
Incidence::Incidence(Estimate* est) {
    this->Nx = est->Nx;
    this->Ny = est->Ny;
    this->d = est->d;
    this->zvals = est->zvals;
    xvals = VectorXd::LinSpaced(Nx + 1, -Nx * d / 2, Nx * d / 2);
    yvals = VectorXd::LinSpaced(Ny + 1, -Ny * d / 2, Ny * d / 2);
    hori_num = (Nx - 1) * Ny;
    vert_num = (Ny - 1) * Nx;
    b.resize(2 * (hori_num + vert_num));

    // Numerical quadrature (order 2 is sufficient)
    order = 2;
    pvals = quadrature_points.block(order - 1, 0, 1, order).transpose();
    wvals = quadrature_weights.block(order - 1, 0, 1, order).transpose();
}

/// @brief Update the Gaussian beam parameters for the specified medium, wavelength, beam waist, and incident direction
/// @param eta: index of refraction of the medium where the light is incident from, usually set to 1.0 (air)
/// @param lambda: the currently simulated wavelength
/// @param w: the primary waist of the Gaussian beam (see our paper for explanations)
/// @param theta_i: the zenith (theta) angle of the incident direction, in spherical coordinate
/// @param phi_i: the azimuth (phi) angle of the incident direction, in spherical coordinate
void Incidence::setParameters(double eta, double lambda, double w, double theta_i, double phi_i) {
    this->eta = eta;
    k = 2 * M_PI / lambda * eta;
    eps = 1 / (mu * c * c) * eta * eta;
    omega = c / lambda * 2 * M_PI;

    // Compute beam waists along the two axes, according to the primary waist and the incident direction
    w0 = w * cos(theta_i);
    w1 = w;
    zR0 = M_PI * w0 * w0 * eta / lambda;
    zR1 = M_PI * w1 * w1 * eta / lambda;

    // Helper rotation matrices for convenient field value evaluations
    R1 << -cos(theta_i) * cos(phi_i), -cos(theta_i) * sin(phi_i), sin(theta_i),
          -sin(phi_i), cos(phi_i), 0,
          -sin(theta_i) * cos(phi_i), -sin(theta_i) * sin(phi_i), -cos(theta_i);
    R2 << -cos(theta_i) * cos(phi_i), -sin(phi_i), -sin(theta_i) * cos(phi_i),
          -cos(theta_i) * sin(phi_i), cos(phi_i), -sin(theta_i) * sin(phi_i),
          sin(theta_i), 0, -cos(theta_i);
    
    // Normalize the field values to have an amplitude of 1.0 at the origin
    MatrixXcd standards = gaussian(0, 0, 0, 1, 1);
    standards = R1 * standards;
    scale_factor = standards(0, 0);
    computePower();
}

/// @brief Compute the incident power by integrating the irradiance over the surface (for two mutually perpendicular linear polarizations)
void Incidence::computePower() {
    irradiance1 = 0;
    irradiance2 = 0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x0 = 0.5 * (xvals(i) + xvals(i + 1));
            double y0 = 0.5 * (yvals(j) + yvals(j + 1));
            double z0 = 0.25 * (zvals(i, j) + zvals(i + 1, j) + zvals(i, j + 1) + zvals(i + 1, j + 1));
            MatrixXcd field1 = gaussian(x0, y0, z0, scale_factor, 1);
            Vector3cd E1 = field1.block(0, 0, 3, 1);
            Vector3cd H1 = (field1.block(0, 1, 3, 1)).conjugate();
            double poynting1 = 0.5 * (E1(0) * H1(1) - E1(1) * H1(0)).real();
            irradiance1 += d * d * abs(poynting1);
            MatrixXcd field2 = gaussian(x0, y0, z0, scale_factor, 2);
            Vector3cd E2 = field2.block(0, 0, 3, 1);
            Vector3cd H2 = (field2.block(0, 1, 3, 1)).conjugate();
            double poynting2 = 0.5 * (E2(0) * H2(1) - E2(1) * H2(0)).real();
            irradiance2 += d * d * abs(poynting2);
        }
    }
}

/// @brief Compute the right-hand-side vector in the BEM linear system, for a given polarization
/// @brief The choice of polarization directions are embedded in our code and this is just one possible way of choosing them
/// @brief Simulations are usually run for both polarizations, and the simulated results are averaged
/// @param polarization: an integer with the value of 1 or 2, indicating which polarization we use
void Incidence::computeVector(int polarization) {
    // Vector segments associated with the horizontally arranged basis functions
    VectorXcd b1 = VectorXcd::Zero(hori_num);
    VectorXcd b3 = VectorXcd::Zero(hori_num);
    parallel_for(hori_num, [&](int start, int end) {
    for (int m = start; m < end; m++) {
        int my = m / (Nx - 1);
        int mx1 = m - (Nx - 1) * my;
        int mx2 = mx1 + 1;
        Vector2i mx(mx1, mx2);
        for (int m_count = 0; m_count < 2; m_count++) {
            double sign = pow(-1.0, m_count);
            int mx0 = mx(m_count);
            int my0 = my;
            double x1 = xvals(mx0);
            double x2 = xvals(mx0 + 1);
            double y1 = yvals(my0);
            double y2 = yvals(my0 + 1);
            double z1 = zvals(mx0, my0);
            double z2 = zvals(mx0 + 1, my0);
            double z3 = zvals(mx0, my0 + 1);
            double z4 = zvals(mx0 + 1, my0 + 1);
            double z11 = (z1 - z2 - z3 + z4) / 4;
            double z10 = (-z1 + z2 - z3 + z4) / 4;
            for (int dim_u = 0; dim_u < order; dim_u++) {
                for (int dim_v = 0; dim_v < order; dim_v++) {
                    double u = pvals(dim_u);
                    double v = pvals(dim_v);
                    double posx = x1 * (1 - u) / 2 + x2 * (1 + u) / 2;
                    double posy = y1 * (1 - v) / 2 + y2 * (1 + v) / 2;
                    double posz = z1 * (u - 1) * (v - 1) / 4 - z2 * (u + 1) * (v - 1) / 4 - z3 * (u - 1) * (v + 1) / 4 + z4 * (u + 1) * (v + 1) / 4;
                    MatrixXcd fields = gaussian(posx, posy, posz, scale_factor, polarization);
                    b1(m) = b1(m) + wvals(dim_u) * wvals(dim_v) * (1 + sign * u) * (0.5 * d * fields(0, 0) + (z11 * v + z10) * fields(2, 0));
                    b3(m) = b3(m) + wvals(dim_u) * wvals(dim_v) * (1 + sign * u) * (0.5 * d * fields(0, 1) + (z11 * v + z10) * fields(2, 1));
                }
            }
        }
    }
    } );

    // Vector segments associated with the vertically arranged basis functions
    VectorXcd b2 = VectorXcd::Zero(vert_num);
    VectorXcd b4 = VectorXcd::Zero(vert_num);
    parallel_for(vert_num, [&](int start, int end) {
    for (int m = start; m < end; m++) {
        int mx = m / (Ny - 1);
        int my1 = m - (Ny - 1) * mx;
        int my2 = my1 + 1;
        Vector2i my(my1, my2);
        for (int m_count = 0; m_count < 2; m_count++) {
            double sign = pow(-1.0, m_count);
            int mx0 = mx;
            int my0 = my(m_count);
            double x1 = xvals(mx0);
            double x2 = xvals(mx0 + 1);
            double y1 = yvals(my0);
            double y2 = yvals(my0 + 1);
            double z1 = zvals(mx0, my0);
            double z2 = zvals(mx0 + 1, my0);
            double z3 = zvals(mx0, my0 + 1);
            double z4 = zvals(mx0 + 1, my0 + 1);
            double z11 = (z1 - z2 - z3 + z4) / 4;
            double z01 = (-z1 - z2 + z3 + z4) / 4;
            for (int dim_u = 0; dim_u < order; dim_u++) {
                for (int dim_v = 0; dim_v < order; dim_v++) {
                    double u = pvals(dim_u);
                    double v = pvals(dim_v);
                    double posx = x1 * (1 - u) / 2 + x2 * (1 + u) / 2;
                    double posy = y1 * (1 - v) / 2 + y2 * (1 + v) / 2;
                    double posz = z1 * (u - 1) * (v - 1) / 4 - z2 * (u + 1) * (v - 1) / 4 - z3 * (u - 1) * (v + 1) / 4 + z4 * (u + 1) * (v + 1) / 4;
                    MatrixXcd fields = gaussian(posx, posy, posz, scale_factor, polarization);
                    b2(m) = b2(m) + wvals(dim_u) * wvals(dim_v) * (1 + sign * v) * (0.5 * d * fields(1, 0) + (z11 * u + z01) * fields(2, 0));
                    b4(m) = b4(m) + wvals(dim_u) * wvals(dim_v) * (1 + sign * v) * (0.5 * d * fields(1, 1) + (z11 * u + z01) * fields(2, 1));
                }
            }
        }
    }
    } );

    // Constructing the b vector in the BEM linear system Ax = b
    b.block(0, 0, hori_num, 1) = b1.cast<fcomp>();
    b.block(hori_num, 0, vert_num, 1) = b2.cast<fcomp>();
    b.block(hori_num + vert_num, 0, hori_num, 1) = -(b3 * eta0DB).cast<fcomp>();
    b.block(2 * hori_num + vert_num, 0, vert_num, 1) = -(b4 * eta0DB).cast<fcomp>();
}

/// @brief Evaluate the given Gaussian beam field value at a given point in space
/// @param x0: x-coordinate of the point position
/// @param y0: y-coordinate of the point position 
/// @param z0: z-coordinate of the point position 
/// @param scale: a scale factor that normalizes the field values to have an amplitude of 1.0 at the origin
/// @param polarization: an integer with the value of 1 or 2, indicating which polarization we use
/// @return The Ex, Ey, Ez, Hx, Hy, Hz components of the field at the queried point
MatrixXcd Incidence::gaussian(double x0, double y0, double z0, dcomp scale, int polarization) {
    Vector3d pos(x0, y0, z0);
    pos = R1 * pos;
    double x = pos(0);
    double y = pos(1);
    double z = pos(2);
    dcomp q0 = z + zR0 * cuDB;
    dcomp q1 = z + zR1 * cuDB;
    dcomp q2 = pow(q0, -0.5);
    dcomp q3 = pow(q1, -0.5);
    dcomp q4 = pow(q0, -1.5);
    dcomp q5 = pow(q1, -1.5);
    dcomp q6 = pow(q0, -2.5);
    dcomp q7 = pow(q1, -2.5);
    dcomp q8 = pow(q0, -3.5);
    dcomp q9 = pow(q1, -3.5);
    dcomp expr0 = sqrt(zR0 * zR1) * exp(-k * cuDB * (0.5 * x * x / q0 + 0.5 * y * y / q1 + z));
    dcomp expr1 = x * x / (2.0 * q0 * q0) + y * y / (2.0 * q1 * q1) - 1.0;
    dcomp U = cuDB * q2 * q3 * expr0;
    dcomp dUdx = k * x * expr0 * q3 * q4;
    dcomp dUdy = k * y * expr0 * q2 * q5;
    dcomp dUdz = -0.5 * cuDB * expr0 * q3 * q4 - 0.5 * cuDB * expr0 * q2 * q5 - k * expr0 * expr1 * q2 * q3;
    dcomp dUdx2 = -k * k * cuDB * x * x * expr0 * q3 * q6 + k * expr0 * q3 * q4;
    dcomp dUdy2 = -k * k * cuDB * y * y * expr0 * q2 * q7 + k * expr0 * q2 * q5;
    dcomp dUdxy = -k * k * cuDB * x * y * expr0 * q4 * q5;
    dcomp dUdxz = k * k * cuDB * x * expr0 * expr1 * q3 * q4 - 1.5 * k * x * expr0 * q3 * q6 - 0.5 * k * x * expr0 * q4 * q5;
    dcomp dUdyz = k * k * cuDB * y * expr0 * expr1 * q2 * q5 - 1.5 * k * y * expr0 * q2 * q7 - 0.5 * k * y * expr0 * q4 * q5;
    Vector3cd Ei(0, 0, 0), Hi(0, 0, 0);
    if (polarization == 1) {
        Ei(0) = dUdx2 + k * k * U;
        Ei(1) = dUdxy;
        Ei(2) = dUdxz;
        Hi(1) = dUdz;
        Hi(2) = -dUdy;
    } else {
        Ei(0) = dUdxy;
        Ei(1) = dUdy2 + k * k * U;
        Ei(2) = dUdyz;
        Hi(0) = -dUdz;
        Hi(2) = dUdx;
    }
    Ei = R2 * Ei / (cuDB * omega * eps * scale);
    Hi = R2 * Hi / scale;
    MatrixXcd fields(3, 2);
    fields << Ei, Hi;
    return fields;
}