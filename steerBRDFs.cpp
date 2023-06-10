#include <unistd.h>
#include "constants.h"
#include "parallel.h"

/// @brief Evaluate the field from a given Gaussian beam at a particular point in space
/// @brief Users should not try to understand or modify this function
MatrixXcd gaussianBeam(double x0, double y0, double z0, dcomp scale, int polarization, double k, double omega, double eps, double zR0, double zR1, Matrix3d R1, Matrix3d R2) {
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

int main(int argc, char **argv) {
    int opt, color, dir, nx, ny, outres;
    double exterior, w_um, x_um, y_um;
    string zname;
    while ((opt = getopt(argc, argv, "c:d:e:m:n:o:w:x:y:z:")) != -1) {
        switch (opt) {
            case 'c': color = atoi(optarg); break;
            case 'd': dir = atoi(optarg); break;
            case 'e': exterior = atof(optarg); break;
            case 'm': nx = atoi(optarg); break;
            case 'n': ny = atoi(optarg); break;
            case 'o': outres = atoi(optarg); break;
            case 'w': w_um = atof(optarg); break;
            case 'x': x_um = atof(optarg); break;
            case 'y': y_um = atof(optarg); break;
            case 'z': zname = optarg; break;
        }
    }

    // Determine the considered Gaussian beam parameters
    MatrixXd wvl = readData("data/" + zname + "/wvl.txt");
    double lambda = wvl(color, 0);
    double k = 2 * M_PI * exterior / lambda;
    double omega = c / lambda * 2 * M_PI;
    double eps = 1 / (mu * c * c) * exterior * exterior;
    MatrixXd basic;
    readBinary("data/" + zname + "/basic.binary", basic);
    double theta_c = basic(dir, 0);
    double phi_c = basic(dir, 1);
    double w0 = w_um * cos(theta_c);
    double w1 = w_um;
    double zR0 = M_PI * w0 * w0 * exterior / lambda;
    double zR1 = M_PI * w1 * w1 * exterior / lambda;
    Matrix3d R1, R2;
    R1 << -cos(theta_c) * cos(phi_c), -cos(theta_c) * sin(phi_c), sin(theta_c),
          -sin(phi_c), cos(phi_c), 0,
          -sin(theta_c) * cos(phi_c), -sin(theta_c) * sin(phi_c), -cos(theta_c);
    R2 << -cos(theta_c) * cos(phi_c), -sin(phi_c), -sin(theta_c) * cos(phi_c),
          -cos(theta_c) * sin(phi_c), cos(phi_c), -sin(theta_c) * sin(phi_c),
          sin(theta_c), 0, -cos(theta_c);
    MatrixXcd standards = gaussianBeam(0, 0, 0, 1, 1, k, omega, eps, zR0, zR1, R1, R2);
    standards = R1 * standards;
    dcomp scale = standards(0, 0);

    // Determine the size of the entire target surface and the underlying height field resolution
    MatrixXd zvals = readData("data/" + zname + "/zvals.txt");
    int Nx = zvals.rows() - 1;
    int Ny = zvals.cols() - 1;
    double d = x_um / Nx;
    
    // Determine the size of each simulated subregion, assuming the shifts between subregions equal the individual Gaussian beam waist (w_um)
    int nshift = round(w_um / d);
    int Nx_sub = Nx - (nx - 1) * nshift;
    int Ny_sub = Ny - (ny - 1) * nshift;
    VectorXd xvals = VectorXd::LinSpaced(Nx_sub + 1, -Nx_sub * d / 2, Nx_sub * d / 2);
    VectorXd yvals = VectorXd::LinSpaced(Ny_sub + 1, -Ny_sub * d / 2, Ny_sub * d / 2);

    // Compute the (identical) Gaussian beam incident field on each simulated subregion of the surface
    MatrixXcd Ex1_sub = MatrixXcd::Zero(Nx_sub, Ny_sub);
    MatrixXcd Ey1_sub = MatrixXcd::Zero(Nx_sub, Ny_sub);
    MatrixXcd Hx1_sub = MatrixXcd::Zero(Nx_sub, Ny_sub);
    MatrixXcd Hy1_sub = MatrixXcd::Zero(Nx_sub, Ny_sub);
    MatrixXcd Ex2_sub = MatrixXcd::Zero(Nx_sub, Ny_sub);
    MatrixXcd Ey2_sub = MatrixXcd::Zero(Nx_sub, Ny_sub);
    MatrixXcd Hx2_sub = MatrixXcd::Zero(Nx_sub, Ny_sub);
    MatrixXcd Hy2_sub = MatrixXcd::Zero(Nx_sub, Ny_sub);
    parallel_for(Nx_sub, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        for (int j = 0; j < Ny_sub; j++) {
            double x0 = 0.5 * (xvals(i) + xvals(i + 1));
            double y0 = 0.5 * (yvals(j) + yvals(j + 1));
            MatrixXcd fields;
            fields = gaussianBeam(x0, y0, 0, scale, 1, k, omega, eps, zR0, zR1, R1, R2);
            Ex1_sub(i, j) = fields(0, 0);
            Ey1_sub(i, j) = fields(1, 0);
            Hx1_sub(i, j) = fields(0, 1);
            Hy1_sub(i, j) = fields(1, 1);
            fields = gaussianBeam(x0, y0, 0, scale, 2, k, omega, eps, zR0, zR1, R1, R2);
            Ex2_sub(i, j) = fields(0, 0);
            Ey2_sub(i, j) = fields(1, 0);
            Hx2_sub(i, j) = fields(0, 1);
            Hy2_sub(i, j) = fields(1, 1);
        }
    }
    } );

    // Synthesize BRDFs for queried incident directions associated to the given basic incident direction
    MatrixXd query;
    readBinary("data/" + zname + "/query.binary", query);
    int numQ = query.rows();
    for (int q = 0; q < numQ; q++) {
        if (round(query(q, 2)) != dir)
            continue;
        double theta_i = query(q, 0);
        double phi_i = query(q, 1);
        double xc = sin(theta_i) * cos(phi_i);
        double yc = sin(theta_i) * sin(phi_i);

        // Compute the combined incident field on the entire surface by adding together subregion incident beams
        MatrixXcd Ex1 = MatrixXcd::Zero(Nx, Ny);
        MatrixXcd Ey1 = MatrixXcd::Zero(Nx, Ny);
        MatrixXcd Hx1 = MatrixXcd::Zero(Nx, Ny);
        MatrixXcd Hy1 = MatrixXcd::Zero(Nx, Ny);
        MatrixXcd Ex2 = MatrixXcd::Zero(Nx, Ny);
        MatrixXcd Ey2 = MatrixXcd::Zero(Nx, Ny);
        MatrixXcd Hx2 = MatrixXcd::Zero(Nx, Ny);
        MatrixXcd Hy2 = MatrixXcd::Zero(Nx, Ny);
        for (int bx = 0; bx < nx; bx++) {
            for (int by = 0; by < ny; by++) {
                double xshift = 0.5 * w_um * (2 * bx - nx + 1);
                double yshift = 0.5 * w_um * (2 * by - ny + 1);
                dcomp phase_shift = exp(cuDB * k * (xc * xshift + yc * yshift));
                int xstart = bx * nshift;
                int ystart = by * nshift;
                Ex1.block(xstart, ystart, Nx_sub, Ny_sub) += phase_shift * Ex1_sub;
                Ey1.block(xstart, ystart, Nx_sub, Ny_sub) += phase_shift * Ey1_sub;
                Hx1.block(xstart, ystart, Nx_sub, Ny_sub) += phase_shift * Hx1_sub;
                Hy1.block(xstart, ystart, Nx_sub, Ny_sub) += phase_shift * Hy1_sub;
                Ex2.block(xstart, ystart, Nx_sub, Ny_sub) += phase_shift * Ex2_sub;
                Ey2.block(xstart, ystart, Nx_sub, Ny_sub) += phase_shift * Ey2_sub;
                Hx2.block(xstart, ystart, Nx_sub, Ny_sub) += phase_shift * Hx2_sub;
                Hy2.block(xstart, ystart, Nx_sub, Ny_sub) += phase_shift * Hy2_sub;
            }
        }
        double irradiance1 = 0;
        double irradiance2 = 0;
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                double poynting;
                poynting = 0.5 * (Ex1(i, j) * conj(Hy1(i, j)) - Ey1(i, j) * conj(Hx1(i, j))).real();
                irradiance1 += abs(poynting);
                poynting = 0.5 * (Ex2(i, j) * conj(Hy2(i, j)) - Ey2(i, j) * conj(Hx2(i, j))).real();
                irradiance2 += abs(poynting);
            }
        }
        irradiance1 = d * d * irradiance1;
        irradiance2 = d * d * irradiance2;

        // Combine scattered fields from each subregion simulation
        MatrixXcf EH1 = MatrixXcf::Zero(2 * outres, 3 * outres);
        MatrixXcf EH2 = MatrixXcf::Zero(2 * outres, 3 * outres);
        for (int bx = 0; bx < nx; bx++) {
            for (int by = 0; by < ny; by++) {
                double xshift = 0.5 * w_um * (2 * bx - nx + 1);
                double yshift = 0.5 * w_um * (2 * by - ny + 1);
                fcomp phase_shift = (fcomp)(exp(cuDB * k * (xc * xshift + yc * yshift)));
                MatrixXf EH1_real, EH1_imag, EH2_real, EH2_imag;
                readBinary("data/" + zname + "/patch" + to_string(bx) + to_string(by) + "/EH1_real.binary", EH1_real);
                readBinary("data/" + zname + "/patch" + to_string(bx) + to_string(by) + "/EH1_imag.binary", EH1_imag);
                readBinary("data/" + zname + "/patch" + to_string(bx) + to_string(by) + "/EH2_real.binary", EH2_real);
                readBinary("data/" + zname + "/patch" + to_string(bx) + to_string(by) + "/EH2_imag.binary", EH2_imag);
                EH1 = EH1 + phase_shift * (EH1_real + cuFL * EH1_imag);
                EH2 = EH2 + phase_shift * (EH2_real + cuFL * EH2_imag);
            }
        }

        // Compute BRDF value for each outgoing direction
        MatrixXf brdf = MatrixXf::Zero(outres, outres);
        parallel_for(outres, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            for (int j = 0; j < outres; j++) {
                float r1 = (i + 0.5f) / outres * 2.0f - 1.0f;
                float r2 = (j + 0.5f) / outres * 2.0f - 1.0f;
                float test = 1 - r1 * r1 - r2 * r2;
                if (test < 0)
                    continue;
                float cos_r = sqrt(test);
                fcomp Ex1 = EH1(0 * outres + i, 0 * outres + j);
                fcomp Ey1 = EH1(0 * outres + i, 1 * outres + j);
                fcomp Ez1 = EH1(0 * outres + i, 2 * outres + j);
                fcomp Hx1 = conj(EH1(1 * outres + i, 0 * outres + j));
                fcomp Hy1 = conj(EH1(1 * outres + i, 1 * outres + j));
                fcomp Hz1 = conj(EH1(1 * outres + i, 2 * outres + j));
                Vector3f poynting1;
                poynting1(0) = 0.5f * (Ey1 * Hz1 - Ez1 * Hy1).real();
                poynting1(1) = 0.5f * (Ez1 * Hx1 - Ex1 * Hz1).real();
                poynting1(2) = 0.5f * (Ex1 * Hy1 - Ey1 * Hx1).real();
                brdf(i, j) += 0.5f * poynting1.norm() / (cos_r * (float)irradiance1);
                fcomp Ex2 = EH2(0 * outres + i, 0 * outres + j);
                fcomp Ey2 = EH2(0 * outres + i, 1 * outres + j);
                fcomp Ez2 = EH2(0 * outres + i, 2 * outres + j);
                fcomp Hx2 = conj(EH2(1 * outres + i, 0 * outres + j));
                fcomp Hy2 = conj(EH2(1 * outres + i, 1 * outres + j));
                fcomp Hz2 = conj(EH2(1 * outres + i, 2 * outres + j));
                Vector3f poynting2;
                poynting2(0) = 0.5f * (Ey2 * Hz2 - Ez2 * Hy2).real();
                poynting2(1) = 0.5f * (Ez2 * Hx2 - Ex2 * Hz2).real();
                poynting2(2) = 0.5f * (Ex2 * Hy2 - Ey2 * Hx2).real();
                brdf(i, j) += 0.5f * poynting2.norm() / (cos_r * (float)irradiance2);
            }
        }
        } );
        writeData("data/" + zname + "/BRDF_wvl" + to_string(color) + "_wi" + to_string(q) + ".binary", brdf);
    }
}