#include <unistd.h>
#include "Incidence.h"
#include "Scattering.h"
#include "Solver.h"

int main(int argc, char **argv) {
    int opt, resolution, type;
    double n1, n2, lambda, phi;
    string xyname;
    while ((opt = getopt(argc, argv, "a:l:p:r:t:z:")) != -1) {
        switch (opt) {
            case 'a': n1 = atof(optarg); break;
            case 'l': lambda = atof(optarg) * 0.001; break;
            case 'p': phi = atof(optarg) * M_PI / 180; break;
            case 'r': resolution = atoi(optarg); break;
            case 't': type = atoi(optarg); break;
            case 'z': xyname = optarg; break;
        }
    }
    MatrixXd xyvals = readData("data/" + xyname + "/xyvals.txt");
    MatrixXi info = readData("data/" + xyname + "/info.txt").cast<int>();
    HankelTable* hankel = new HankelTable(lambda);
    MatrixBuild* mat = new MatrixBuild(xyvals, info, hankel);
    mat->assemble(lambda, n1);
    Incidence* inci = new Incidence(xyvals, mat->geo_info, mat->mat_info);
    Solver* solver = new Solver(mat);
    Scattering* scatter = new Scattering(xyvals, mat->geo_info, mat->mat_info, hankel, inci);
    
    // Locate critical points in the structure
    double thickness = abs(xyvals(0, 1) - xyvals(info(1, 0), 1));
    double xcorner, ycorner;
    for (int i = info(1, 0); i < xyvals.rows(); i++) {
        if (abs(xyvals(0, 1) - xyvals(i, 1)) > thickness) {
            xcorner = xyvals(i, 0);
            ycorner = xyvals(i, 1);
            break;
        }
    }
    double xbottom = 0;
    double ybottom = xyvals.block(0, 1, xyvals.rows(), 1).minCoeff();
    double tilt = phi - 0.5 * M_PI;
    tilt = asin(sin(tilt) / n1);
    VectorXd w0 = VectorXd::Zero(6);
    VectorXd xc = VectorXd::Zero(6);
    VectorXd yc = VectorXd::Zero(6);
    VectorXi nBeams = VectorXi::Zero(6);
    
    // Incidence condition 1: full illumination, flat top
    w0(0) = 3.0;
    double xleft = xyvals(0, 0);
    double xright = xyvals(info(1, 0) - 1, 0) - thickness * tan(tilt);
    double spacing = xright - xleft;
    xc(0) = 0.5 * (xleft + xright);
    nBeams(0) = floor(spacing / w0(0) - 4);
    
    // Incidence condition 2: left illumination, narrow Gaussian
    w0(1) = 2.0;
    xc(1) = -abs(xcorner) - abs(ycorner) * tan(tilt) + 0.5 * w0(1);
    nBeams(1) = 1;
    
    // Incidence condition 3: right illumination, narrow Gaussian, position 1
    w0(2) = 2.0;
    xc(2) = abs(xcorner) - abs(ycorner) * tan(tilt) - 0.4 * w0(2);
    nBeams(2) = 1;
    
    // Incidence condition 4: right illumination, narrow Gaussian, position 2
    w0(3) = 2.0;
    xc(3) = abs(xcorner) - abs(ycorner) * tan(tilt) - 0.3 * w0(3);
    nBeams(3) = 1;
    
    // Incidence condition 5: right illumination, narrow Gaussian, position 3
    w0(4) = 2.0;
    xc(4) = abs(xcorner) - abs(ycorner) * tan(tilt) - 0.2 * w0(4);
    nBeams(4) = 1;
    
    // Incidence condition 6: right illumination, narrow Gaussian, position 4
    w0(5) = 2.0;
    xc(5) = abs(xcorner) - abs(ycorner) * tan(tilt) - 0.1 * w0(5);
    nBeams(5) = 1;
    
    // Temporary code that chooses which simulations to run
    vector<int> indices;
    if (type == 1)
        indices.push_back(1);
    else if (type == 2) {
        indices.push_back(0);
        indices.push_back(1);
    } else if (type == 3) {
        indices.push_back(2);
        indices.push_back(3);
        indices.push_back(4);
        indices.push_back(5);
    }
    
    // Perform all wave simulations
    for (int count = 0; count < indices.size(); count++) {
        int i = indices[count];
        inci->computeIncidence(lambda, phi, w0(i), xc(i), yc(i), nBeams(i));
        solver->transferMatrix(1);
        solver->csMINRES(inci->v_TE, 5000, 1e-4);
        VectorXcf x_TE = solver->x;
        solver->transferMatrix(2);
        solver->csMINRES(inci->v_TM, 5000, 1e-4);
        VectorXcf x_TM = solver->x;
        scatter->computeFarField(lambda, x_TE, x_TM, resolution);
        MatrixXf refl = MatrixXf::Zero(resolution, 2);
        refl.block(0, 0, resolution, 1) = scatter->far.block(0, 0, resolution, 1) / inci->irra_TE;
        refl.block(0, 1, resolution, 1) = scatter->far.block(0, 1, resolution, 1) / inci->irra_TM;
        writeData("data/" + xyname + "/far_" + to_string(i) + "_" + to_string((int)round(phi * 180.0 / M_PI - 90.0)) + "_" + to_string((int)round(1000 * lambda)) + ".binary", refl);
        if (lambda > 0.5495 && lambda < 0.5505) {
            double xmin = 0.8 * xyvals.block(0, 0, xyvals.rows(), 1).minCoeff();
            double xmax = 0.8 * xyvals.block(0, 0, xyvals.rows(), 1).maxCoeff();
            double y1 = xyvals.block(0, 1, xyvals.rows(), 1).minCoeff();
            double y2 = xyvals.block(0, 1, xyvals.rows(), 1).maxCoeff();
            double ymin = y1 - 0.2 * (y2 - y1);
            double ymax = y2 + 0.2 * (y2 - y1);
            int xres = ceil((xmax - xmin) / 0.02);
            xres = 20 * ceil(0.05 * xres);
            int yres = ceil((ymax - ymin) / 0.02);
            yres = 20 * ceil(0.05 * yres);
            scatter->computeNearField(lambda, n1, x_TE, x_TM, xres, xmin, xmax, yres, ymin, ymax, true);
            MatrixXf near_real = scatter->near_TE.real();
            MatrixXf near_imag = scatter->near_TE.imag();
            writeBinary("data/" + xyname + "/nearR_" + to_string(i) + "_" + to_string((int)round(phi * 180.0 / M_PI - 90.0)) + ".binary", near_real);
            writeBinary("data/" + xyname + "/nearI_" + to_string(i) + "_" + to_string((int)round(phi * 180.0 / M_PI - 90.0)) + ".binary", near_imag);
        }
    }
    delete hankel;
    delete mat;
    delete inci;
    delete solver;
    delete scatter;
}
