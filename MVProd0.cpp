/* CPU implementation for computing matrix-vector products in BEM.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "MVProd0.h"

/// @brief CPU implementation for computing matrix-vector products in BEM
/// @param est: an Estimate object with information on the simulated surface
/// @param singular: a Singular object that computes the matrix elements with singularties in the underlying integrals
/// @param grid: a Grid object with information on the 3D grid of point sources
MVProd0::MVProd0(Estimate* est, Singular* singular, Grid* grid): MVProd(est, singular, grid) {
    xvals = VectorXf::LinSpaced(Nx + 1, -Nx * d / 2, Nx * d / 2);
    yvals = VectorXf::LinSpaced(Ny + 1, -Ny * d / 2, Ny * d / 2);
    zvals = est->zvals.cast<float>();
    numX = grid->numX;
    numY = grid->numY;
    numZ = grid->numZ;

    // Some numerical quadratures for numerical integration
    orderH = 4;
    orderL = 2;
    pH = (quadrature_points.block(orderH - 1, 0, 1, orderH).transpose()).cast<float>();
    pL = (quadrature_points.block(orderL - 1, 0, 1, orderL).transpose()).cast<float>();
    wH = (quadrature_weights.block(orderH - 1, 0, 1, orderH).transpose()).cast<float>();
    wL = (quadrature_weights.block(orderL - 1, 0, 1, orderL).transpose()).cast<float>();

    // Allocate FFT arrays and create FFT computation plans
    int n[] = {totalX, totalY, totalZ};
    data1.resize(N, 1);
    data3.resize(N, 3);
    data41.resize(N, 4);
    data42.resize(N, 4);
    array1 = (fftwf_complex*) data1.data();
    array3 = (fftwf_complex*) data3.data();
    array41 = (fftwf_complex*) data41.data();
    array42 = (fftwf_complex*) data42.data();
    fftwf_init_threads();
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);
    int num_thr = (int) nb_threads;
    fftwf_plan_with_nthreads(num_thr);
    forward2 = fftwf_plan_many_dft(3, n, 2, array3, n, 1, N, array3, n, 1, N, FFTW_FORWARD, FFTW_MEASURE);
    forward3 = fftwf_plan_many_dft(3, n, 3, array3, n, 1, N, array3, n, 1, N, FFTW_FORWARD, FFTW_MEASURE);
    forward41 = fftwf_plan_many_dft(3, n, 4, array41, n, 1, N, array41, n, 1, N, FFTW_FORWARD, FFTW_MEASURE);
    forward42 = fftwf_plan_many_dft(3, n, 4, array42, n, 1, N, array42, n, 1, N, FFTW_FORWARD, FFTW_MEASURE);
    backward1 = fftwf_plan_many_dft(3, n, 1, array1, n, 1, N, array1, n, 1, N, FFTW_BACKWARD, FFTW_MEASURE);
    backward2 = fftwf_plan_many_dft(3, n, 2, array3, n, 1, N, array3, n, 1, N, FFTW_BACKWARD, FFTW_MEASURE);
    backward3 = fftwf_plan_many_dft(3, n, 3, array3, n, 1, N, array3, n, 1, N, FFTW_BACKWARD, FFTW_MEASURE);
}

/// @brief Perform initializations for computing matrix-vector products in a simulation with given media parameters and wavelengths
/// @brief Compute the point source approximations, the sparse correction matrices, and Fourier transform of Green's functions
/// @param eta1: index of refraction of the medium where the light is incident from, usually 1.0 (air)
/// @param eta2: index of refraction of the surface material (could be complex-valued)
/// @param lambda: the currently simulated wavelength
void MVProd0::setParameters(double eta1, dcomp eta2, double lambda) {
    grid->computeCoefficients(eta1, eta2, lambda);
    if (eta2.imag() == 0)
        isDielectric = true;
    else
        isDielectric = false;
    k1 = (fcomp)(2.0 * M_PI / lambda * eta1);
    k2 = (fcomp)(2.0 * M_PI / lambda * eta2);
    const1 = (fcomp)(cuDB * mu * c * 2.0 * M_PI / lambda);
    const21 = (fcomp)(cuDB * mu * c * lambda / (2.0 * M_PI * eta1 * eta1));
    const22 = (fcomp)(cuDB * mu * c * lambda / (2.0 * M_PI * eta2 * eta2));
    c1 = 2.0f * const1;
    c2 = const21 + const22;
    c3 = -(fcomp)(eta1 * eta1 + eta2 * eta2) * const1;
    c4 = -(fcomp)(eta1 * eta1) * const21 - (fcomp)(eta2 * eta2) * const22;
    this->eta1 = (fcomp)eta1;
    this->eta2 = (fcomp)eta2;
    computeGreens();
    sparseHH();
    sparseHV();
    sparseVV();
    fftwf_execute(forward41);
    if (isDielectric)
        fftwf_execute(forward42);
}

/// @brief Compute the shift-invarient g (Green's) function values for pairs of points in the 3D grid
void MVProd0::computeGreens() {
    VectorXf diffX = VectorXf::LinSpaced(totalX - 1, -(numX - 1) * dx, (numX - 1) * dx);
    VectorXf diffY = VectorXf::LinSpaced(totalY - 1, -(numY - 1) * dy, (numY - 1) * dy);
    VectorXf diffZ = VectorXf::LinSpaced(totalZ - 1, -(numZ - 1) * dz, (numZ - 1) * dz);
    data41 = MatrixXcf::Zero(N, 4);
    data42 = MatrixXcf::Zero(N, 4);

    // Compute and store the shift-invariant functions for all different values of r - r'
    for (int countX = 1; countX < totalX; countX++) {
        for (int countY = 1; countY < totalY; countY++) {
            for (int countZ = 1; countZ < totalZ; countZ++) {
                int linear = totalY * totalZ * countX + totalZ * countY + countZ;
                float distance = sqrt(diffX(countX - 1) * diffX(countX - 1) + diffY(countY - 1) * diffY(countY - 1) + diffZ(countZ - 1) * diffZ(countZ - 1));
                fcomp green11 = 0, green12 = 0, green21 = 0, green22 = 0;
                if (distance > 0) {
                    green11 = exp(-cuFL * k1 * distance) / (4.0f * M_PIFL * distance);
                    green12 = green11 * (1.0f + cuFL * k1 * distance) / (distance * distance);
                    if (isDielectric) {
                        green21 = exp(-cuFL * k2 * distance) / (4.0f * M_PIFL * distance);
                        green22 = green21 * (1.0f + cuFL * k2 * distance) / (distance * distance);
                    }
                }
                data41(linear, 0) = green11;
                data41(linear, 1) = green12 * diffX(countX - 1);
                data41(linear, 2) = green12 * diffY(countY - 1);
                data41(linear, 3) = green12 * diffZ(countZ - 1);

                // Lossy materials does not need the far component of the BEM matrix
                if (isDielectric) {
                    data42(linear, 0) = green21;
                    data42(linear, 1) = green22 * diffX(countX - 1);
                    data42(linear, 2) = green22 * diffY(countY - 1);
                    data42(linear, 3) = green22 * diffZ(countZ - 1);
                }
            }
        }
    }
}

/// @brief Construct the sparse correction matrix for matrix blocks that involve pairs of horizontally arranged basis functions
void MVProd0::sparseHH() {
    ee_fill.reserve(2 * (est->A[0].rows() + est->B[0].rows()));
    mm_fill.reserve(2 * (est->A[0].rows() + est->B[0].rows()));
    em_fill.reserve(2 * (est->A[0].rows() + est->B[0].rows()));

    // Compute matrix elements
    singular->computeHH();
    computeHH();

    // Fill the triplets that represent the sparse matrices
    for (int i = 0; i < est->A[0].rows(); i++) {
        int m = est->A[0](i, 0);
        int n = est->A[0](i, 1);
        ee_fill.push_back(T(m, n, c1 * singular->quarter(i, 0) + c2 * singular->quarter(i, 1)));
        mm_fill.push_back(T(m, n, c3 * singular->quarter(i, 0) + c4 * singular->quarter(i, 1)));
        em_fill.push_back(T(m, n, singular->quarter(i, 2)));
        ee_fill.push_back(T(n, m, c1 * singular->quarter(i, 0) + c2 * singular->quarter(i, 1)));
        mm_fill.push_back(T(n, m, c3 * singular->quarter(i, 0) + c4 * singular->quarter(i, 1)));
        em_fill.push_back(T(n, m, singular->quarter(i, 2)));
    }
    for (int i = 0; i < est->B[0].rows(); i++) {
        int m = est->B[0](i, 0);
        int n = est->B[0](i, 1);
        ee_fill.push_back(T(m, n, qData(i, 0)));
        mm_fill.push_back(T(m, n, qData(i, 1)));
        em_fill.push_back(T(m, n, qData(i, 2)));
        ee_fill.push_back(T(n, m, qData(i, 0)));
        mm_fill.push_back(T(n, m, qData(i, 1)));
        em_fill.push_back(T(n, m, qData(i, 2)));
    }

    // Generate sparse matrices
    Zeehh.resize(hori_num, hori_num);
    Zmmhh.resize(hori_num, hori_num);
    Zemhh.resize(hori_num, hori_num);
    Zeehh.setFromTriplets(ee_fill.begin(), ee_fill.end());
    Zmmhh.setFromTriplets(mm_fill.begin(), mm_fill.end());
    Zemhh.setFromTriplets(em_fill.begin(), em_fill.end());
    vector<T>().swap(ee_fill);
    vector<T>().swap(mm_fill);
    vector<T>().swap(em_fill);
}

/// @brief Construct the sparse correction matrix for matrix blocks that involve pairs of one horizontally arranged basis function and one vertically arranged basis function
void MVProd0::sparseHV() {
    ee_fill.reserve(est->A[1].rows() + est->B[1].rows() + est->A[2].rows() + est->B[2].rows());
    mm_fill.reserve(est->A[1].rows() + est->B[1].rows() + est->A[2].rows() + est->B[2].rows());
    em_fill.reserve(est->A[1].rows() + est->B[1].rows() + est->A[2].rows() + est->B[2].rows());

    // Compute matrix elements
    singular->computeHV(1);
    computeHV(1);

    // Fill the triplets that represent the sparse matrices
    for (int i = 0; i < est->A[1].rows(); i++) {
        int m = est->A[1](i, 0);
        int n = est->A[1](i, 1);
        ee_fill.push_back(T(m, n, c1 * singular->quarter(i, 0) + c2 * singular->quarter(i, 1)));
        mm_fill.push_back(T(m, n, c3 * singular->quarter(i, 0) + c4 * singular->quarter(i, 1)));
        em_fill.push_back(T(m, n, singular->quarter(i, 2)));
    }
    for (int i = 0; i < est->B[1].rows(); i++) {
        int m = est->B[1](i, 0);
        int n = est->B[1](i, 1);
        ee_fill.push_back(T(m, n, qData(i, 0)));
        mm_fill.push_back(T(m, n, qData(i, 1)));
        em_fill.push_back(T(m, n, qData(i, 2)));
    }

    // Compute more matrix elements
    singular->computeHV(2);
    computeHV(2);

    // Add to the triplets that represent the sparse matrices
    for (int i = 0; i < est->A[2].rows(); i++) {
        int m = est->A[2](i, 0);
        int n = est->A[2](i, 1);
        ee_fill.push_back(T(m, n, c1 * singular->quarter(i, 0) + c2 * singular->quarter(i, 1)));
        mm_fill.push_back(T(m, n, c3 * singular->quarter(i, 0) + c4 * singular->quarter(i, 1)));
        em_fill.push_back(T(m, n, singular->quarter(i, 2)));
    }
    for (int i = 0; i < est->B[2].rows(); i++) {
        int m = est->B[2](i, 0);
        int n = est->B[2](i, 1);
        ee_fill.push_back(T(m, n, qData(i, 0)));
        mm_fill.push_back(T(m, n, qData(i, 1)));
        em_fill.push_back(T(m, n, qData(i, 2)));
    }

    // Generate sparse matrices
    Zeehv.resize(hori_num, vert_num);
    Zmmhv.resize(hori_num, vert_num);
    Zemhv.resize(hori_num, vert_num);
    Zeehv.setFromTriplets(ee_fill.begin(), ee_fill.end());
    Zmmhv.setFromTriplets(mm_fill.begin(), mm_fill.end());
    Zemhv.setFromTriplets(em_fill.begin(), em_fill.end());
    vector<T>().swap(ee_fill);
    vector<T>().swap(mm_fill);
    vector<T>().swap(em_fill);
}

/// @brief Construct the sparse correction matrix for matrix blocks that involve pairs of vertically arranged basis functions
void MVProd0::sparseVV() {
    ee_fill.reserve(2 * (est->A[3].rows() + est->B[3].rows()));
    mm_fill.reserve(2 * (est->A[3].rows() + est->B[3].rows()));
    em_fill.reserve(2 * (est->A[3].rows() + est->B[3].rows()));

    // Compute matrix elements
    singular->computeVV();
    computeVV();

    // Fill the triplets that represent the sparse matrices
    for (int i = 0; i < est->A[3].rows(); i++) {
        int m = est->A[3](i, 0);
        int n = est->A[3](i, 1);
        ee_fill.push_back(T(m, n, c1 * singular->quarter(i, 0) + c2 * singular->quarter(i, 1)));
        mm_fill.push_back(T(m, n, c3 * singular->quarter(i, 0) + c4 * singular->quarter(i, 1)));
        em_fill.push_back(T(m, n, singular->quarter(i, 2)));
        ee_fill.push_back(T(n, m, c1 * singular->quarter(i, 0) + c2 * singular->quarter(i, 1)));
        mm_fill.push_back(T(n, m, c3 * singular->quarter(i, 0) + c4 * singular->quarter(i, 1)));
        em_fill.push_back(T(n, m, singular->quarter(i, 2)));
    }
    for (int i = 0; i < est->B[3].rows(); i++) {
        int m = est->B[3](i, 0);
        int n = est->B[3](i, 1);
        ee_fill.push_back(T(m, n, qData(i, 0)));
        mm_fill.push_back(T(m, n, qData(i, 1)));
        em_fill.push_back(T(m, n, qData(i, 2)));
        ee_fill.push_back(T(n, m, qData(i, 0)));
        mm_fill.push_back(T(n, m, qData(i, 1)));
        em_fill.push_back(T(n, m, qData(i, 2)));
    }

    // Generate sparse matrices
    Zeevv.resize(vert_num, vert_num);
    Zmmvv.resize(vert_num, vert_num);
    Zemvv.resize(vert_num, vert_num);
    Zeevv.setFromTriplets(ee_fill.begin(), ee_fill.end());
    Zmmvv.setFromTriplets(mm_fill.begin(), mm_fill.end());
    Zemvv.setFromTriplets(em_fill.begin(), em_fill.end());
    vector<T>().swap(ee_fill);
    vector<T>().swap(mm_fill);
    vector<T>().swap(em_fill);
}

/// @brief Perform matrix-vector multiplication using the BEM matrix
/// @param x: the input vector
/// @return The product vector
VectorXcf MVProd0::multiply(VectorXcf x) {
    VectorXcf y0 = near(x);
    VectorXcf y1 = far(x, eta1, const21, data41, grid->hori_x, grid->hori_z, grid->hori_d, grid->vert_y, grid->vert_z, grid->vert_d);
    VectorXcf y = y0 + y1;
    if (!isDielectric)
        return y;
    VectorXcf y2 = far(x, eta2, const22, data42, grid->hori_X, grid->hori_Z, grid->hori_D, grid->vert_Y, grid->vert_Z, grid->vert_D);
    return y + y2;
}

/// @brief Multiply the sparse correction matrix to the given vector
/// @param x: the input vector
/// @return The product vector
VectorXcf MVProd0::near(VectorXcf x) {
    VectorXcf x1 = x.block(0, 0, hori_num, 1);
    VectorXcf x2 = x.block(hori_num, 0, vert_num, 1);
    VectorXcf x3 = x.block(hori_num + vert_num, 0, hori_num, 1);
    VectorXcf x4 = x.block(2 * hori_num + vert_num, 0, vert_num, 1);
    VectorXcf y1 = Zeehh * x1 + Zeehv * x2 + Zemhh * x3 + Zemhv * x4;
    VectorXcf y2 = Zeehv.transpose() * x1 + Zeevv * x2 + Zemhv.transpose() * x3 + Zemvv * x4;
    VectorXcf y3 = Zemhh * x1 + Zemhv * x2 + Zmmhh * x3 + Zmmhv * x4;
    VectorXcf y4 = Zemhv.transpose() * x1 + Zemvv * x2 + Zmmhv.transpose() * x3 + Zmmvv * x4;
    VectorXcf y(2 * hori_num + 2 * vert_num);
    y.block(0, 0, hori_num, 1) = y1;
    y.block(hori_num, 0, vert_num, 1) = y2;
    y.block(hori_num + vert_num, 0, hori_num, 1) = y3;
    y.block(2 * hori_num + vert_num, 0, vert_num, 1) = y4;
    return y;
}

/// @brief Multiply the base approximation matrix to the given vector
/// @param x: the input vector
/// @return The product vector
VectorXcf MVProd0::far(VectorXcf x, fcomp eta, fcomp const2, MatrixXcf& data4, MatrixXcf& hori_x, MatrixXcf& hori_z, MatrixXcf& hori_d, MatrixXcf& vert_y, MatrixXcf& vert_z, MatrixXcf& vert_d) {
    VectorXcf x1 = x.block(0, 0, hori_num + vert_num, 1);
    VectorXcf x2 = x.block(hori_num + vert_num, 0, hori_num + vert_num, 1);
    VectorXcf y1 = VectorXcf::Zero(hori_num + vert_num);
    VectorXcf y2 = VectorXcf::Zero(hori_num + vert_num);
    
    // Round 1: 3 forward FFTs
    data3 = MatrixXcf::Zero(N, 3);
    for (int count = 0; count < hori_col; count++) {
        int linear = grid->hori_f(0, count);
        for (int pt = 1; pt < hori_row; pt++) {
            int index = grid->hori_f(pt, count);
            if (index == -1)
                break;
            int r0 = index % num_pts;
            int c0 = index / num_pts;
            data3(linear, 0) += hori_x(r0, c0) * x1(c0);
            data3(linear, 2) += hori_z(r0, c0) * x1(c0);
        }
    }
    for (int count = 0; count < vert_col; count++) {
        int linear = grid->vert_f(0, count);
        for (int pt = 1; pt < vert_row; pt++) {
            int index = grid->vert_f(pt, count);
            if (index == -1)
                break;
            int r0 = index % num_pts;
            int c0 = index / num_pts;
            data3(linear, 1) += vert_y(r0, c0) * x1(hori_num + c0);
            data3(linear, 2) += vert_z(r0, c0) * x1(hori_num + c0);
        }
    }
    fftwf_execute(forward3);
    
    // Round 1: 3 backward FFTs, one at at time, for Ame
    data1 = data3.block(0, 0, N, 1).cwiseProduct(data4.block(0, 2, N, 1)) - data3.block(0, 1, N, 1).cwiseProduct(data4.block(0, 1, N, 1));
    fftwf_execute(backward1);
    for (int count = 0; count < hori_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->hori_b(pt, count);
            y2(count) += eta0FL * hori_z(pt, count) * data1(linear);
        }
    }
    for (int count = 0; count < vert_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->vert_b(pt, count);
            y2(hori_num + count) += eta0FL * vert_z(pt, count) * data1(linear);
        }
    }
    data1 = data3.block(0, 1, N, 1).cwiseProduct(data4.block(0, 3, N, 1)) - data3.block(0, 2, N, 1).cwiseProduct(data4.block(0, 2, N, 1));
    fftwf_execute(backward1);
    for (int count = 0; count < hori_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->hori_b(pt, count);
            y2(count) += eta0FL * hori_x(pt, count) * data1(linear);
        }
    }
    data1 = data3.block(0, 2, N, 1).cwiseProduct(data4.block(0, 1, N, 1)) - data3.block(0, 0, N, 1).cwiseProduct(data4.block(0, 3, N, 1));
    fftwf_execute(backward1);
    for (int count = 0; count < vert_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->vert_b(pt, count);
            y2(hori_num + count) += eta0FL * vert_y(pt, count) * data1(linear);
        }
    }
    
    // Round 1: 3 backward FFTs, in a row, for Aee
    data3.block(0, 0, N, 1) = data3.block(0, 0, N, 1).cwiseProduct(const1 * data4.block(0, 0, N, 1));
    data3.block(0, 1, N, 1) = data3.block(0, 1, N, 1).cwiseProduct(const1 * data4.block(0, 0, N, 1));
    data3.block(0, 2, N, 1) = data3.block(0, 2, N, 1).cwiseProduct(const1 * data4.block(0, 0, N, 1));
    fftwf_execute(backward3);
    for (int count = 0; count < hori_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->hori_b(pt, count);
            y1(count) += hori_x(pt, count) * data3(linear, 0) + hori_z(pt, count) * data3(linear, 2);
        }
    }
    for (int count = 0; count < vert_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->vert_b(pt, count);
            y1(hori_num + count) += vert_y(pt, count) * data3(linear, 1) + vert_z(pt, count) * data3(linear, 2);
        }
    }
    
    // Round 2: 3 forward FFTs
    data3 = MatrixXcf::Zero(N, 3);
    for (int count = 0; count < hori_col; count++) {
        int linear = grid->hori_f(0, count);
        for (int pt = 1; pt < hori_row; pt++) {
            int index = grid->hori_f(pt, count);
            if (index == -1)
                break;
            int r0 = index % num_pts;
            int c0 = index / num_pts;
            data3(linear, 0) += hori_x(r0, c0) * x2(c0);
            data3(linear, 2) += hori_z(r0, c0) * x2(c0);
        }
    }
    for (int count = 0; count < vert_col; count++) {
        int linear = grid->vert_f(0, count);
        for (int pt = 1; pt < vert_row; pt++) {
            int index = grid->vert_f(pt, count);
            if (index == -1)
                break;
            int r0 = index % num_pts;
            int c0 = index / num_pts;
            data3(linear, 1) += vert_y(r0, c0) * x2(hori_num + c0);
            data3(linear, 2) += vert_z(r0, c0) * x2(hori_num + c0);
        }
    }
    fftwf_execute(forward3);
    
    // Round 2: 3 backward FFTs, one at a time, for Aem
    data1 = data3.block(0, 0, N, 1).cwiseProduct(data4.block(0, 2, N, 1)) - data3.block(0, 1, N, 1).cwiseProduct(data4.block(0, 1, N, 1));
    fftwf_execute(backward1);
    for (int count = 0; count < hori_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->hori_b(pt, count);
            y1(count) += eta0FL * hori_z(pt, count) * data1(linear);
        }
    }
    for (int count = 0; count < vert_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->vert_b(pt, count);
            y1(hori_num + count) += eta0FL * vert_z(pt, count) * data1(linear);
        }
    }
    data1 = data3.block(0, 1, N, 1).cwiseProduct(data4.block(0, 3, N, 1)) - data3.block(0, 2, N, 1).cwiseProduct(data4.block(0, 2, N, 1));
    fftwf_execute(backward1);
    for (int count = 0; count < hori_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->hori_b(pt, count);
            y1(count) += eta0FL * hori_x(pt, count) * data1(linear);
        }
    }
    data1 = data3.block(0, 2, N, 1).cwiseProduct(data4.block(0, 1, N, 1)) - data3.block(0, 0, N, 1).cwiseProduct(data4.block(0, 3, N, 1));
    fftwf_execute(backward1);
    for (int count = 0; count < vert_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->vert_b(pt, count);
            y1(hori_num + count) += eta0FL * vert_y(pt, count) * data1(linear);
        }
    }
    
    // Round 2: 3 backward FFTs, in a row, for Amm
    data3.block(0, 0, N, 1) = data3.block(0, 0, N, 1).cwiseProduct(const1 * eta * eta * data4.block(0, 0, N, 1));
    data3.block(0, 1, N, 1) = data3.block(0, 1, N, 1).cwiseProduct(const1 * eta * eta * data4.block(0, 0, N, 1));
    data3.block(0, 2, N, 1) = data3.block(0, 2, N, 1).cwiseProduct(const1 * eta * eta * data4.block(0, 0, N, 1));
    fftwf_execute(backward3);
    for (int count = 0; count < hori_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->hori_b(pt, count);
            y2(count) -= hori_x(pt, count) * data3(linear, 0) + hori_z(pt, count) * data3(linear, 2);
        }
    }
    for (int count = 0; count < vert_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->vert_b(pt, count);
            y2(hori_num + count) -= vert_y(pt, count) * data3(linear, 1) + vert_z(pt, count) * data3(linear, 2);
        }
    }
    
    // Round 3: 2 forward FFTs, in a row
    data3 = MatrixXcf::Zero(N, 3);
    for (int count = 0; count < hori_col; count++) {
        int linear = grid->hori_f(0, count);
        for (int pt = 1; pt < hori_row; pt++) {
            int index = grid->hori_f(pt, count);
            if (index == -1)
                break;
            int r0 = index % num_pts;
            int c0 = index / num_pts;
            data3(linear, 0) += hori_d(r0, c0) * x1(c0);
            data3(linear, 1) += hori_d(r0, c0) * x2(c0);
        }
    }
    for (int count = 0; count < vert_col; count++) {
        int linear = grid->vert_f(0, count);
        for (int pt = 1; pt < vert_row; pt++) {
            int index = grid->vert_f(pt, count);
            if (index == -1)
                break;
            int r0 = index % num_pts;
            int c0 = index / num_pts;
            data3(linear, 0) += vert_d(r0, c0) * x1(hori_num + c0);
            data3(linear, 1) += vert_d(r0, c0) * x2(hori_num + c0);
        }
    }
    fftwf_execute(forward2);
    
    // Round 3: 2 backward FFTs, in a row, for Aee and Amm
    data3.block(0, 0, N, 1) = data3.block(0, 0, N, 1).cwiseProduct(const2 * data4.block(0, 0, N, 1));
    data3.block(0, 1, N, 1) = data3.block(0, 1, N, 1).cwiseProduct(const2 * eta * eta * data4.block(0, 0, N, 1));
    fftwf_execute(backward2);
    for (int count = 0; count < hori_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->hori_b(pt, count);
            y1(count) -= hori_d(pt, count) * data3(linear, 0);
            y2(count) += hori_d(pt, count) * data3(linear, 1);
        }
    }
    for (int count = 0; count < vert_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int linear = grid->vert_b(pt, count);
            y1(hori_num + count) -= vert_d(pt, count) * data3(linear, 0);
            y2(hori_num + count) += vert_d(pt, count) * data3(linear, 1);
        }
    }
    // Final result
    VectorXcf y(2 * hori_num + 2 * vert_num);
    y.block(0, 0, hori_num + vert_num, 1) = y1 / N;
    y.block(hori_num + vert_num, 0, hori_num + vert_num, 1) = y2 / N;
    return y;
}

/// @brief Compute sparse correction matrix elements for matrix blocks that involve pairs of horizontally arranged basis functions
void MVProd0::computeHH() {
    qData = MatrixXcf::Zero(est->B[0].rows(), 3);
    parallel_for(est->B[0].rows(), [&](int start, int end) {
    for (int i = start; i < end; i++) {
        int m = est->B[0](i, 0);
        int n = est->B[0](i, 1);
        int my = m / (Nx - 1);
        int mx = m - (Nx - 1) * my;
        int ny = n / (Nx - 1);
        int nx = n - (Nx - 1) * ny;

        // Brute-force computation of the accurate matrix element value
        fcomp store1 = 0, store2 = 0, store3 = 0;
        individual(mx, my, nx, ny, 1, 1, 1, store1, store2, store3);
        individual(mx, my, nx + 1, ny, 1, 1, 0, store1, store2, store3);
        individual(mx + 1, my, nx, ny, 1, 0, 1, store1, store2, store3);
        individual(mx + 1, my, nx + 1, ny, 1, 0, 0, store1, store2, store3);

        // Subtract the inaccurate approximated matrix element value to obtain the correction value
        for (int pt1 = 0; pt1 < num_pts; pt1++) {
            int x1 = grid->hori_b(pt1, m) / (totalY * totalZ);
            int z1 = grid->hori_b(pt1, m) % totalZ;
            int y1 = (grid->hori_b(pt1, m) - z1 - totalY * totalZ * x1) / totalZ;
            for (int pt2 = 0; pt2 < num_pts; pt2++) {
                int x2 = grid->hori_b(pt2, n) / (totalY * totalZ);
                int z2 = grid->hori_b(pt2, n) % totalZ;
                int y2 = (grid->hori_b(pt2, n) - z2 - totalY * totalZ * x2) / totalZ;
                int linear = (x1 - x2 + numX) * totalY * totalZ + (y1 - y2 + numY) * totalZ + (z1 - z2 + numZ);
                store1 -= data41(linear, 0) * (const1 * grid->hori_x(pt1, m) * grid->hori_x(pt2, n) + const1 * grid->hori_z(pt1, m) * grid->hori_z(pt2, n) - const21 * grid->hori_d(pt1, m) * grid->hori_d(pt2, n));
                store3 -= data41(linear, 2) * (grid->hori_z(pt1, m) * grid->hori_x(pt2, n) - grid->hori_x(pt1, m) * grid->hori_z(pt2, n));
                if (isDielectric) {
                    store2 -= data42(linear, 0) * (const1 * grid->hori_X(pt1, m) * grid->hori_X(pt2, n) + const1 * grid->hori_Z(pt1, m) * grid->hori_Z(pt2, n) - const22 * grid->hori_D(pt1, m) * grid->hori_D(pt2, n));
                    store3 -= data42(linear, 2) * (grid->hori_Z(pt1, m) * grid->hori_X(pt2, n) - grid->hori_X(pt1, m) * grid->hori_Z(pt2, n));
                }
            }
        }

        // Store in data structure to use for building sparse matrices
        qData(i, 0) = store1 + store2;
        qData(i, 1) = -eta1 * eta1 * store1 - eta2 * eta2 * store2;
        qData(i, 2) = eta0FL * store3;
        if (m == n) {
            qData(i, 0) = 0.5f * qData(i, 0);
            qData(i, 1) = 0.5f * qData(i, 1);
            qData(i, 2) = 0.5f * qData(i, 2);
        }
    }
    } );
}

/// @brief Compute sparse correction matrix elements for matrix blocks that involve pairs of one horizontally arranged basis function and one vertically arranged basis function
/// @param dev: an integer value of 1 or 2 (the computations are done in two groups)
void MVProd0::computeHV(int dev) {
    qData = MatrixXcf::Zero(est->B[dev].rows(), 3);
    parallel_for(est->B[dev].rows(), [&](int start, int end) {
    for (int i = start; i < end; i++) {
        int m = est->B[dev](i, 0);
        int n = est->B[dev](i, 1);
        int my = m / (Nx - 1);
        int mx = m - (Nx - 1) * my;
        int nx = n / (Ny - 1);
        int ny = n - (Ny - 1) * nx;

        // Brute-force computation of the accurate matrix element value
        fcomp store1 = 0, store2 = 0, store3 = 0;
        individual(mx, my, nx, ny, 2, 1, 1, store1, store2, store3);
        individual(mx, my, nx, ny + 1, 2, 1, 0, store1, store2, store3);
        individual(mx + 1, my, nx, ny, 2, 0, 1, store1, store2, store3);
        individual(mx + 1, my, nx, ny + 1, 2, 0, 0, store1, store2, store3);

        // Subtract the inaccurate approximated matrix element value to obtain the correction value
        for (int pt1 = 0; pt1 < num_pts; pt1++) {
            int x1 = grid->hori_b(pt1, m) / (totalY * totalZ);
            int z1 = grid->hori_b(pt1, m) % totalZ;
            int y1 = (grid->hori_b(pt1, m) - z1 - totalY * totalZ * x1) / totalZ;
            for (int pt2 = 0; pt2 < num_pts; pt2++) {
                int x2 = grid->vert_b(pt2, n) / (totalY * totalZ);
                int z2 = grid->vert_b(pt2, n) % totalZ;
                int y2 = (grid->vert_b(pt2, n) - z2 - totalY * totalZ * x2) / totalZ;
                int linear = (x1 - x2 + numX) * totalY * totalZ + (y1 - y2 + numY) * totalZ + (z1 - z2 + numZ);
                store1 -= data41(linear, 0) * (const1 * grid->hori_z(pt1, m) * grid->vert_z(pt2, n) - const21 * grid->hori_d(pt1, m) * grid->vert_d(pt2, n));
                store3 -= data41(linear, 3) * grid->hori_x(pt1, m) * grid->vert_y(pt2, n) - data41(linear, 1) * grid->hori_z(pt1, m) * grid->vert_y(pt2, n) - data41(linear, 2) * grid->hori_x(pt1, m) * grid->vert_z(pt2, n);
                if (isDielectric) {
                    store2 -= data42(linear, 0) * (const1 * grid->hori_Z(pt1, m) * grid->vert_Z(pt2, n) - const22 * grid->hori_D(pt1, m) * grid->vert_D(pt2, n));
                    store3 -= data42(linear, 3) * grid->hori_X(pt1, m) * grid->vert_Y(pt2, n) - data42(linear, 1) * grid->hori_Z(pt1, m) * grid->vert_Y(pt2, n) - data42(linear, 2) * grid->hori_X(pt1, m) * grid->vert_Z(pt2, n);
                }
            }
        }

        // Store in data structure to use for building sparse matrices
        qData(i, 0) = store1 + store2;
        qData(i, 1) = -eta1 * eta1 * store1 - eta2 * eta2 * store2;
        qData(i, 2) = eta0FL * store3;
    }
    } );
}

/// @brief Compute sparse correction matrix elements for matrix blocks that involve pairs of vertically arranged basis functions
void MVProd0::computeVV() {
    qData = MatrixXcf::Zero(est->B[3].rows(), 3);
    parallel_for(est->B[3].rows(), [&](int start, int end) {
    for (int i = start; i < end; i++) {
        int m = est->B[3](i, 0);
        int n = est->B[3](i, 1);
        int mx = m / (Ny - 1);
        int my = m - (Ny - 1) * mx;
        int nx = n / (Ny - 1);
        int ny = n - (Ny - 1) * nx;

        // Brute-force computation of the accurate matrix element value
        fcomp store1 = 0, store2 = 0, store3 = 0;
        individual(mx, my, nx, ny, 3, 1, 1, store1, store2, store3);
        individual(mx, my, nx, ny + 1, 3, 1, 0, store1, store2, store3);
        individual(mx, my + 1, nx, ny, 3, 0, 1, store1, store2, store3);
        individual(mx, my + 1, nx, ny + 1, 3, 0, 0, store1, store2, store3);

        // Subtract the inaccurate approximated matrix element value to obtain the correction value
        for (int pt1 = 0; pt1 < num_pts; pt1++) {
            int x1 = grid->vert_b(pt1, m) / (totalY * totalZ);
            int z1 = grid->vert_b(pt1, m) % totalZ;
            int y1 = (grid->vert_b(pt1, m) - z1 - totalY * totalZ * x1) / totalZ;
            for (int pt2 = 0; pt2 < num_pts; pt2++) {
                int x2 = grid->vert_b(pt2, n) / (totalY * totalZ);
                int z2 = grid->vert_b(pt2, n) % totalZ;
                int y2 = (grid->vert_b(pt2, n) - z2 - totalY * totalZ * x2) / totalZ;
                int linear = (x1 - x2 + numX) * totalY * totalZ + (y1 - y2 + numY) * totalZ + (z1 - z2 + numZ);
                store1 -= data41(linear, 0) * (const1 * grid->vert_y(pt1, m) * grid->vert_y(pt2, n) + const1 * grid->vert_z(pt1, m) * grid->vert_z(pt2, n) - const21 * grid->vert_d(pt1, m) * grid->vert_d(pt2, n));
                store3 -= data41(linear, 1) * (grid->vert_y(pt1, m) * grid->vert_z(pt2, n) - grid->vert_z(pt1, m) * grid->vert_y(pt2, n));
                if (isDielectric) {
                    store2 -= data42(linear, 0) * (const1 * grid->vert_Y(pt1, m) * grid->vert_Y(pt2, n) + const1 * grid->vert_Z(pt1, m) * grid->vert_Z(pt2, n) - const22 * grid->vert_D(pt1, m) * grid->vert_D(pt2, n));
                    store3 -= data42(linear, 1) * (grid->vert_Y(pt1, m) * grid->vert_Z(pt2, n) - grid->vert_Z(pt1, m) * grid->vert_Y(pt2, n));
                }
            }
        }

        // Store in data structure to use for building sparse matrices
        qData(i, 0) = store1 + store2;
        qData(i, 1) = -eta1 * eta1 * store1 - eta2 * eta2 * store2;
        qData(i, 2) = eta0FL * store3;
        if (m == n) {
            qData(i, 0) = 0.5f * qData(i, 0);
            qData(i, 1) = 0.5f * qData(i, 1);
            qData(i, 2) = 0.5f * qData(i, 2);
        }
    }
    } );
}

/// @brief Compute a BEM matrix element using brute-force numerical integration (for nearby basis element pairs)
/// @param mx, my, nx, ny: x, y indices of the considered pair of basis elements
/// @param type: specifies directions of the integrated basis functions (horizontal or vertical)
/// @param side1, side2: specify which segment of the basis functions to integrate over
/// @param store1, store2, store3: computed matrix elements in different blocks
void MVProd0::individual(int mx, int my, int nx, int ny, int type, int side1, int side2, fcomp& store1, fcomp& store2, fcomp& store3) {
    float x00_m = (xvals(mx) + xvals(mx + 1)) / 2;
    float y00_m = (yvals(my) + yvals(my + 1)) / 2;
    float z11_m = (zvals(mx, my) - zvals(mx + 1, my) - zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    float z10_m = (-zvals(mx, my) + zvals(mx + 1, my) - zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    float z01_m = (-zvals(mx, my) - zvals(mx + 1, my) + zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    float z00_m = (zvals(mx, my) + zvals(mx + 1, my) + zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    float x00_n = (xvals(nx) + xvals(nx + 1)) / 2;
    float y00_n = (yvals(ny) + yvals(ny + 1)) / 2;
    float z11_n = (zvals(nx, ny) - zvals(nx + 1, ny) - zvals(nx, ny + 1) + zvals(nx + 1, ny + 1)) / 4;
    float z10_n = (-zvals(nx, ny) + zvals(nx + 1, ny) - zvals(nx, ny + 1) + zvals(nx + 1, ny + 1)) / 4;
    float z01_n = (-zvals(nx, ny) - zvals(nx + 1, ny) + zvals(nx, ny + 1) + zvals(nx + 1, ny + 1)) / 4;
    float z00_n = (zvals(nx, ny) + zvals(nx + 1, ny) + zvals(nx, ny + 1) + zvals(nx + 1, ny + 1)) / 4;
    fcomp ee1 = 0, ee2 = 0, em0 = 0;
    VectorXf pvals, wvals;
    int order;
    if (abs(mx - nx) + abs(my - ny) <= 3) {
        pvals = pH;
        wvals = wH;
        order = orderH;
    } else {
        pvals = pL;
        wvals = wL;
        order = orderL;
    }
    for (int um = 0; um < order; um++) {
        float u1 = pvals(um);
        for (int vm = 0; vm < order; vm++) {
            float v1 = pvals(vm);
            for (int un = 0; un < order; un++) {
                float u2 = pvals(un);
                for (int vn = 0; vn < order; vn++) {
                    float v2 = pvals(vn);
                    float deltax = 0.5f * d * (u1 - u2) + x00_m - x00_n;
                    float deltay = 0.5f * d * (v1 - v2) + y00_m - y00_n;
                    float deltaz = z11_m * u1 * v1 + z10_m * u1 + z01_m * v1 + z00_m - z11_n * u2 * v2 - z10_n * u2 - z01_n * v2 - z00_n;
                    float r = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
                    fcomp green11, green12, green2;
                    if (abs(mx - nx) + abs(my - ny) <= 1) {
                        if (r < 1e-4) {
                            green11 = -cuFL * k1 / (4.0f * M_PIFL);
                            green12 = -cuFL * k2 / (4.0f * M_PIFL);
                        } else {
                            green11 = (exp(-cuFL * k1 * r) - 1.0f) / (4.0f * M_PIFL * r);
                            green12 = (exp(-cuFL * k2 * r) - 1.0f) / (4.0f * M_PIFL * r);
                            green2 = (exp(-cuFL * k1 * r) * (cuFL * k1 * r + 1.0f) - 1.0f) / (4.0f * M_PIFL * r * r * r) + (exp(-cuFL * k2 * r) * (cuFL * k2 * r + 1.0f) - 1.0f) / (4.0f * M_PIFL * r * r * r);
                        }
                    } else {
                        green11 = exp(-cuFL * k1 * r) / (4.0f * M_PIFL * r);
                        green12 = exp(-cuFL * k2 * r) / (4.0f * M_PIFL * r);
                        green2 = exp(-cuFL * k1 * r) * (cuFL * k1 * r + 1.0f) / (4.0f * M_PIFL * r * r * r) + exp(-cuFL * k2 * r) * (cuFL * k2 * r + 1.0f) / (4.0f * M_PIFL * r * r * r);
                    }
                    float weight = wvals(um) * wvals(vm) * wvals(un) * wvals(vn);
                    float common, dot;
                    if (type == 1) {
                        common = (1.0f - pow(-1, side1) * u1) * (1.0f - pow(-1, side2) * u2);
                        dot = 0.25f * d * d + (z11_m * v1 + z10_m) * (z11_n * v2 + z10_n);
                    } else if (type == 2) {
                        common = (1.0f - pow(-1, side1) * u1) * (1.0f - pow(-1, side2) * v2);
                        dot = (z11_m * v1 + z10_m) * (z11_n * u2 + z01_n);
                    } else {
                        common = (1.0f - pow(-1, side1) * v1) * (1.0f - pow(-1, side2) * v2);
                        dot = 0.25f * d * d + (z11_m * u1 + z01_m) * (z11_n * u2 + z01_n);
                    }
                    ee1 += weight * green11 * (const1 * common * dot - (float)pow(-1, side1 + side2) * const21);
                    ee2 += weight * green12 * (const1 * common * dot - (float)pow(-1, side1 + side2) * const22);
                    if (abs(mx - nx) + abs(my - ny) > 0) {
                        float cross;
                        if (type == 1)
                            cross = deltay * 0.5f * d * (z11_m * v1 + z10_m - z11_n * v2 - z10_n);
                        else if (type == 2)
                            cross = -deltax * 0.5f * d * (z11_m * v1 + z10_m) - deltay * 0.5f * d * (z11_n * u2 + z01_n) + deltaz * 0.25f * d * d;
                        else
                            cross = deltax * 0.5f * d * (z11_n * u2 + z01_n - z11_m * u1 - z01_m);
                        em0 += weight * green2 * common * cross;
                    }
                }
            }
        }
    }
    store1 += ee1;
    store2 += ee2;
    store3 += em0;
}

/// @brief Destroy the FFT computation plans and deallocated associated memory
void MVProd0::cleanAll() {
    fftwf_destroy_plan(forward2);
    fftwf_destroy_plan(forward3);
    fftwf_destroy_plan(forward41);
    fftwf_destroy_plan(forward42);
    fftwf_destroy_plan(backward1);
    fftwf_destroy_plan(backward2);
    fftwf_destroy_plan(backward3);
    fftwf_cleanup_threads();
}