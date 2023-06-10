/* This module computes scattered fields from surface currents and derive BRDF values.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "Scattering.h"

/// @brief Compute scattered fields from surface currents and derive BRDF values
/// @param est: an Estimate object
/// @param grid: a Grid object
Scattering::Scattering(Estimate* est, Grid* grid) {
    this->Nx = est->Nx;
    this->Ny = est->Ny;
    this->grid = grid;
    hori_num = (Nx - 1) * Ny;
    vert_num = (Ny - 1) * Nx;
    num_pts = grid->num_pts;
    numX = grid->numX;
    numY = grid->numY;
    numZ = grid->numZ;
    
    // Determine whether we need to filter and downsample the data based on the simulation resolution
    space_range = 48;
    double d0 = grid->dx;
    if (d0 >= 0.05) {
        downSample = false;
        M1 = round(space_range / grid->dx);
        M2 = round(space_range / grid->dy);
        M3 = round(space_range / grid->dz);
    } else {
        downSample = true;
        M1 = round(space_range / grid->dx) / 2;
        M2 = round(space_range / grid->dy) / 2;
        M3 = round(space_range / grid->dz);
        
        // Set up a 21-point low-pass filter
        VectorXd pts = VectorXd::LinSpaced(21, -10 * d0, 10 * d0);
        double interval = 11.0 * d0 / 4.0;
        double B = 0.5 / interval;
        double midpoint = 2.0 * interval;
        double c0 = 0.5 / (midpoint * midpoint);
        filter = VectorXf::Zero(21);
        normalize = 0;
        for (int i = 0; i < 21; i++) {
            double x0 = abs(pts(i));
            double val;
            if (x0 > midpoint)
                val = (c0 * x0 * x0 - 4.0 * c0 * midpoint * x0 + 4.0 * c0 * midpoint * midpoint) * 2.0 * B * sin(2.0 * B * M_PI * x0) / (2.0 * B * M_PI * x0);
            else if (x0 == 0)
                val = 2.0 * B;
            else
                val = (1.0 - c0 * x0 * x0) * 2.0 * B * sin(2.0 * B * M_PI * x0) / (2.0 * B * M_PI * x0);
            normalize += val;
            filter(i) = (float)val;
        }
    }
    
    // Initialize FFT computations
    fft_block.resize(M1 * M2 * M3, 1);
    fft_data = (fftwf_complex*) fft_block.data();
    fftwf_init_threads();
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);
    int num_thr = (int) nb_threads;
    fftwf_plan_with_nthreads(num_thr);
    plan = fftwf_plan_dft_3d(M1, M2, M3, fft_data, fft_data, FFTW_BACKWARD, FFTW_MEASURE);
}

/// @brief Compute a BRDF lobe from a set of computed current densities
/// @brief This method should only be called in single simulations, rather than in beam steering contexts
/// @param eta: index of refraction of the medium where the light is incident from
/// @param lambda: the simulated wavelength
/// @param x: the BEM linear system solution vector, representing the current densities
/// @param outres: resolution of the BRDF as a function of outgoing directions
/// @param irra: the total integrated incident power on the surface
void Scattering::computeBRDF(double eta, double lambda, VectorXcf x, int outres, float irra) {
    computeFields(eta, lambda, x, outres, 0, 0);
    brdf = MatrixXf::Zero(outres, outres);

    // Compute BRDF values from far field scattered field values, for each queried outgoing direction
    for (int i = 0; i < outres; i++) {
        for (int j = 0; j < outres; j++) {
            float r1 = (i + 0.5f) / outres * 2.0f - 1.0f;
            float r2 = (j + 0.5f) / outres * 2.0f - 1.0f;
            float test = 1 - r1 * r1 - r2 * r2;
            if (test < 0)
                continue;
            float cos_r = sqrt(test);
            fcomp Ex0 = Ex(i, j);
            fcomp Ey0 = Ey(i, j);
            fcomp Ez0 = Ez(i, j);
            fcomp Hx0 = conj(Hx(i, j));
            fcomp Hy0 = conj(Hy(i, j));
            fcomp Hz0 = conj(Hz(i, j));
            Vector3f poynting;
            poynting(0) = 0.5f * (Ey0 * Hz0 - Ez0 * Hy0).real();
            poynting(1) = 0.5f * (Ez0 * Hx0 - Ex0 * Hz0).real();
            poynting(2) = 0.5f * (Ex0 * Hy0 - Ey0 * Hx0).real();
            brdf(i, j) = poynting.norm() / (cos_r * irra);
        }
    }
}

/// @brief Compute the far field scattered field values from a surface, given a set of computed current densities
/// @brief This method can be called by the computeBRDF() method, or called externally by the driver code in beam steering subregion simulations
/// @param eta: index of refraction of the medium where the light is incident from
/// @param lambda: the simulated wavelength
/// @param x: the BEM linear system solution vector, representing the current densities
/// @param outres: resolution of the BRDF as a function of outgoing directions
/// @param xshift: this value is nonzero only in beam steering contexts; it represents the shift in the x direction from the global to the subregion coordinate
/// @param yshift: this value is nonzero only in beam steering contexts; it represents the shift in the y direction from the global to the subregion coordinate
void Scattering::computeFields(double eta, double lambda, VectorXcf x, int outres, double xshift, double yshift) {
    double eps = 1 / (mu * c * c) * eta * eta;
    double k = 2 * M_PI / lambda * eta;
    double omega = c / lambda * 2 * M_PI;
    c1 = (fcomp)(cuDB * k * k / (4.0 * M_PI * omega * eps));
    c2 = (fcomp)(-cuDB * k / (4.0 * M_PI));
    c3 = (fcomp)(cuDB * k / (4.0 * M_PI));
    c4 = (fcomp)(cuDB * k * k / (4.0 * M_PI * omega * mu));
    
    // Compute Fourier transformed Jx, Jy, Jz, Mx, My, Mz values
    pointSources(x);
    h_interp = ceil(space_range * eta / lambda);
    n_interp = 2 * h_interp + 1;
    r_range = VectorXf::LinSpaced(n_interp, -h_interp * lambda / (space_range * eta), h_interp * lambda / (space_range * eta));
    interp_data = MatrixXcf::Zero(n_interp * n_interp * n_interp, 6);
    for (int i = 0; i < 6; i++)
        computeComponent(i);
    
    // Compute field values through trilinearly interpolating results from previous FFT computations
    Ex = MatrixXcf::Zero(outres, outres);
    Ey = MatrixXcf::Zero(outres, outres);
    Ez = MatrixXcf::Zero(outres, outres);
    Hx = MatrixXcf::Zero(outres, outres);
    Hy = MatrixXcf::Zero(outres, outres);
    Hz = MatrixXcf::Zero(outres, outres);
    for (int num_x = 0; num_x < outres; num_x++) {
        for (int num_y = 0; num_y < outres; num_y++) {
            float r1 = (num_x + 0.5f) / outres * 2.0f - 1.0f;
            float r2 = (num_y + 0.5f) / outres * 2.0f - 1.0f;
            float test = 1 - r1 * r1 - r2 * r2;
            if (test < 0)
                continue;
            float r3 = sqrt(test);
            interpolate(num_x, num_y, r1, r2, r3);
            fcomp phase_shift = (fcomp)(exp(cuDB * k * (xshift * r1 + yshift * r2)));
            Ex(num_x, num_y) = phase_shift * Ex(num_x, num_y);
            Ey(num_x, num_y) = phase_shift * Ey(num_x, num_y);
            Ez(num_x, num_y) = phase_shift * Ez(num_x, num_y);
            Hx(num_x, num_y) = phase_shift * Hx(num_x, num_y);
            Hy(num_x, num_y) = phase_shift * Hy(num_x, num_y);
            Hz(num_x, num_y) = phase_shift * Hz(num_x, num_y);
        }
    }
}

/// @brief Project the current density distribution from basis functions to point sources in the 3D grid
/// @param x: the BEM linear system solution vector, representing the current densities
void Scattering::pointSources(VectorXcf x) {
    VectorXcf J_hori = x.block(0, 0, hori_num, 1);
    VectorXcf J_vert = x.block(hori_num, 0, vert_num, 1);
    VectorXcf M_hori = x.block(hori_num + vert_num, 0, hori_num, 1);
    VectorXcf M_vert = x.block(2 * hori_num + vert_num, 0, vert_num, 1);
    JMxyz = MatrixXcf::Zero(numX * numY * numZ, 6);

    // Project from horizontally arranged basis functions
    for (int count = 0; count < hori_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int x0 = grid->hori_b(pt, count) / (grid->totalY * grid->totalZ);
            int z0 = grid->hori_b(pt, count) % grid->totalZ;
            int y0 = (grid->hori_b(pt, count) - z0 - grid->totalY * grid->totalZ * x0) / grid->totalZ;
            x0 = x0 - numX;
            y0 = y0 - numY;
            z0 = z0 - numZ;
            int linear = numX * numY * z0 + numX * y0 + x0;
            JMxyz(linear, 0) += J_hori(count) * grid->hori_x(pt, count);
            JMxyz(linear, 2) += J_hori(count) * grid->hori_z(pt, count);
            JMxyz(linear, 3) += M_hori(count) * grid->hori_x(pt, count);
            JMxyz(linear, 5) += M_hori(count) * grid->hori_z(pt, count);
        }
    }
    
    // Project from vertically arranged basis functions
    for (int count = 0; count < vert_num; count++) {
        for (int pt = 0; pt < num_pts; pt++) {
            int x0 = grid->vert_b(pt, count) / (grid->totalY * grid->totalZ);
            int z0 = grid->vert_b(pt, count) % grid->totalZ;
            int y0 = (grid->vert_b(pt, count) - z0 - grid->totalY * grid->totalZ * x0) / grid->totalZ;
            x0 = x0 - numX;
            y0 = y0 - numY;
            z0 = z0 - numZ;
            int linear = numX * numY * z0 + numX * y0 + x0;
            JMxyz(linear, 1) += J_vert(count) * grid->vert_y(pt, count);
            JMxyz(linear, 2) += J_vert(count) * grid->vert_z(pt, count);
            JMxyz(linear, 4) += M_vert(count) * grid->vert_y(pt, count);
            JMxyz(linear, 5) += M_vert(count) * grid->vert_z(pt, count);
        }
    }
}

/// @brief Optionally smooth and downsample, and then (inverse) Fourier transform the point source distributions
/// @param component: an integer with the value of 0 (Jx), 1 (Jy), 2 (Jz), 3 (Mx), 4 (My), or 5 (Mz) that selects a point source distribution to process
void Scattering::computeComponent(int component) {
    int xLow = round(grid->xrange(0) / grid->dx);
    int yLow = round(grid->yrange(0) / grid->dy);
    int zLow = round(grid->zrange(0) / grid->dz);
    fft_block = MatrixXcf::Zero(M1 * M2 * M3, 1);
    
    // Optionally apply the low-pass filter and downsample the data, and then compute inverse FFTs
    if (downSample) {
        // Need to shuffle the order of the data
        parallel_for(numZ, [&](int start, int end) {
        for (int nz = start; nz < end; nz++) {
            int indz = nz + zLow;
            if (indz < 0)
                indz = indz + M3;
            for (int ny = 0; ny < numY; ny++) {
                int indy = ny + yLow;
                if (indy < 0)
                    indy = indy + 2 * M2;
                if (indy % 2 != 0)
                    continue;
                indy = indy / 2;
                for (int nx = 0; nx < numX; nx++) {
                    int indx = nx + xLow;
                    if (indx < 0)
                        indx = indx + 2 * M1;
                    if (indx % 2 != 0)
                        continue;
                    indx = indx / 2;
                    fcomp sum = 0;
                    for (int i = 0; i < 21; i++) {
                        int shift_y = ny + i - 10;
                        if (shift_y < 0 || shift_y >= numY)
                            continue;
                        for (int j = 0; j < 21; j++) {
                            int shift_x = nx + j - 10;
                            if (shift_x < 0 || shift_x >= numX)
                                continue;
                            sum += filter(i) * filter(j) * JMxyz(numX * numY * nz + numX * shift_y + shift_x, component);
                        }
                    }
                    fft_block(M2 * M3 * indx + M3 * indy + indz, 0) = sum;
                }
            }
        }
        } );
        fft_block = fft_block / (float)(0.25 * normalize * normalize);
    } else {
        // Need to shuffle the order of the data
        parallel_for(numZ, [&](int start, int end) {
        for (int nz = start; nz < end; nz++) {
            int indz = nz + zLow;
            if (indz < 0)
                indz = indz + M3;
            for (int ny = 0; ny < numY; ny++) {
                int indy = ny + yLow;
                if (indy < 0)
                    indy = indy + M2;
                for (int nx = 0; nx < numX; nx++) {
                    int indx = nx + xLow;
                    if (indx < 0)
                        indx = indx + M1;
                    fft_block(M2 * M3 * indx + M3 * indy + indz, 0) = JMxyz(numX * numY * nz + numX * ny + nx, component);
                }
            }
        }
        } );
    }
    if (component >= 3)
        fft_block = eta0FL * fft_block;
    fftwf_execute(plan);
    
    // Interpolation
    parallel_for(n_interp, [&](int start, int end) {
    for (int n1 = start; n1 < end; n1++) {
        int m1 = n1 - h_interp;
        if (m1 < 0)
            m1 = m1 + M1;
        for (int n2 = 0; n2 < n_interp; n2++) {
            int m2 = n2 - h_interp;
            if (m2 < 0)
                m2 = m2 + M2;
            for (int n3 = 0; n3 < n_interp; n3++) {
                int m3 = n3 - h_interp;
                if (m3 < 0)
                    m3 = m3 + M3;
                int ind_small = n_interp * n_interp * n1 + n_interp * n2 + n3;
                int ind_large = M2 * M3 * m1 + M3 * m2 + m3;
                interp_data(ind_small, component) = fft_block(ind_large, 0);
            }
        }
    }
    } );
}

/// @brief Trilinearly interpolating results from previous FFT computations
/// @param num1: row index into the matrices storing far field scattered field values
/// @param num2: column index into the matrices storing far field scattered field values
/// @param r1: 1st Cartesian component of the unit outgoing direction
/// @param r2: 2nd Cartesian component of the unit outgoing direction
/// @param r3: 3rd Cartesian component of the unit outgoing direction
void Scattering::interpolate(int num1, int num2, float r1, float r2, float r3) {
    int s1 = floor((r1 - r_range(0)) / (r_range(1) - r_range(0)));
    int s2 = floor((r2 - r_range(0)) / (r_range(1) - r_range(0)));
    int s3 = floor((r3 - r_range(0)) / (r_range(1) - r_range(0)));
    float t1 = (r1 - r_range(s1)) / (r_range(1) - r_range(0));
    float t2 = (r2 - r_range(s2)) / (r_range(1) - r_range(0));
    float t3 = (r3 - r_range(s3)) / (r_range(1) - r_range(0));
    if (s1 == n_interp - 1) {
        s1 = n_interp - 2;
        t1 = 1.0f;
    }
    if (s2 == n_interp - 1) {
        s2 = n_interp - 2;
        t2 = 1.0f;
    }
    if (s3 == n_interp - 1) {
        s3 = n_interp - 2;
        t3 = 1.0f;
    }
    VectorXcf vals = VectorXcf::Zero(6);
    for (int i = 0; i < 6; i++) {
        int base1 = n_interp * n_interp * s1 + n_interp * s2 + s3;
        vals(i) += (1 - t1) * (1 - t2) * (1 - t3) * interp_data(base1, i);
        vals(i) += (1 - t1) * (1 - t2) * t3 * interp_data(base1 + 1, i);
        vals(i) += (1 - t1) * t2 * (1 - t3) * interp_data(base1 + n_interp, i);
        vals(i) += (1 - t1) * t2 * t3 * interp_data(base1 + n_interp + 1, i);
        int base2 = base1 + n_interp * n_interp;
        vals(i) += t1 * (1 - t2) * (1 - t3) * interp_data(base2, i);
        vals(i) += t1 * (1 - t2) * t3 * interp_data(base2 + 1, i);
        vals(i) += t1 * t2 * (1 - t3) * interp_data(base2 + n_interp, i);
        vals(i) += t1 * t2 * t3 * interp_data(base2 + n_interp + 1, i);
    }

    // Formulas come from the source-field relationships in homogeneous media
    Ex(num1, num2) += c1 * (r1 * r3 * vals(2) + r1 * r2 * vals(1) - (1 - r1 * r1) * vals(0));
    Ey(num1, num2) += c1 * (r2 * r1 * vals(0) + r2 * r3 * vals(2) - (1 - r2 * r2) * vals(1));
    Ez(num1, num2) += c1 * (r3 * r2 * vals(1) + r3 * r1 * vals(0) - (1 - r3 * r3) * vals(2));
    Ex(num1, num2) += c2 * (r3 * vals(4) - r2 * vals(5));
    Ey(num1, num2) += c2 * (r1 * vals(5) - r3 * vals(3));
    Ez(num1, num2) += c2 * (r2 * vals(3) - r1 * vals(4));
    Hx(num1, num2) += c3 * (r3 * vals(1) - r2 * vals(2));
    Hy(num1, num2) += c3 * (r1 * vals(2) - r3 * vals(0));
    Hz(num1, num2) += c3 * (r2 * vals(0) - r1 * vals(1));
    Hx(num1, num2) += c4 * (r1 * r3 * vals(5) + r1 * r2 * vals(4) - (1 - r1 * r1) * vals(3));
    Hy(num1, num2) += c4 * (r2 * r1 * vals(3) + r2 * r3 * vals(5) - (1 - r2 * r2) * vals(4));
    Hz(num1, num2) += c4 * (r3 * r2 * vals(4) + r3 * r1 * vals(3) - (1 - r3 * r3) * vals(5));
}

/// @brief Destroy the FFT computation plans and deallocated associated memory
void Scattering::cleanAll() {
    fftwf_destroy_plan(plan);
    fftwf_cleanup_threads();
}