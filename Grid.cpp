/* This module constructs the 3D grid of point sources used for approximating basis functions in BEM.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "Grid.h"

/// @brief Builds a 3D grid of point sources of appropriate size, using information in the given Estimate object
/// @brief Seeks to match the field radiated from each basis function that carries currents with the field radiated from nearby point sources
/// @param est: an Estimate object that contains information about the simulated surface
Grid::Grid(Estimate* est) {
    // 2D height field
    this->Nx = est->Nx;
    this->Ny = est->Ny;
    this->d = est->d;
    this->zvals = est->zvals;
    xvals = VectorXd::LinSpaced(Nx + 1, -Nx * d / 2, Nx * d / 2);
    yvals = VectorXd::LinSpaced(Ny + 1, -Ny * d / 2, Ny * d / 2);
    hori_num = (Nx - 1) * Ny;
    vert_num = (Ny - 1) * Nx;
    
    // 3D point source grid
    dx = 1.6 * d;
    dy = dx;
    dz = dx / 3 * 4;
    xrange = est->xrange;
    yrange = est->yrange;
    zrange = est->zrange;
    numX = xrange.rows();
    numY = yrange.rows();
    numZ = zrange.rows();
    totalX = 2 * numX;
    totalY = 2 * numY;
    totalZ = 2 * numZ;
    
    // Least-square solving
    readBinary("include/farFieldMatch.binary", directions);
    num_directions = directions.rows();
    num_pts = 48;
    regularization = 1e-4;
    order = 3;
    pvals = quadrature_points.block(order - 1, 0, 1, order).transpose();
    wvals = quadrature_weights.block(order - 1, 0, 1, order).transpose();
    computeIndices();
}

/// @brief Stores linear indices of point sources used to approximate each basis function into relevant data structures
/// @brief The forward and backward indices are both used in later steps and are convenient for maintaining thread safety
void Grid::computeIndices() {
    int total = numX * numY * numZ;
    
    // Compute horizontal indices
    MatrixXi hori_xyz = MatrixXi::Zero(11, hori_num);
    parallel_for(hori_num, [&](int start, int end) {
    for (int m = start; m < end; m++) {
        int my = m / (Nx - 1);
        int mx = m - (Nx - 1) * my;
        double xc = xvals(mx + 1);
        int xp = floor((xc - xrange(0)) / dx);
        if (xp - 1 < 0)
            xp = 1;
        if (xp + 2 >= numX)
            xp = numX - 3;
        Vector4i xvec(xp - 1, xp, xp + 1, xp + 2);
        double yc = (yvals(my) + yvals(my + 1)) / 2;
        int yp = floor((yc - yrange(0)) / dy);
        if (yp - 1 < 0)
            yp = 1;
        if (yp + 2 >= numY)
            yp = numY - 3;
        Vector4i yvec(yp - 1, yp, yp + 1, yp + 2);
        double zc = (zvals(mx + 1, my) + zvals(mx + 1, my + 1)) / 2;
        int zp = round((zc - zrange(0)) / dz);
        Vector3i zvec(zp - 1, zp, zp + 1);
        hori_xyz.block(0, m, 4, 1) = xvec;
        hori_xyz.block(4, m, 4, 1) = yvec;
        hori_xyz.block(8, m, 3, 1) = zvec;
    }
    } );
    
    // Fill in backward indices
    hori_b = MatrixXi::Zero(num_pts, hori_num);
    vector<vector<int>> indices1;
    for (int i = 0; i < total; i++) {
        vector<int> local;
        local.reserve(50);
        indices1.push_back(local);
    }
    for (int m = 0; m < hori_num; m++) {
        for (int nx = 0; nx < 4; nx++) {
            for (int ny = 0; ny < 4; ny++) {
                for (int nz = 0; nz < 3; nz++) {
                    int index = 12 * nx + 3 * ny + nz;
                    hori_b(index, m) = totalY * totalZ * (hori_xyz(nx, m) + numX) + totalZ * (hori_xyz(ny + 4, m) + numY) + (hori_xyz(nz + 8, m) + numZ);
                    int curr_pt = numY * numZ * hori_xyz(nx, m) + numZ * hori_xyz(ny + 4, m) + hori_xyz(nz + 8, m);
                    indices1[curr_pt].push_back(num_pts * m + index);
                }
            }
        }
    }
    
    // Fill in forward indices
    vector<int> points1;
    int max_size1 = 0;
    for (int i = 0; i < total; i++) {
        int curr_size = indices1[i].size();
        if (curr_size > 0) {
            points1.push_back(i);
            if (curr_size > max_size1)
                max_size1 = curr_size;
        }
    }
    int relevant1 = points1.size();
    hori_f = -MatrixXi::Ones(max_size1 + 1, relevant1);
    for (int i = 0; i < relevant1; i++) {
        int pt = points1[i];
        int x0 = pt / (numY * numZ);
        int z0 = pt % numZ;
        int y0 = (pt - z0 - numY * numZ * x0) / numZ;
        hori_f(0, i) = totalY * totalZ * x0 + totalZ * y0 + z0;
        int curr_size = indices1[pt].size();
        for (int j = 0; j < curr_size; j++)
            hori_f(j + 1, i) = indices1[pt][j];
    }
    
    // Compute vertical indices
    MatrixXi vert_xyz = MatrixXi::Zero(11, vert_num);
    parallel_for(vert_num, [&](int start, int end) {
    for (int m = start; m < end; m++) {
        int mx = m / (Ny - 1);
        int my = m - (Ny - 1) * mx;
        double xc = (xvals(mx) + xvals(mx + 1)) / 2;
        int xp = floor((xc - xrange(0)) / dx);
        if (xp - 1 < 0)
            xp = 1;
        if (xp + 2 >= numX)
            xp = numX - 3;
        Vector4i xvec(xp - 1, xp, xp + 1, xp + 2);
        double yc = yvals(my + 1);
        int yp = floor((yc - yrange(0)) / dy);
        if (yp - 1 < 0)
            yp = 1;
        if (yp + 2 >= numY)
            yp = numY - 3;
        Vector4i yvec(yp - 1, yp, yp + 1, yp + 2);
        double zc = (zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 2;
        int zp = round((zc - zrange(0)) / dz);
        Vector3i zvec(zp - 1, zp, zp + 1);
        vert_xyz.block(0, m, 4, 1) = xvec;
        vert_xyz.block(4, m, 4, 1) = yvec;
        vert_xyz.block(8, m, 3, 1) = zvec;
    }
    } );
    
    // Fill in backward indices
    vert_b = MatrixXi::Zero(num_pts, vert_num);
    vector<vector<int>> indices2;
    for (int i = 0; i < total; i++) {
        vector<int> local;
        local.reserve(50);
        indices2.push_back(local);
    }
    for (int m = 0; m < vert_num; m++) {
        for (int nx = 0; nx < 4; nx++) {
            for (int ny = 0; ny < 4; ny++) {
                for (int nz = 0; nz < 3; nz++) {
                    int index = 12 * nx + 3 * ny + nz;
                    vert_b(index, m) = totalY * totalZ * (vert_xyz(nx, m) + numX) + totalZ * (vert_xyz(ny + 4, m) + numY) + (vert_xyz(nz + 8, m) + numZ);
                    int curr_pt = numY * numZ * vert_xyz(nx, m) + numZ * vert_xyz(ny + 4, m) + vert_xyz(nz + 8, m);
                    indices2[curr_pt].push_back(num_pts * m + index);
                }
            }
        }
    }
    
    // Fill in forward indices
    vector<int> points2;
    int max_size2 = 0;
    for (int i = 0; i < total; i++) {
        int curr_size = indices2[i].size();
        if (curr_size > 0) {
            points2.push_back(i);
            if (curr_size > max_size2)
                max_size2 = curr_size;
        }
    }
    int relevant2 = points2.size();
    vert_f = -MatrixXi::Ones(max_size2 + 1, relevant2);
    for (int i = 0; i < relevant2; i++) {
        int pt = points2[i];
        int x0 = pt / (numY * numZ);
        int z0 = pt % numZ;
        int y0 = (pt - z0 - numY * numZ * x0) / numZ;
        vert_f(0, i) = totalY * totalZ * x0 + totalZ * y0 + z0;
        int curr_size = indices2[pt].size();
        for (int j = 0; j < curr_size; j++)
            vert_f(j + 1, i) = indices2[pt][j];
    }
}

/// @brief Compute point source approximation coefficients for basis functions, based on the given material parameters and simulated wavelength
/// @param eta1: index of refraction of the region where light is incident from, usually set to 1.0 (air) 
/// @param eta2: index of refraction of the simulated surface material (may be complex-valued) 
/// @param lambda: the currently simulated wavelength
void Grid::computeCoefficients(double eta1, dcomp eta2, double lambda) {
    k1 = 2 * M_PI / lambda * eta1;
    k2 = 2 * M_PI / lambda * eta2;
    computeHori(eta2.imag() == 0);
    computeVert(eta2.imag() == 0);
}

/// @brief Compute point source approximation coefficients for horizontally arranged (x-axis aligned) basis functions
/// @brief Finding the optimal approximation coefficients requires solving a least-square system for each basis function
/// @param isDielectric: a boolean value that indicates whether the surface material is dielectric 
void Grid::computeHori(bool isDielectric) {
    // Compute coefficients for the exterior region
    hori_x = MatrixXcf::Zero(num_pts, hori_num);
    hori_z = MatrixXcf::Zero(num_pts, hori_num);
    hori_d = MatrixXcf::Zero(num_pts, hori_num);
    parallel_for(hori_num, [&](int start, int end) {
    for (int m = start; m < end; m++) {
        MatrixXcf lhs = horiLHS(m, k1).cast<fcomp>();
        MatrixXcf rhs = horiRHS(m, k1).cast<fcomp>();
        VectorXcf vec1 = rhs.block(0, 0, num_directions + num_pts, 1);
        VectorXcf vec2 = rhs.block(0, 1, num_directions + num_pts, 1);
        VectorXcf vec3 = rhs.block(0, 2, num_directions + num_pts, 1);
        hori_x.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec1);
        hori_z.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec2);
        hori_d.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec3);
    }
    } );
    
    // Compute coefficients for the interior region if it is dielectric
    if (isDielectric) {
        hori_X = MatrixXcf::Zero(num_pts, hori_num);
        hori_Z = MatrixXcf::Zero(num_pts, hori_num);
        hori_D = MatrixXcf::Zero(num_pts, hori_num);
        parallel_for(hori_num, [&](int start, int end) {
        for (int m = start; m < end; m++) {
            MatrixXcf lhs = horiLHS(m, k2).cast<fcomp>();
            MatrixXcf rhs = horiRHS(m, k2).cast<fcomp>();
            VectorXcf vec1 = rhs.block(0, 0, num_directions + num_pts, 1);
            VectorXcf vec2 = rhs.block(0, 1, num_directions + num_pts, 1);
            VectorXcf vec3 = rhs.block(0, 2, num_directions + num_pts, 1);
            hori_X.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec1);
            hori_Z.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec2);
            hori_D.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec3);
        }
        } );
    }
}

/// @brief Helper function that computes the matrix in the least-square system for approximating one basis function
/// @param m: index of the horizontal basis function
/// @param k: wavenumber of the incident light
/// @return A computed least-square matrix
MatrixXcd Grid::horiLHS(int m, dcomp k) {
    int my = m / (Nx - 1);
    int mx = m - (Nx - 1) * my;
    double xc = xvals(mx + 1);
    int xp = floor((xc - xrange(0)) / dx);
    if (xp - 1 < 0)
        xp = 1;
    if (xp + 2 >= numX)
        xp = numX - 3;
    Vector4i xvec(xp - 1, xp, xp + 1, xp + 2);
    double yc = (yvals(my) + yvals(my + 1)) / 2;
    int yp = floor((yc - yrange(0)) / dy);
    if (yp - 1 < 0)
        yp = 1;
    if (yp + 2 >= numY)
        yp = numY - 3;
    Vector4i yvec(yp - 1, yp, yp + 1, yp + 2);
    double zc = (zvals(mx + 1, my) + zvals(mx + 1, my + 1)) / 2;
    int zp = round((zc - zrange(0)) / dz);
    Vector3i zvec(zp - 1, zp, zp + 1);
    
    // Compute matrix elements for the least squares system
    MatrixXcd results = MatrixXcd::Zero(num_directions + num_pts, num_pts);
    for (int nd = 0; nd < num_directions; nd++) {
        double r0 = directions(nd, 0);
        double r1 = directions(nd, 1);
        double r2 = directions(nd, 2);
        for (int nx = 0; nx < 4; nx++) {
            for (int ny = 0; ny < 4; ny++) {
                for (int nz = 0; nz < 3; nz++) {
                    int index = 12 * nx + 3 * ny + nz;
                    double x0 = xrange(xvec(nx));
                    double y0 = yrange(yvec(ny));
                    double z0 = zrange(zvec(nz));
                    results(nd, index) = exp(cuDB * k * (r0 * x0 + r1 * y0 + r2 * z0));
                }
            }
        }
    }
    
    // Add some regularization
    for (int i = 0; i < num_pts; i++)
        results(num_directions + i, i) = regularization;
    return results;
}

/// @brief Helper function that computes the right-hand-side vector in the least-square system for approximating one basis function
/// @param m: index of the horizontal basis function
/// @param k: wavenumber of the incident light
/// @return A computed least-square solution vector
MatrixXcd Grid::horiRHS(int m, dcomp k) {
    int my = m / (Nx - 1);
    int mx = m - (Nx - 1) * my;
    MatrixXcd results = MatrixXcd::Zero(num_directions + num_pts, 3);
    double xc, yc, z11, z10, z01, z00;
    
    // Integrate over the left half
    xc = (xvals(mx) + xvals(mx + 1)) / 2;
    yc = (yvals(my) + yvals(my + 1)) / 2;
    z11 = (zvals(mx, my) - zvals(mx + 1, my) - zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    z10 = (-zvals(mx, my) + zvals(mx + 1, my) - zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    z01 = (-zvals(mx, my) - zvals(mx + 1, my) + zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    z00 = (zvals(mx, my) + zvals(mx + 1, my) + zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    for (int nd = 0; nd < num_directions; nd++) {
        double r0 = directions(nd, 0);
        double r1 = directions(nd, 1);
        double r2 = directions(nd, 2);
        dcomp xxx = 0, zzz = 0, div = 0;
        for (int nu = 0; nu < order; nu++) {
            double u = pvals(nu);
            for (int nv = 0; nv < order; nv++) {
                double v = pvals(nv);
                double weight = wvals(nu) * wvals(nv);
                double x0 = xc + 0.5 * d * u;
                double y0 = yc + 0.5 * d * v;
                double z0 = z11 * u * v + z10 * u + z01 * v + z00;
                dcomp green = exp(cuDB * k * (r0 * x0 + r1 * y0 + r2 * z0));
                xxx += weight * green * (1 + u) * 0.5 * d;
                zzz += weight * green * (1 + u) * (z11 * v + z10);
                div += weight * green;
            }
        }
        results(nd, 0) += xxx;
        results(nd, 1) += zzz;
        results(nd, 2) += div;
    }
    
    // Integrate over the right half
    xc = (xvals(mx + 1) + xvals(mx + 2)) / 2;
    yc = (yvals(my) + yvals(my + 1)) / 2;
    z11 = (zvals(mx + 1, my) - zvals(mx + 2, my) - zvals(mx + 1, my + 1) + zvals(mx + 2, my + 1)) / 4;
    z10 = (-zvals(mx + 1, my) + zvals(mx + 2, my) - zvals(mx + 1, my + 1) + zvals(mx + 2, my + 1)) / 4;
    z01 = (-zvals(mx + 1, my) - zvals(mx + 2, my) + zvals(mx + 1, my + 1) + zvals(mx + 2, my + 1)) / 4;
    z00 = (zvals(mx + 1, my) + zvals(mx + 2, my) + zvals(mx + 1, my + 1) + zvals(mx + 2, my + 1)) / 4;
    for (int nd = 0; nd < num_directions; nd++) {
        double r0 = directions(nd, 0);
        double r1 = directions(nd, 1);
        double r2 = directions(nd, 2);
        dcomp xxx = 0, zzz = 0, div = 0;
        for (int nu = 0; nu < order; nu++) {
            double u = pvals(nu);
            for (int nv = 0; nv < order; nv++) {
                double v = pvals(nv);
                double weight = wvals(nu) * wvals(nv);
                double x0 = xc + 0.5 * d * u;
                double y0 = yc + 0.5 * d * v;
                double z0 = z11 * u * v + z10 * u + z01 * v + z00;
                dcomp green = exp(cuDB * k * (r0 * x0 + r1 * y0 + r2 * z0));
                xxx += weight * green * (1 - u) * 0.5 * d;
                zzz += weight * green * (1 - u) * (z11 * v + z10);
                div -= weight * green;
            }
        }
        results(nd, 0) += xxx;
        results(nd, 1) += zzz;
        results(nd, 2) += div;
    }
    return results;
}

/// @brief Compute point source approximation coefficients for vertically arranged (y-axis aligned) basis functions
/// @brief Finding the optimal approximation coefficients requires solving a least-square system for each basis function
/// @param isDielectric: a boolean value that indicates whether the surface material is dielectric 
void Grid::computeVert(bool isDielectric) {
    // Compute coefficients for the exterior region
    vert_y = MatrixXcf::Zero(num_pts, vert_num);
    vert_z = MatrixXcf::Zero(num_pts, vert_num);
    vert_d = MatrixXcf::Zero(num_pts, vert_num);
    parallel_for(vert_num, [&](int start, int end) {
    for (int m = start; m < end; m++) {
        MatrixXcf lhs = vertLHS(m, k1).cast<fcomp>();
        MatrixXcf rhs = vertRHS(m, k1).cast<fcomp>();
        VectorXcf vec1 = rhs.block(0, 0, num_directions + num_pts, 1);
        VectorXcf vec2 = rhs.block(0, 1, num_directions + num_pts, 1);
        VectorXcf vec3 = rhs.block(0, 2, num_directions + num_pts, 1);
        vert_y.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec1);
        vert_z.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec2);
        vert_d.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec3);
    }
    } );
    
    // Compute coefficients for the interior region if it is dielectric
    if (isDielectric) {
        vert_Y = MatrixXcf::Zero(num_pts, vert_num);
        vert_Z = MatrixXcf::Zero(num_pts, vert_num);
        vert_D = MatrixXcf::Zero(num_pts, vert_num);
        parallel_for(vert_num, [&](int start, int end) {
        for (int m = start; m < end; m++) {
            MatrixXcf lhs = vertLHS(m, k2).cast<fcomp>();
            MatrixXcf rhs = vertRHS(m, k2).cast<fcomp>();
            VectorXcf vec1 = rhs.block(0, 0, num_directions + num_pts, 1);
            VectorXcf vec2 = rhs.block(0, 1, num_directions + num_pts, 1);
            VectorXcf vec3 = rhs.block(0, 2, num_directions + num_pts, 1);
            vert_Y.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec1);
            vert_Z.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec2);
            vert_D.block(0, m, num_pts, 1) = lhs.colPivHouseholderQr().solve(vec3);
        }
        } );
    }
}

/// @brief Helper function that computes the matrix in the least-square system for approximating one basis function
/// @param m: index of the vertical basis function
/// @param k: wavenumber of the incident light
/// @return A computed least-square matrix
MatrixXcd Grid::vertLHS(int m, dcomp k) {
    int mx = m / (Ny - 1);
    int my = m - (Ny - 1) * mx;
    double xc = (xvals(mx) + xvals(mx + 1)) / 2;
    int xp = floor((xc - xrange(0)) / dx);
    if (xp - 1 < 0)
        xp = 1;
    if (xp + 2 >= numX)
        xp = numX - 3;
    Vector4i xvec(xp - 1, xp, xp + 1, xp + 2);
    double yc = yvals(my + 1);
    int yp = floor((yc - yrange(0)) / dy);
    if (yp - 1 < 0)
        yp = 1;
    if (yp + 2 >= numY)
        yp = numY - 3;
    Vector4i yvec(yp - 1, yp, yp + 1, yp + 2);
    double zc = (zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 2;
    int zp = round((zc - zrange(0)) / dz);
    Vector3i zvec(zp - 1, zp, zp + 1);
    
    // Compute matrix elements for the least squares system
    MatrixXcd results = MatrixXcd::Zero(num_directions + num_pts, num_pts);
    for (int num_d = 0; num_d < num_directions; num_d++) {
        double r0 = directions(num_d, 0);
        double r1 = directions(num_d, 1);
        double r2 = directions(num_d, 2);
        for (int num_x = 0; num_x < 4; num_x++) {
            for (int num_y = 0; num_y < 4; num_y++) {
                for (int num_z = 0; num_z < 3; num_z++) {
                    int index = 12 * num_x + 3 * num_y + num_z;
                    double x0 = xrange(xvec(num_x));
                    double y0 = yrange(yvec(num_y));
                    double z0 = zrange(zvec(num_z));
                    results(num_d, index) = exp(cuDB * k * (r0 * x0 + r1 * y0 + r2 * z0));
                }
            }
        }
    }
    
    // Add some regularization
    for (int i = 0; i < num_pts; i++)
        results(num_directions + i, i) = regularization;
    return results;
}

/// @brief Helper function that computes the right-hand-side vector in the least-square system for approximating one basis function
/// @param m: index of the vertical basis function
/// @param k: wavenumber of the incident light
/// @return A computed least-square solution vector
MatrixXcd Grid::vertRHS(int m, dcomp k) {
    int mx = m / (Ny - 1);
    int my = m - (Ny - 1) * mx;
    MatrixXcd results = MatrixXcd::Zero(num_directions + num_pts, 3);
    double xc, yc, z11, z10, z01, z00;
    
    // Integrate over the lower half
    xc = (xvals(mx) + xvals(mx + 1)) / 2;
    yc = (yvals(my) + yvals(my + 1)) / 2;
    z11 = (zvals(mx, my) - zvals(mx + 1, my) - zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    z10 = (-zvals(mx, my) + zvals(mx + 1, my) - zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    z01 = (-zvals(mx, my) - zvals(mx + 1, my) + zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    z00 = (zvals(mx, my) + zvals(mx + 1, my) + zvals(mx, my + 1) + zvals(mx + 1, my + 1)) / 4;
    for (int num_d = 0; num_d < num_directions; num_d++) {
        double r0 = directions(num_d, 0);
        double r1 = directions(num_d, 1);
        double r2 = directions(num_d, 2);
        dcomp yyy = 0, zzz = 0, div = 0;
        for (int num_u = 0; num_u < order; num_u++) {
            double u = pvals(num_u);
            for (int num_v = 0; num_v < order; num_v++) {
                double v = pvals(num_v);
                double weight = wvals(num_u) * wvals(num_v);
                double x0 = xc + 0.5 * d * u;
                double y0 = yc + 0.5 * d * v;
                double z0 = z11 * u * v + z10 * u + z01 * v + z00;
                dcomp green = exp(cuDB * k * (r0 * x0 + r1 * y0 + r2 * z0));
                yyy += weight * green * (1 + v) * 0.5 * d;
                zzz += weight * green * (1 + v) * (z11 * u + z01);
                div += weight * green;
            }
        }
        results(num_d, 0) += yyy;
        results(num_d, 1) += zzz;
        results(num_d, 2) += div;
    }
    
    // Integrate over the right half
    xc = (xvals(mx) + xvals(mx + 1)) / 2;
    yc = (yvals(my + 1) + yvals(my + 2)) / 2;
    z11 = (zvals(mx, my + 1) - zvals(mx + 1, my + 1) - zvals(mx, my + 2) + zvals(mx + 1, my + 2)) / 4;
    z10 = (-zvals(mx, my + 1) + zvals(mx + 1, my + 1) - zvals(mx, my + 2) + zvals(mx + 1, my + 2)) / 4;
    z01 = (-zvals(mx, my + 1) - zvals(mx + 1, my + 1) + zvals(mx, my + 2) + zvals(mx + 1, my + 2)) / 4;
    z00 = (zvals(mx, my + 1) + zvals(mx + 1, my + 1) + zvals(mx, my + 2) + zvals(mx + 1, my + 2)) / 4;
    for (int num_d = 0; num_d < num_directions; num_d++) {
        double r0 = directions(num_d, 0);
        double r1 = directions(num_d, 1);
        double r2 = directions(num_d, 2);
        dcomp yyy = 0, zzz = 0, div = 0;
        for (int num_u = 0; num_u < order; num_u++) {
            double u = pvals(num_u);
            for (int num_v = 0; num_v < order; num_v++) {
                double v = pvals(num_v);
                double weight = wvals(num_u) * wvals(num_v);
                double x0 = xc + 0.5 * d * u;
                double y0 = yc + 0.5 * d * v;
                double z0 = z11 * u * v + z10 * u + z01 * v + z00;
                dcomp green = exp(cuDB * k * (r0 * x0 + r1 * y0 + r2 * z0));
                yyy += weight * green * (1 - v) * 0.5 * d;
                zzz += weight * green * (1 - v) * (z11 * u + z01);
                div -= weight * green;
            }
        }
        results(num_d, 0) += yyy;
        results(num_d, 1) += zzz;
        results(num_d, 2) += div;
    }
    return results;
}