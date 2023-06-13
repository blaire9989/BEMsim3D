#include <unistd.h>
#include "Incidence.h"
#include "Scattering.h"
#include "Solver.h"

int main(int argc, char **argv) {
    int opt, bx = -1, by = -1, color = -1, dir = -1, nx, ny, outres;
    double exterior, l_um, w_um;
    string zname;
    while ((opt = getopt(argc, argv, "a:b:c:d:e:l:m:n:o:w:z:")) != -1) {
        switch (opt) {
            case 'a': bx = atoi(optarg); break;
            case 'b': by = atoi(optarg); break;
            case 'c': color = atoi(optarg); break;
            case 'd': dir = atoi(optarg); break;
            case 'e': exterior = atof(optarg); break;
            case 'l': l_um = atof(optarg); break;
            case 'm': nx = atoi(optarg); break;
            case 'n': ny = atoi(optarg); break;
            case 'o': outres = atoi(optarg); break;
            case 'w': w_um = atof(optarg); break;
            case 'z': zname = optarg; break;
        }
    }

    // Extract height field data, required incident directions, and simulated wavelengths
    MatrixXd z, wi, wvl = readData("data/" + zname + "/wvl.txt");
    bool useProvided, isDielectric = (wvl(0, 2) == 0);
    double shift;
    if (bx == -1 || by == -1) {
        // Characterizing an entire surface sample
        z = readData("data/" + zname + "/zvals.txt");
        wi = readData("data/" + zname + "/wi.txt");
        useProvided = false;
        shift = 0;
    } else {
        // Characterizing a subregion of a surface sample
        z = readData("data/" + zname + "/patch" + to_string(bx) + to_string(by) + "/zvals.txt");
        MatrixXd global = readData("data/" + zname + "/zvals.txt");
        useProvided = true;
        shift = global.mean();
        readBinary("data/" + zname + "/basic.binary", wi);
        if (color == -1) {
            printf("You did not specify the considered wavelength for this subregion simulation. ");
            printf("Please provide a integer between 0 and %d, inclusive. ", int(wvl.rows() - 1));
            printf("Only simulating the first wavelength.\n");
        }
        if (dir == -1) {
            printf("You did not specify the considered incident direction for this subregion simulation. ");
            printf("Please provide a integer between 0 and %d, inclusive. ", int(wi.rows() - 1));
            printf("Only simulating the first incident direction.\n");
        }
    }

    // Estimate the memory usage in each simulation and select a computation plan
    Estimate* est = new Estimate(l_um, l_um, z, useProvided, shift);
    double mem11, mem12, mem21, mem22, mem34;
    est->memGB(mem11, mem12, mem21, mem22, mem34);
    int deviceCount, plan, ind1 = -1, ind40 = -1, ind41 = -1, ind42 = -1, ind43 = -1;
    cudaError_t t = cudaGetDeviceCount(&deviceCount);
    if (t != cudaSuccess)
        plan = 0;
    else if (deviceCount < 4) {
        double maxMem = 0;
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            double currMem = prop.totalGlobalMem / pow(1024.0, 3);
            if (currMem > maxMem) {
                ind1 = i;
                maxMem = currMem;
            }
        }
        if (isDielectric) {
            if (mem21 <= maxMem)
                plan = 2;
            else if (mem11 <= maxMem)
                plan = 1;
            else
                plan = 0;
        } else {
            if (mem22 <= maxMem)
                plan = 2;
            else if (mem12 <= maxMem)
                plan = 1;
            else
                plan = 0;
        }
    } else {
        vector<double> mem;
        vector<int> inds;
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            double currMem = prop.totalGlobalMem / pow(1024.0, 3);
            mem.push_back(currMem);
            inds.push_back(i);
        }
        for (int i = 0; i < deviceCount; i++) {
            for (int j = i + 1; j < deviceCount; j++) {
                if (mem[i] < mem[j]) {
                    double temp1 = mem[i];
                    mem[i] = mem[j];
                    mem[j] = temp1;
                    int temp2 = inds[i];
                    inds[i] = inds[j];
                    inds[j] = temp2;
                }
            }
        }
        double maxMem = mem[3];
        ind40 = inds[0];
        ind41 = inds[1];
        ind42 = inds[2];
        ind43 = inds[3];
        if (isDielectric) {
            if (mem34 <= maxMem)
                plan = 3;
            else
                plan = 0;
        } else {
            if (mem34 <= maxMem)
                plan = 4;
            else
                plan = 0;
        }
    }
    if (plan == 0)
        printf("Performing simulations on the CPU because there is no GPU detected or GPU memory is insufficient for your simulation size.\n");
    
    // Initialize simulations
    Singular* singular;
    MVProd* mv;
    Grid* grid = new Grid(est);
    if (plan == 0) {
        singular = new Singular0(est);
        mv = new MVProd0(est, singular, grid);
    } else if (plan <= 2) {
        singular = new Singular12(est, ind1);
        if (plan == 1)
            mv = new MVProd1(est, singular, grid, isDielectric, ind1);
        else
            mv = new MVProd2(est, singular, grid, isDielectric, ind1);
    } else {
        singular = new Singular34(est, ind40, ind41, ind42, ind43);
        if (plan == 3)
            mv = new MVProd3(est, singular, grid, ind40, ind41, ind42, ind43);
        else
            mv = new MVProd4(est, singular, grid, ind40, ind41, ind42, ind43);
    }
    Incidence* inci = new Incidence(est);
    Solver* solver = new Solver(est, mv);
    Scattering* scatter = new Scattering(est, grid);
    
    // Perform simulations for the required wavelengths and illumination directions
    int c1 = color, c2 = color + 1, d1 = dir, d2 = dir + 1;
    if (color == -1) {
        c1 = 0;
        if (bx == -1 || by == -1)
            c2 = wvl.rows();
        else
            c2 = 1;
    }
    if (dir == -1) {
        d1 = 0;
        if (bx == -1 || by == -1)
            d2 = wi.rows();
        else
            d2 = 1;
    }
    for (int i = c1; i < c2; i++) {
        double lambda = wvl(i, 0);
        dcomp interior(wvl(i, 1), -wvl(i, 2));
        mv->setParameters(exterior, interior, lambda);
        for (int j = d1; j < d2; j++) {
            printf("Simulating wavelength %d, direction %d...\n", i, j);
            inci->setParameters(exterior, lambda, w_um, wi(j, 0), wi(j, 1));
            inci->computeVector(1);
            solver->csMINRES(inci->b, 2500, 2.5e-4);
            VectorXcf x1 = solver->x;
            inci->computeVector(2);
            solver->csMINRES(inci->b, 2500, 2.5e-4);
            VectorXcf x2 = solver->x;
            if (bx == -1 || by == -1) {
                MatrixXf averageBRDF = MatrixXf::Zero(outres, outres);
                scatter->computeBRDF(exterior, lambda, x1, outres, (float)(inci->irradiance1));
                averageBRDF = averageBRDF + 0.5f * scatter->brdf;
                scatter->computeBRDF(exterior, lambda, x2, outres, (float)(inci->irradiance2));
                averageBRDF = averageBRDF + 0.5f * scatter->brdf;
                writeData("data/" + zname + "/BRDF_wvl" + to_string(i) + "_wi" + to_string(j) + ".binary", averageBRDF);
            } else {
                // For subregion simulations, only one wavelength and direction is simulated at a time.
                MatrixXcf EH = MatrixXcf::Zero(2 * outres, 3 * outres);
                double xshift = 0.5 * w_um * (2 * bx - nx + 1);
                double yshift = 0.5 * w_um * (2 * by - ny + 1);
                MatrixXf EH_real, EH_imag;
                scatter->computeFields(exterior, lambda, x1, outres, xshift, yshift);
                EH.block(0 * outres, 0 * outres, outres, outres) = scatter->Ex;
                EH.block(0 * outres, 1 * outres, outres, outres) = scatter->Ey;
                EH.block(0 * outres, 2 * outres, outres, outres) = scatter->Ez;
                EH.block(1 * outres, 0 * outres, outres, outres) = scatter->Hx;
                EH.block(1 * outres, 1 * outres, outres, outres) = scatter->Hy;
                EH.block(1 * outres, 2 * outres, outres, outres) = scatter->Hz;
                EH_real = EH.real();
                EH_imag = EH.imag();
                // Scattered fields written as intermediate results. Will be post-processed to synthesize BRDFs and then overwritten.
                writeBinary("data/" + zname + "/patch" + to_string(bx) + to_string(by) + "/EH1_real.binary", EH_real);
                writeBinary("data/" + zname + "/patch" + to_string(bx) + to_string(by) + "/EH1_imag.binary", EH_imag);
                scatter->computeFields(exterior, lambda, x2, outres, xshift, yshift);
                EH.block(0 * outres, 0 * outres, outres, outres) = scatter->Ex;
                EH.block(0 * outres, 1 * outres, outres, outres) = scatter->Ey;
                EH.block(0 * outres, 2 * outres, outres, outres) = scatter->Ez;
                EH.block(1 * outres, 0 * outres, outres, outres) = scatter->Hx;
                EH.block(1 * outres, 1 * outres, outres, outres) = scatter->Hy;
                EH.block(1 * outres, 2 * outres, outres, outres) = scatter->Hz;
                EH_real = EH.real();
                EH_imag = EH.imag();
                // Scattered fields written as intermediate results. Will be post-processed to synthesize BRDFs and then overwritten.
                writeBinary("data/" + zname + "/patch" + to_string(bx) + to_string(by) + "/EH2_real.binary", EH_real);
                writeBinary("data/" + zname + "/patch" + to_string(bx) + to_string(by) + "/EH2_imag.binary", EH_imag);
            }
        }
    }
    
    // Deallocate memory
    mv->cleanAll();
    scatter->cleanAll();
    delete est;
    delete singular;
    delete grid;
    delete mv;
    delete inci;
    delete solver;
    delete scatter;
}