/* Implementation of a MINRES solver for complex symmetric linear systems.
Users do not need to read or understand any method in this file. 
PLEASE DO NOT MODIFY THE FOLLOWING CODE. */

#include "Solver.h"

/// @brief A MINRES solver for complex symmetric linear systems
/// @param est: an Estimate object
/// @param mv: an MVProd object
Solver::Solver(Estimate* est, MVProd* mv) {
    int Nx = est->Nx;
    int Ny = est->Ny;
    hori_num = (Nx - 1) * Ny;
    vert_num = (Ny - 1) * Nx;
    this->mv = mv;
}

/// @brief Execute MINRES iterations to solve the given linear system
/// @param b: the right-hand-side vector
/// @param maxit: the maximum number of iterations
/// @param rtol: the error tolerance for termination
/// @param show: a boolean that indicates whether to print information from each iteration
/// @return the solution vector
int Solver::csMINRES(VectorXcf b, int maxit, float rtol, bool show) {
    // Initializations
    x = VectorXcf::Zero(2 * hori_num + 2 * vert_num);
    int n = b.size();
    VectorXcf r2 = b, r3 = r2;
    float beta1 = r2.norm();
    int flag0 = -2, flag = flag0, iter = 0;
    float beta = 0, betal = 0, betan = beta1;
    fcomp tau = 0, taul = 0, phi = beta1;
    fcomp cs = -1, sn = 0, cr2 = -1, sr2 = 0, dltan = 0, eplnn = 0, gama = 0, gamal = 0, eta = 0, etal = 0, etal2 = 0, vepln = 0, veplnl = 0, veplnl2 = 0, ul = 0, ul2 = 0, ul3 = 0;
    float rnorm = betan, xnorm = 0, xl2norm = 0, Axnorm = 0, Anorm = 0;
    VectorXcf w = VectorXcf::Zero(n);
    VectorXcf wl = VectorXcf::Zero(n);
    VectorXcf r1 = VectorXcf::Zero(n);
    if (show) {
        printf("Min-length solution of symmetric (A-sI)x = b or min ||(A-sI)x - b||\n");
        printf("n  =%7i   ||b|| =%10.3e   rtol  =%10.3e   maxit  =%5i\n", n, beta1, rtol, maxit);
        printf("    iter     rnorm     Anorm      xnorm\n");
    }
    // Main iteration
    while (flag == flag0 && iter < maxit) {
        // Lanczos
        iter = iter + 1;
        betal = beta;
        beta = betan;
        VectorXcf v = r3 * (1 / beta);
        r3 = mv->multiply(v.conjugate());
        if (iter > 1)
            r3 = r3 - (beta / betal) * r1;
        fcomp alfa = v.adjoint() * r3;
        r3 = r3 - (alfa / beta) * r2;
        r1 = r2;
        r2 = r3;
        betan = r3.norm();
        if (iter == 1) {
            if (betan == 0) {
                if (alfa.real() == 0 && alfa.imag() == 0)
                    break;
                else {
                    x = b.conjugate() / alfa;
                    break;
                }
            }
        }
        // Apply previous left reflection Q_{k-1}
        fcomp dbar = dltan;
        fcomp dlta = cs * dbar + sn * alfa;
        fcomp epln = eplnn;
        fcomp gbar = conj(sn) * dbar - cs * alfa;
        eplnn = sn * betan;
        dltan = -cs * betan;
        fcomp dlta_QLP = dlta;
        // Compute the current left reflection Q_k
        fcomp gamal2 = gamal;
        gamal = gama;
        VectorXcf results1 = symOrtho(gbar, betan);
        cs = results1(0);
        sn = results1(1);
        gama = results1(2);
        fcomp gama_tmp = gama;
        fcomp taul2 = taul;
        taul = tau;
        tau = cs * phi;
        Axnorm = sqrt(Axnorm * Axnorm + norm(tau) * norm(tau));
        phi = phi * conj(sn);
        // Apply the previous right reflection P{k-2, k}
        if (iter > 2) {
            veplnl2 = veplnl;
            etal2 = etal;
            etal = eta;
            fcomp dlta_tmp = sr2 * vepln - cr2 * dlta;
            veplnl = cr2 * vepln + conj(sr2) * dlta;
            dlta = dlta_tmp;
            eta = conj(sr2) * gama;
            gama = -gama * cr2;
        }
        // Compute the current right reflection P{k-1, k}, P_12, P_23, ...
        if (iter > 1) {
            VectorXcf results2 = symOrtho(conj(gamal), conj(dlta));
            fcomp cr1 = results2(0);
            fcomp sr1 = results2(1);
            gamal = conj(results2(2));
            vepln = conj(sr1) * gama;
            gama = -gama * cr1;
        }
        // Update xnorm
        float xnorml = xnorm;
        fcomp ul4 = ul3;
        ul3 = ul2;
        if (iter > 2)
            ul2 = (taul2 - etal2 * ul4 - veplnl2 * ul3) / gamal2;
        if (iter > 1)
            ul = (taul - etal * ul3 - veplnl * ul2) / gamal;
        fcomp u = (tau - eta * ul2 - vepln * ul) / gama;
        xl2norm = sqrt(xl2norm * xl2norm + norm(ul2));
        xnorm = sqrt(xl2norm * xl2norm + norm(ul) + norm(u));
        // Update w, update x except if it will become too big
        VectorXcf wl2 = wl;
        wl = w;
        w = (v.conjugate() - epln * wl2 - dlta_QLP * wl) / gama_tmp;
        x = x + tau * w;
        // Compute the next right reflection P{k-1, k+1}
        VectorXcf results3 = symOrtho(conj(gamal), conj(eplnn));
        cr2 = results3(0);
        sr2 = results3(1);
        gamal = conj(results3(2));
        // Estimate various norms
        float abs_gama = abs(gama);
        float Anorml = Anorm;
        float max_temp = (Anorm < abs(gamal)) ? abs(gamal) : Anorm;
        Anorm = (max_temp < abs_gama) ? abs_gama : max_temp;
        float rnorml = rnorm;
        rnorm = abs(phi);
        float relres = rnorm / (Anorm * xnorm + beta1);
        // See if any of the stopping criteria are satisfied
        if (flag == flag0) {
            if (iter >= maxit)
                flag = 1;
            if (relres <= rtol)
                flag = 2;
        }
        if (show)
            printf("%7i %10.2e %10.2e %10.2e\n", iter - 1, rnorml, Anorml, xnorml);
    }
    // We have exited the main loop
    r1 = b - mv->multiply(x);
    rnorm = r1.norm();
    xnorm = x.norm();
    if (show)
        printf("%7i %10.2e %10.2e %10.2e\n", iter, rnorm, Anorm, xnorm);
    return iter;
}

/// @brief A helper function to the MINRES iterations
VectorXcf Solver::symOrtho(fcomp a1, fcomp a2) {
    VectorXcf results(3);
    float absa1 = abs(a1);
    float absa2 = abs(a2);
    fcomp signa1, signa2;
    if (absa1 == 0)
        signa1 = 0;
    else
        signa1 = a1 / absa1;
    if (absa2 == 0)
        signa2 = 0;
    else
        signa2 = a2 / absa2;
    if (a1.imag() == 0 && a2.imag() == 0) {
        if (a2.real() == 0) {
            if (a1.real() == 0)
                results << 1, 0, absa1;
            else
                results << signa1, 0, absa1;
            return results;
        } else if (a1.real() == 0) {
            results << 0, signa2, absa2;
            return results;
        }
        if (absa2 > absa1) {
            fcomp t = a1 / a2;
            fcomp s = signa2 / sqrt(1.0f + t * t);
            fcomp c = s * t;
            fcomp r = a2 / s;
            results << c, s, r;
        } else {
            fcomp t = a2 / a1;
            fcomp c = signa1 / sqrt(1.0f + t * t);
            fcomp s = c * t;
            fcomp r = a1 / c;
            results << c, s, r;
        }
        return results;
    }
    if (a2.real() == 0 && a2.imag() == 0) {
        results << 1.0f, 0.0f, a1;
        return results;
    } else if (a1.real() == 0 && a1.imag() == 0) {
        results << 0.0f, 1.0f, a2;
        return results;
    }
    if (absa2 > absa1) {
        fcomp t = absa1 / absa2;
        fcomp c = 1.0f / sqrt(1.0f + t * t);
        fcomp s = c * conj(signa2 / signa1);
        c = c * t;
        fcomp r = a2 / conj(s);
        results << c, s, r;
    } else {
        fcomp t = absa2 / absa1;
        fcomp c = 1.0f / sqrt(1.0f + t * t);
        fcomp s = c * t * conj(signa2) / conj(signa1);
        fcomp r = a1 / c;
        results << c, s, r;
    }
    return results;
}