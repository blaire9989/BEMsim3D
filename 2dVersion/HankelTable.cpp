#include "HankelTable.h"

HankelTable::HankelTable(double lambda) {
    readBinary("include/htable.binary", dielectric);
    dielectric_min = 1e-4;
    dielectric_max = 6e3;
    dielectric_key = VectorXd::LinSpaced(6e7, dielectric_min, dielectric_max);
}

dcomp HankelTable::lookUp(dcomp k, double dist, int type) {
    if (k.imag() == 0)
        return lookUpDielectric(k.real() * dist, type);
    else
        return 0;
}

dcomp HankelTable::lookUpDielectric(double value, int type) {
    if (value <= dielectric_min)
        return 1.0 - cunit * 2.0 / M_PI * log(1.781 * value / 2.0);
    if (value >= dielectric_max)
        return 0;
    int num = floor(value / dielectric_min);
    double s = (value - dielectric_key(num - 1)) / dielectric_min;
    dcomp lower, upper;
    if (type == 0) {
        lower = dielectric(num - 1, 0) + cunit * dielectric(num - 1, 1);
        upper = dielectric(num, 0) + cunit * dielectric(num, 1);
    } else {
        lower = dielectric(num - 1, 2) + cunit * dielectric(num - 1, 3);
        upper = dielectric(num, 2) + cunit * dielectric(num, 3);
    }
    return (1 - s) * lower + s * upper;
}