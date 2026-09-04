#pragma once

/**
 * @file forces.hpp
 * @brief Single source of truth for the Kremer-Grest force model.
 *
 * Every tool that needs inter-particle forces (force ellipsoid, backbone,
 * pressure, ...) calls these functions instead of re-deriving the formulas
 * locally. The helpers are deliberately data-structure-agnostic: they take
 * raw displacements / points as std::array<double,3> and return force
 * vectors, so they compose with any particle representation in the suite.
 *
 * Model (Kremer-Grest + cosine attraction), all in LJ units (epsilon = sigma = 1):
 *   WCA repulsion     : r < 2^(1/6) sigma
 *   Cosine attraction : 2^(1/6) sigma <= r < sqrt(2) * 2^(1/6) sigma
 *   FENE bond         : U = -0.5 K Rmax^2 ln(1 - (r/Rmax)^2),  K = 30, Rmax = 1.5
 *   Harmonic bond     : U = 0.5 k (r - r0)^2,                  k = 30, r0  = 1.5
 *   Cosine bending    : U(theta) with cutoff theta_c = pi / b
 *
 * All force helpers return F = -grad U, i.e. the physical force on particle 1
 * for the displacement dr = r2 - r1 (see the per-function notes below).
 *
 * Sign convention: every pair helper returns the force on particle 1 given the
 * displacement dr = r2 - r1. The partner force is exactly its negative
 * (Newton's third law), so callers that need both simply negate.
 */

#include <array>
#include <cmath>
#include <algorithm>

namespace md {

using Vec3 = std::array<double, 3>;

// ── Kremer-Grest parameters ───────────────────────────────────────────────────
namespace kg {
inline constexpr double EPSILON   = 1.0;     ///< LJ well depth
inline constexpr double SIGMA_PP  = 1.0;     ///< polymer-polymer diameter
inline constexpr double SIGMA_PS  = 1.0;     ///< polymer-solvent diameter
inline constexpr double SIGMA_SS  = 0.75;    ///< solvent-solvent diameter
inline constexpr double ALPHA     = 0.5145;  ///< attractive-tail amplitude
inline constexpr double FENE_K    = 30.0;    ///< FENE spring constant [eps/sigma^2]
inline constexpr double FENE_R0   = 1.5;     ///< FENE maximum extension [sigma]
inline constexpr double BOND_K    = 30.0;    ///< harmonic spring constant
inline constexpr double BOND_R0   = 1.5;     ///< harmonic rest length [sigma]
inline constexpr double BEND_A    = 4.5;     ///< bending amplitude [eps]
inline constexpr double BEND_B    = 1.5;     ///< bending wavenumber
inline constexpr double TWO_POW_1_6 = 1.122462048309372981; ///< 2^(1/6)
} // namespace kg

/// WCA / attractive cutoff radius for a given diameter.
inline constexpr double wcaCutoff(double sigma) noexcept {
    return kg::TWO_POW_1_6 * sigma;
}

/// Minimum-image a single displacement component in a periodic box of length L.
inline void minimumImage(double& d, double L) noexcept {
    if      (d >  0.5 * L) d -= L;
    else if (d < -0.5 * L) d += L;
}

/// Minimum-image a full displacement vector (in place).
inline void minimumImage(Vec3& d, double Lx, double Ly, double Lz) noexcept {
    minimumImage(d[0], Lx);
    minimumImage(d[1], Ly);
    minimumImage(d[2], Lz);
}

// ── Non-bonded pair force (WCA core + cosine-attractive tail) ──────────────────

/**
 * @brief Non-bonded force on particle 1 due to particle 2.
 * @param dr     Displacement r2 - r1 (minimum-imaged by the caller if needed).
 * @param eps    LJ well depth.
 * @param sigma  Pair diameter (sets the WCA cutoff internally).
 * @param alpha  Attractive-tail amplitude.
 * @return       Force vector on particle 1.
 */
inline Vec3 nonbondedForce(const Vec3& dr, double eps, double sigma,
                           double alpha) noexcept {
    const double r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
    Vec3 f{0.0, 0.0, 0.0};

    const double rc = wcaCutoff(sigma);
    if (r2 < rc*rc && r2 >= 1e-24) {                 // WCA repulsive core
        const double s  = sigma / std::sqrt(r2);
        const double s6 = s*s*s*s*s*s;
        // F = (dU/dr)(dr/r) with U = 4eps(s^12 - s^6); dU/dr < 0 in the core,
        // so the force on particle 1 points away from particle 2 (repulsive).
        const double fw = -24.0 * eps * (2.0*s6*s6 - s6) / r2;
        f[0] += fw*dr[0]; f[1] += fw*dr[1]; f[2] += fw*dr[2];
    }

    const double r     = std::sqrt(r2);
    const double rCutA = rc * std::sqrt(2.0);
    if (r >= rc && r < rCutA && r >= 1e-12) {        // cosine-attractive tail
        // U = alpha[cos(pi (r/rc)^2) - 1]; coeff = (dU/dr)/r = -2 pi alpha sin(arg)/rc^2.
        const double arg = M_PI * (r/rc) * (r/rc);
        const double fa  = -alpha * 2.0 * M_PI * std::sin(arg) / (rc*rc);
        f[0] += fa*dr[0]; f[1] += fa*dr[1]; f[2] += fa*dr[2];
    }
    return f;
}

// ── Bonded forces ─────────────────────────────────────────────────────────────

/**
 * @brief FENE bond force on particle 1, dr = r2 - r1. Zero for r >= Rmax.
 *
 * From U = -0.5 K Rmax^2 ln(1 - (r/Rmax)^2), the force on particle 1 is
 * F = (dU/dr)(dr/r) = K/(1 - (r/Rmax)^2) * dr, i.e. restoring (points toward
 * particle 2) and diverging as r -> Rmax.
 */
inline Vec3 feneBondForce(const Vec3& dr, double K = kg::FENE_K,
                          double Rmax = kg::FENE_R0) noexcept {
    const double r = std::sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
    if (r <= 0.0 || r >= Rmax) return {0.0, 0.0, 0.0};
    const double c = (K * r / (1.0 - (r*r)/(Rmax*Rmax))) / r; // (dU/dr)/r
    return {c*dr[0], c*dr[1], c*dr[2]};
}

/**
 * @brief Harmonic bond force on particle 1, dr = r2 - r1.
 *
 * From U = 0.5 k (r - r0)^2, the force on particle 1 is
 * F = (dU/dr)(dr/r) = k (r - r0)/r * dr (restoring about r0).
 */
inline Vec3 harmonicBondForce(const Vec3& dr, double k = kg::BOND_K,
                              double r0 = kg::BOND_R0) noexcept {
    const double r = std::sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
    if (r < 1e-12) return {0.0, 0.0, 0.0};
    const double c = k * (r - r0) / r;
    return {c*dr[0], c*dr[1], c*dr[2]};
}

/// Bending forces on the three particles of an angle i-j-k (apex at j).
struct BendForces { Vec3 fi, fj, fk; };

/**
 * @brief Cosine bending forces for the triplet (ri, rj, rk), apex rj.
 *
 * Returns the analytic gradient of the cosine bending potential on all three
 * particles; fj = -(fi + fk) enforces momentum conservation. Zero beyond the
 * cutoff angle theta_c = pi / b.
 */
inline BendForces bendingForces(const Vec3& ri, const Vec3& rj, const Vec3& rk,
                                double a = kg::BEND_A, double b = kg::BEND_B) noexcept {
    BendForces out{{0,0,0}, {0,0,0}, {0,0,0}};
    const double theta_c = M_PI / b;

    const Vec3 r21{ri[0]-rj[0], ri[1]-rj[1], ri[2]-rj[2]};
    const Vec3 r23{rk[0]-rj[0], rk[1]-rj[1], rk[2]-rj[2]};
    const double n21 = std::sqrt(r21[0]*r21[0]+r21[1]*r21[1]+r21[2]*r21[2]);
    const double n23 = std::sqrt(r23[0]*r23[0]+r23[1]*r23[1]+r23[2]*r23[2]);
    if (n21 < 1e-12 || n23 < 1e-12) return out;

    double cosT = (r21[0]*r23[0] + r21[1]*r23[1] + r21[2]*r23[2]) / (n21*n23);
    cosT = std::clamp(cosT, -1.0, 1.0);
    const double theta = std::acos(cosT);
    if (theta >= theta_c) return out;

    const double sinT = std::sqrt(1.0 - cosT*cosT);
    if (sinT < 1e-8) return out;

    const double coeff = (-a * b * std::sin(2.0*b*theta)) / sinT;
    for (int d = 0; d < 3; ++d) {
        out.fi[d] = coeff * (r23[d]/(n21*n23) - cosT*r21[d]/(n21*n21));
        out.fk[d] = coeff * (r21[d]/(n21*n23) - cosT*r23[d]/(n23*n23));
        out.fj[d] = -out.fi[d] - out.fk[d];
    }
    return out;
}

// ── Symmetric 3x3 eigensolver (Cardano) ───────────────────────────────────────

/**
 * @brief Eigenvalues of a real symmetric 3x3 matrix, sorted descending.
 * @return {lambda1 >= lambda2 >= lambda3}
 */
inline Vec3 eigenvalues3(const double m[3][3]) noexcept {
    const double p1 = m[0][1]*m[0][1] + m[0][2]*m[0][2] + m[1][2]*m[1][2];
    if (p1 < 1e-30) {
        Vec3 ev{m[0][0], m[1][1], m[2][2]};
        std::sort(ev.begin(), ev.end(), [](double x, double y){ return x > y; });
        return ev;
    }
    const double q  = (m[0][0] + m[1][1] + m[2][2]) / 3.0;
    const double p2 = (m[0][0]-q)*(m[0][0]-q)
                    + (m[1][1]-q)*(m[1][1]-q)
                    + (m[2][2]-q)*(m[2][2]-q) + 2.0*p1;
    const double p  = std::sqrt(p2 / 6.0);
    double B[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            B[i][j] = (m[i][j] - q*(i==j)) / p;
    double r = ( B[0][0]*(B[1][1]*B[2][2]-B[1][2]*B[2][1])
               - B[0][1]*(B[1][0]*B[2][2]-B[1][2]*B[2][0])
               + B[0][2]*(B[1][0]*B[2][1]-B[1][1]*B[2][0]) ) / 2.0;
    r = std::clamp(r, -1.0, 1.0);
    const double phi = std::acos(r) / 3.0;
    return { q + 2.0*p*std::cos(phi),
             q + 2.0*p*std::cos(phi + 2.0*M_PI/3.0),
             q + 2.0*p*std::cos(phi + 4.0*M_PI/3.0) };
}

/**
 * @brief Unit eigenvector of a symmetric 3x3 matrix for eigenvalue @p lambda.
 *
 * Computed as the best null-space direction of (m - lambda I) via cross products
 * of its rows. The sign is arbitrary (it is a director, not an oriented vector).
 * Returns the zero vector for a degenerate/ill-conditioned matrix.
 */
inline Vec3 eigenvector3(const double m[3][3], double lambda) noexcept {
    const Vec3 r0{m[0][0]-lambda, m[0][1],        m[0][2]};
    const Vec3 r1{m[1][0],        m[1][1]-lambda, m[1][2]};
    const Vec3 r2{m[2][0],        m[2][1],        m[2][2]-lambda};

    auto cross = [](const Vec3& u, const Vec3& v) -> Vec3 {
        return { u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0] };
    };
    auto norm2 = [](const Vec3& v){ return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]; };

    const Vec3 c0 = cross(r0, r1), c1 = cross(r0, r2), c2 = cross(r1, r2);
    const double a = norm2(c0), b = norm2(c1), c = norm2(c2);

    Vec3 v = (a >= b && a >= c) ? c0 : (b >= c ? c1 : c2);
    const double n = std::sqrt(norm2(v));
    if (n < 1e-30) return {0.0, 0.0, 0.0};
    return {v[0]/n, v[1]/n, v[2]/n};
}

} // namespace md
