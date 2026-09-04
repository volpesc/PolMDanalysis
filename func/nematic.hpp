#pragma once

/**
 * @file nematic.hpp
 * @brief Nematic order parameter S(x) and director field n̂(x).
 *
 * @details
 * Computes the nematic order parameter S and the nematic director n̂ as a
 * function of position x, by building the Q-tensor from bond vectors binned
 * along x:
 *
 *   Q_αβ = (1/N_bonds) Σ_i [ (3/2) û_α û_β − (1/2) δ_αβ ]
 *
 * where û_i is the unit bond vector of bond i.  The nematic order parameter
 * S is the largest eigenvalue of Q, computed analytically via Cardano's
 * formula for the real symmetric 3×3 case.  The director n̂ (eigenvector
 * corresponding to S) is found by power iteration.
 *
 *   S = 0      → isotropic (randomly oriented bonds)
 *   S = 1      → perfectly aligned bonds
 *   S ~ 0.5    → weakly nematic
 *
 * MIC is applied on y,z bond components; x is the gradient direction.
 * Chain end-bonds (last bond of each chain) are excluded.
 *
 * Output file: columns  x  S  nx  ny  nz
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utility.hpp"

namespace md {

/**
 * @brief Configuration for a nematic order parameter run.
 */
struct NematicConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int    frameIndex  {0};     ///< Frame to analyse
    int    Nm          {500};   ///< Monomers per chain
    int    Nc          {1000};  ///< Number of chains
    int    N_s         {0};     ///< Solvent particles to skip
    int    nBins       {85};    ///< Spatial bins along x
};

// ── Q-tensor (3×3 symmetric, stored as full matrix) ──────────────────────────

struct QTensor {
    double m[3][3] = {};

    void accumulate(double ux, double uy, double uz) noexcept {
        m[0][0] += 1.5*ux*ux - 0.5;  m[0][1] += 1.5*ux*uy;         m[0][2] += 1.5*ux*uz;
        m[1][0] += 1.5*uy*ux;         m[1][1] += 1.5*uy*uy - 0.5;  m[1][2] += 1.5*uy*uz;
        m[2][0] += 1.5*uz*ux;         m[2][1] += 1.5*uz*uy;         m[2][2] += 1.5*uz*uz - 0.5;
    }

    void normalise(int n) noexcept {
        for (auto& row : m) for (auto& v : row) v /= n;
    }

    /** Largest eigenvalue via Cardano's formula (real symmetric 3×3). */
    double maxEigenvalue() const noexcept {
        const double p1 = m[0][1]*m[0][1] + m[0][2]*m[0][2] + m[1][2]*m[1][2];
        if (p1 == 0.0)
            return std::max({m[0][0], m[1][1], m[2][2]});

        const double q  = (m[0][0] + m[1][1] + m[2][2]) / 3.0;
        const double p2 = (m[0][0]-q)*(m[0][0]-q)
                        + (m[1][1]-q)*(m[1][1]-q)
                        + (m[2][2]-q)*(m[2][2]-q) + 2.0*p1;
        const double p  = std::sqrt(p2 / 6.0);

        // B = (Q - q I) / p
        double B[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                B[i][j] = (m[i][j] - q*(i==j)) / p;

        double r = ( B[0][0]*(B[1][1]*B[2][2]-B[1][2]*B[2][1])
                   - B[0][1]*(B[1][0]*B[2][2]-B[1][2]*B[2][0])
                   + B[0][2]*(B[1][0]*B[2][1]-B[1][1]*B[2][0]) ) / 2.0;
        r = std::max(-1.0, std::min(1.0, r));

        const double phi = std::acos(r) / 3.0;
        return q + 2.0*p*std::cos(phi);
    }

    /** Principal eigenvector via power iteration. */
    std::array<double,3> director(int maxIter = 100, double tol = 1e-6) const noexcept {
        std::array<double,3> v{1.0, 0.0, 0.0};
        for (int it = 0; it < maxIter; ++it) {
            std::array<double,3> Mv{
                m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],
                m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],
                m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]
            };
            const double norm = std::sqrt(Mv[0]*Mv[0]+Mv[1]*Mv[1]+Mv[2]*Mv[2]);
            if (norm == 0.0) break;
            Mv[0]/=norm; Mv[1]/=norm; Mv[2]/=norm;
            if (std::abs(Mv[0]-v[0]) < tol &&
                std::abs(Mv[1]-v[1]) < tol &&
                std::abs(Mv[2]-v[2]) < tol) { v = Mv; break; }
            v = Mv;
        }
        return v;
    }
};

/**
 * @brief Compute nematic order parameter S(x) and director n̂(x).
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination file (default: "nematic.dat").
 */
inline void computeNematic(const NematicConfig& cfg,
                            const std::string& outputFile = "nematic.dat") {
    std::cout << "===== Starting Nematic Order Parameter S(x) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    // ── Load frame ────────────────────────────────────────────────────────────
    Frame fr = loadFrame(cfg.filenamePrefix, cfg.frameIndex, cfg.Nm, cfg.Nc,
                         UnwrapPolicy::None);
    auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
    const int N = fr.N;
    const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;

    const double dx = Lx / cfg.nBins;
    std::vector<QTensor> bins(cfg.nBins);
    std::vector<int>     counts(cfg.nBins, 0);
    const int N_p = cfg.Nm * cfg.Nc; // polymer particles only

    // ── Accumulate bond vectors into bins ─────────────────────────────────────
    for (int i = 0; i < N_p - 1 && i < N - 1; ++i) {
        // Skip last bond of each chain to avoid chain-end artefacts
        if ((i + 1) % cfg.Nm == 0) continue;

        float bx, by, bz;
        minImageVecYZ(rx, ry, rz, i, i+1, Ly, Lz, bx, by, bz);
        const double r = std::sqrt(bx*bx+by*by+bz*bz);
        if (r == 0.0) continue;

        const double ux = bx/r, uy = by/r, uz = bz/r;
        const double xCenter = 0.5*(rx[i]+rx[i+1]);
        const int bin = std::min(static_cast<int>(xCenter/dx), cfg.nBins-1);
        if (bin < 0) continue;

        bins[bin].accumulate(ux, uy, uz);
        ++counts[bin];
    }

    // ── Compute S and director per bin ────────────────────────────────────────
    std::ofstream fo(outputFile);
    fo << "# x\t S\t nx\t ny\t nz\n";

    for (int b = 0; b < cfg.nBins; ++b) {
        if (counts[b] < 2) {
            fo << b*dx << "\t0\t0\t0\t0\n";
            continue;
        }
        bins[b].normalise(counts[b]);
        const double S             = bins[b].maxEigenvalue();
        const auto   n             = bins[b].director();
        fo << b*dx << '\t' << S << '\t'
           << n[0] << '\t' << n[1] << '\t' << n[2] << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time Nematic: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
