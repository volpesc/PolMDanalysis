#pragma once

/**
 * @file ppa.hpp
 * @brief Primitive Path Analysis (PPA).
 *
 * @details
 * Computes three key entanglement observables from a polymer configuration
 * by treating each chain as its own primitive path:
 *
 *   bpp   Average primitive-path bond length [σ].
 *         = mean distance between consecutive beads along all chains.
 *
 *   app   Primitive-path Kuhn length [σ].
 *         = <Re²> / <L_contour>  where Re is the end-to-end distance
 *         and L_contour the total contour length of each chain.
 *
 *   Ne    Entanglement length [monomers].
 *         = app / bpp.  Directly related to the plateau modulus via
 *         G_e ≈ ρ k_B T / Ne.
 *
 * Minimum image convention is applied to all bond distances.
 *
 * Output file: key=value pairs for easy parsing.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utility.hpp"

namespace md {

/**
 * @brief Configuration for a PPA run.
 */
struct PPAConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int  frameIndex {0};        ///< Frame to analyse
    int  Nm         {10};       ///< Monomers per chain
    int  Nc         {100};      ///< Number of chains
};

/**
 * @brief Results from a single PPA computation.
 */
struct PPAResult {
    double bpp {}; ///< Average bond length along primitive path [σ]
    double app {}; ///< Primitive-path Kuhn length [σ]
    double Ne  {}; ///< Entanglement length [monomers]
};

/**
 * @brief Perform Primitive Path Analysis on one configuration frame.
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination file (default: "ppa.dat").
 * @return PPAResult  Computed observables (also written to file).
 */
inline PPAResult computePPA(const PPAConfig& cfg,
                             const std::string& outputFile = "ppa.dat") {
    std::cout << "===== Starting Primitive Path Analysis (PPA) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    // ── Load frame ────────────────────────────────────────────────────────────
    Frame fr = loadFrame(cfg.filenamePrefix, cfg.frameIndex, cfg.Nm, cfg.Nc,
                         UnwrapPolicy::None);
    auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
    const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;
    const int Nc = fr.N / cfg.Nm;

    // ── bpp: average bond length ──────────────────────────────────────────────
    double sumBond = 0.0;
    long   nBonds  = 0;

    for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < cfg.Nm - 1; ++j) {
            const int p0 = i * cfg.Nm + j;
            const int p1 = p0 + 1;
            float dx, dy, dz;
            minImageVec(rx, ry, rz, p0, p1, Lx, Ly, Lz, dx, dy, dz);
            sumBond += std::sqrt(dx*dx + dy*dy + dz*dz);
            ++nBonds;
        }
    }
    const double bpp = nBonds > 0 ? sumBond / nBonds : 0.0;

    // ── app: <Re²> / <L_contour>  ─────────────────────────────────────────────
    double sumRe2 = 0.0, sumL = 0.0;

    for (int i = 0; i < Nc; ++i) {
        // End-to-end via bond-by-bond reconstruction (PBC-safe)
        double ex = 0.0, ey = 0.0, ez = 0.0;
        double contour = 0.0;

        for (int j = 0; j < cfg.Nm - 1; ++j) {
            const int p0 = i * cfg.Nm + j;
            const int p1 = p0 + 1;
            float dx, dy, dz;
            minImageVec(rx, ry, rz, p0, p1, Lx, Ly, Lz, dx, dy, dz);
            const double bl = std::sqrt(dx*dx + dy*dy + dz*dz);
            ex += dx; ey += dy; ez += dz;
            contour += bl;
        }
        sumRe2 += ex*ex + ey*ey + ez*ez;
        sumL   += contour;
    }

    const double app = sumL > 0.0 ? sumRe2 / sumL : 0.0;
    const double Ne  = bpp > 0.0  ? app / bpp      : 0.0;

    // ── Output ────────────────────────────────────────────────────────────────
    PPAResult result{bpp, app, Ne};

    std::ofstream fo(outputFile);
    fo << "# Primitive Path Analysis\n"
       << "# frame = " << cfg.frameIndex << "\n"
       << "# Nm = "    << cfg.Nm         << "  Nc = " << Nc << "\n"
       << "bpp = " << bpp << "\n"
       << "app = " << app << "\n"
       << "Ne  = " << Ne  << "\n";

    std::cout << "  bpp = " << bpp << " σ\n"
              << "  app = " << app << " σ\n"
              << "  Ne  = " << Ne  << " monomers\n";

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time PPA: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";

    return result;
}

} // namespace md
