#pragma once

/**
 * @file density.hpp
 * @brief Spatial density profile ρ(x).
 *
 * @details
 * Computes the number density ρ(x) as a function of position along the x-axis,
 * averaged over a range of trajectory frames:
 *
 *   ρ(x) = <N(x)> / (A · Δx)
 *
 * where <N(x)> is the mean number of particles in the slab [x, x+Δx],
 * A = Ly · Lz is the cross-sectional area, and Δx the bin width.
 *
 * Separate profiles are computed for polymer and solvent particles when
 * N_s > 0, enabling visualisation of layering and depletion effects.
 *
 * Output file: columns  x  rho_polymer  [rho_solvent]
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utility.hpp"

namespace md {

/**
 * @brief Configuration for a density profile run.
 */
struct DensityConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int    frameStart   {0};
    int    frameStop    {100};
    int    frameStep    {1};
    int    Nm           {10};   ///< Monomers per chain
    int    Nc           {100};  ///< Number of chains
    int    N_s          {0};    ///< Number of solvent particles (0 = none)
    int    nBins        {100};  ///< Number of bins along x
};

/**
 * @brief Compute ρ(x) averaged over a trajectory range.
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination file (default: "density.dat").
 */
inline void computeDensity(const DensityConfig& cfg,
                            const std::string& outputFile = "density.dat") {
    std::cout << "===== Starting Density Profile rho(x) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    const int N_p = cfg.Nm * cfg.Nc;

    std::vector<double> histPolymer(cfg.nBins, 0.0);
    std::vector<double> histSolvent(cfg.nBins, 0.0);

    int    frameCount = 0;
    double avgLx = 0.0, avgLy = 0.0, avgLz = 0.0;

    for (int smp = cfg.frameStart; smp <= cfg.frameStop; smp += cfg.frameStep) {
        Frame fr = loadFrame(cfg.filenamePrefix, smp, cfg.Nm, cfg.Nc, UnwrapPolicy::None,
                             cfg.N_s > 0 ? Species::Both : Species::Polymer);
        auto& rx = fr.rx;
        const int N = fr.N;
        const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;

        avgLx += Lx; avgLy += Ly; avgLz += Lz;
        const float dx = Lx / cfg.nBins;

        // Polymer particles
        for (int i = 0; i < N_p && i < N; ++i) {
            // Map x into [0, Lx) with periodic wrap
            float x = rx[i] - Lx * std::floor(rx[i] / Lx);
            const int bin = static_cast<int>(x / dx);
            if (bin >= 0 && bin < cfg.nBins) histPolymer[bin] += 1.0;
        }

        // Solvent particles
        for (int i = N_p; i < N_p + cfg.N_s && i < N; ++i) {
            float x = rx[i] - Lx * std::floor(rx[i] / Lx);
            const int bin = static_cast<int>(x / dx);
            if (bin >= 0 && bin < cfg.nBins) histSolvent[bin] += 1.0;
        }

        ++frameCount;
        printProgress("Density", cfg.frameStop + 1, smp);
    }
    std::cout << "\n100%\n";

    // ── Normalise ─────────────────────────────────────────────────────────────
    avgLx /= frameCount; avgLy /= frameCount; avgLz /= frameCount;
    const double binWidth = avgLx / cfg.nBins;
    const double A        = avgLy * avgLz;
    const double norm     = A * binWidth * frameCount;

    std::ofstream fo(outputFile);
    if (cfg.N_s > 0)
        fo << "# x [sigma]\t rho_polymer\t rho_solvent\n";
    else
        fo << "# x [sigma]\t rho_polymer\n";

    for (int b = 0; b < cfg.nBins; ++b) {
        fo << (b + 0.5) * binWidth << '\t' << histPolymer[b] / norm;
        if (cfg.N_s > 0) fo << '\t' << histSolvent[b] / norm;
        fo << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time Density: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd - wallStart).count()
              << " ms\n";
}

} // namespace md
