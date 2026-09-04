#pragma once

/**
 * @file rdf.hpp
 * @brief Radial Distribution Function g(r).
 *
 * @details
 * Computes the radial distribution function g(r) between all particle pairs,
 * averaged over a range of trajectory frames:
 *
 *   g(r) = V / (N² · M) · Σ_frames Σ_{i≠j} δ(r - r_ij) / (4πr² Δr)
 *
 * where V is the box volume, N the number of particles, M the number of frames,
 * and Δr the bin width. In the ideal-gas limit g(r) → 1 at large r.
 *
 * Minimum image convention is applied to all pair distances so the function
 * is correct for periodic boundary conditions.
 *
 * Output file: two columns  r  g(r)
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
 * @brief Configuration for an RDF run.
 */
struct RDFConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int    frameStart   {0};
    int    frameStop    {100};
    int    frameStep    {1};
    int    Nm           {10};   ///< Monomers per chain
    int    Nc           {100};  ///< Number of chains
    int    nBins        {200};  ///< Number of histogram bins
    float  rCut         {5.0f}; ///< Maximum distance [σ]
};

/**
 * @brief Accumulate one frame's pair counts into @p hist (pure kernel, OpenMP).
 *
 * Separated from I/O for direct unit testing. The all-pairs loop is parallelised
 * with a per-thread histogram reduction (array section on the underlying
 * buffer), so threads accumulate without racing.
 */
inline void accumulateRDF(const std::vector<float>& rx,
                          const std::vector<float>& ry,
                          const std::vector<float>& rz,
                          int N, float Lx, float Ly, float Lz,
                          float rCut, int nBins, float dr,
                          std::vector<double>& hist) {
    const int   nb = nBins;
    double* const h = hist.data();
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 256) reduction(+:h[:nb])
#endif
    for (int i = 0; i < N - 1; ++i) {
        for (int j = i + 1; j < N; ++j) {
            float dx = rx[j] - rx[i];
            float dy = ry[j] - ry[i];
            float dz = rz[j] - rz[i];
            applyMinimumImage(dx, Lx);
            applyMinimumImage(dy, Ly);
            applyMinimumImage(dz, Lz);
            const float r = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (r < rCut) {
                const int bin = static_cast<int>(r / dr);
                if (bin < nb) h[bin] += 2.0; // i→j and j→i
            }
        }
    }
}

/**
 * @brief Compute g(r) averaged over a trajectory range.
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination file (default: "rdf.dat").
 */
inline void computeRDF(const RDFConfig& cfg,
                       const std::string& outputFile = "rdf.dat") {
    std::cout << "===== Starting RDF g(r) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    const float dr = cfg.rCut / cfg.nBins;
    std::vector<double> hist(cfg.nBins, 0.0);

    int    frameCount = 0;
    double sumVolume  = 0.0;
    long   sumN       = 0;

    for (int smp = cfg.frameStart; smp <= cfg.frameStop; smp += cfg.frameStep) {
        Frame fr = loadFrame(cfg.filenamePrefix, smp, cfg.Nm, cfg.Nc, UnwrapPolicy::None);
        auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
        const int N = fr.N;
        const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;

        sumVolume += static_cast<double>(Lx) * Ly * Lz;
        sumN      += N;

        accumulateRDF(rx, ry, rz, N, Lx, Ly, Lz, cfg.rCut, cfg.nBins, dr, hist);

        ++frameCount;
        printProgress("RDF", cfg.frameStop + 1, smp);
    }
    std::cout << "\n100%\n";

    // ── Normalise ─────────────────────────────────────────────────────────────
    const double avgV   = sumVolume / frameCount;
    const double avgN   = static_cast<double>(sumN) / frameCount;
    const double rho    = avgN / avgV;
    const double norm   = (4.0 / 3.0) * M_PI * rho * avgN * frameCount;

    std::ofstream fo(outputFile);
    fo << "# r [sigma]\t g(r)\n";
    for (int b = 0; b < cfg.nBins; ++b) {
        const double rLo = b       * dr;
        const double rHi = (b + 1) * dr;
        const double shellVol = (rHi*rHi*rHi - rLo*rLo*rLo); // proportional to 4/3 π Δr³
        const double g = (shellVol > 0.0) ? hist[b] / (norm * shellVol) : 0.0;
        fo << (b + 0.5) * dr << '\t' << g << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time RDF: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd - wallStart).count()
              << " ms\n";
}

} // namespace md
