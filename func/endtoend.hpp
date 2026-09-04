#pragma once

/**
 * @file endtoend.hpp
 * @brief End-to-end distance distribution and mean values.
 *
 * @details
 * Computes two observables from the end-to-end vector Re = r(Nm-1) - r(0):
 *
 *   <Re²>   Mean squared end-to-end distance, reported per frame so the
 *           time evolution can be tracked.
 *
 *   P(Re²)  Histogram of Re² values across all chains and frames, normalised
 *           to a probability distribution. For a Gaussian chain this should
 *           follow P(r) ∝ r² exp(-3r²/2<Re²>).
 *
 * Minimum image convention is applied to each backbone bond before
 * accumulating the end-to-end vector.
 *
 * Output files:
 *   <out>_time.dat   columns: frame  <Re²>  sqrt(<Re²>)   (RMS end-to-end)
 *   <out>_hist.dat   columns: Re²    P(Re²)
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
 * @brief Configuration for an end-to-end distance run.
 */
struct EndToEndConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int    frameStart   {0};
    int    frameStop    {100};
    int    frameStep    {1};
    int    Nm           {10};   ///< Monomers per chain
    int    Nc           {100};  ///< Number of chains
    int    nBins        {100};  ///< Histogram bins for P(Re²)
};

/**
 * @brief Compute end-to-end distance statistics over a trajectory.
 *
 * @param cfg        Run parameters.
 * @param outputStem File stem; produces <stem>_time.dat and <stem>_hist.dat.
 */
inline void computeEndToEnd(const EndToEndConfig& cfg,
                             const std::string& outputStem = "endtoend") {
    std::cout << "===== Starting End-to-End Distance <Re²> =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    std::vector<double> re2perFrame;
    std::vector<double> allRe2; // for histogram

    for (int smp = cfg.frameStart; smp <= cfg.frameStop; smp += cfg.frameStep) {
        Frame fr = loadFrame(cfg.filenamePrefix, smp, cfg.Nm, cfg.Nc, UnwrapPolicy::None);
        auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
        const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;

        // Iterate exactly the polymer chains. Using fr.N/Nm would miscount when
        // the frame also contains solvent (fr.N is the total atom count).
        double frameSum = 0.0;

        for (int i = 0; i < cfg.Nc; ++i) {
            // Reconstruct end-to-end vector bond-by-bond (PBC-safe)
            double ex = 0.0, ey = 0.0, ez = 0.0;
            for (int j = 0; j < cfg.Nm - 1; ++j) {
                const int pid = i * cfg.Nm + j;
                float dx, dy, dz;
                minImageVec(rx, ry, rz, pid, pid+1, Lx, Ly, Lz, dx, dy, dz);
                ex += dx; ey += dy; ez += dz;
            }
            const double re2 = ex*ex + ey*ey + ez*ez;
            frameSum += re2;
            allRe2.push_back(re2);
        }

        re2perFrame.push_back(frameSum / cfg.Nc);
        printProgress("EndToEnd", cfg.frameStop + 1, smp);
    }
    std::cout << "\n100%\n";

    // ── Time series output ────────────────────────────────────────────────────
    {
        std::ofstream fo(outputStem + "_time.dat");
        fo << "# frame\t <Re2>\t sqrt(<Re2>)\n";
        int idx = 0;
        for (int smp = cfg.frameStart; smp <= cfg.frameStop; smp += cfg.frameStep) {
            fo << smp << '\t'
               << re2perFrame[idx] << '\t'
               << std::sqrt(re2perFrame[idx]) << '\n';
            ++idx;
        }
    }

    // ── Histogram output ─────────────────────────────────────────────────────
    {
        const double re2max = *std::max_element(allRe2.begin(), allRe2.end());
        const double dBin   = re2max / cfg.nBins;
        std::vector<double> hist(cfg.nBins, 0.0);
        for (double v : allRe2) {
            const int bin = static_cast<int>(v / dBin);
            if (bin >= 0 && bin < cfg.nBins) hist[bin] += 1.0;
        }
        const double norm = static_cast<double>(allRe2.size()) * dBin;

        std::ofstream fo(outputStem + "_hist.dat");
        fo << "# Re2\t P(Re2)\n";
        for (int b = 0; b < cfg.nBins; ++b)
            fo << (b + 0.5) * dBin << '\t' << hist[b] / norm << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time End-to-End: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd - wallStart).count()
              << " ms\n";
}

} // namespace md
