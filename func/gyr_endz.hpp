#pragma once

/**
 * Spatial profile of the chain radius of gyration: <Rg²(x)>.
 *
 * For each spatial bin along x, collects chains whose center of mass falls
 * inside the bin and accumulates their per-direction squared gyration radius.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utility.hpp"

namespace md {

/**
 * @brief Compute the spatial gyration-radius profile and write it to a file.
 *
 * @param filenamePrefix  Trajectory file prefix.
 * @param frameIndex      Index of the frame to analyse.
 * @param outputFile      Destination file for the profile.
 * @param Nm              Monomers per chain.
 * @param Nc              Number of chains.
 * @param binWidth        Width of each spatial bin [σ].
 * @param nBins           Number of bins along x.
 */
inline void computeGyrationProfile(const std::string& filenamePrefix,
                                   int frameIndex,
                                   const std::string& outputFile,
                                   int Nm, int Nc,
                                   float binWidth = 3.0f,
                                   int   nBins    = 62) {
    std::cout << "===== Starting <Rg²(x)> =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    // ── Load frame ────────────────────────────────────────────────────────────
    Frame fr = loadFrame(filenamePrefix, frameIndex, Nm, Nc, UnwrapPolicy::None);
    auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
    const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;
    Nc = std::min(Nc, fr.N / Nm);  // cap to atoms present; ignore any solvent tail

    // ── Center of mass (PBC-aware) ────────────────────────────────────────────
    std::vector<float> cx, cy, cz;
    computeCoMPBC(Nm, Nc, rx, ry, rz, cx, cy, cz, Lx, Ly, Lz);

    // ── Accumulate Rg² per bin ────────────────────────────────────────────────
    std::vector<double> rgx(nBins, 0.0), rgy(nBins, 0.0), rgz(nBins, 0.0);
    std::vector<int>    binCount(nBins, 0);

    for (int l = 0; l < nBins; ++l) {
        const float x1 = l       * binWidth;
        const float x2 = (l + 1) * binWidth;

        for (int i = 0; i < Nc; ++i) {
            if (cx[i] < x1 || cx[i] >= x2) continue;

            ++binCount[l];
            double sx{}, sy{}, sz{};
            for (int j = 0; j < Nm; ++j) {
                const int pid = i*Nm + j;
                float dx = rx[pid] - cx[i];
                float dy = ry[pid] - cy[i];
                float dz = rz[pid] - cz[i];
                applyMinimumImage(dx, Lx);
                applyMinimumImage(dy, Ly);
                applyMinimumImage(dz, Lz);
                sx += dx*dx;  sy += dy*dy;  sz += dz*dz;
            }
            rgx[l] += sx / Nm;
            rgy[l] += sy / Nm;
            rgz[l] += sz / Nm;
        }

        // Average Rg² over the chains that fell in this bin: ⟨Rg²(x)⟩.
        if (binCount[l] > 0) {
            rgx[l] /= binCount[l];
            rgy[l] /= binCount[l];
            rgz[l] /= binCount[l];
        }
    }

    // ── Write output ──────────────────────────────────────────────────────────
    std::ofstream fo(outputFile);
    fo << "# x\t Rg2_x\t Rg2_y\t Rg2_z\t Rg2_total\n";
    for (int l = 0; l < nBins; ++l)
        fo << l * binWidth << '\t'
           << rgx[l] << '\t' << rgy[l] << '\t' << rgz[l] << '\t'
           << (rgx[l] + rgy[l] + rgz[l]) << '\n';

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time <Rg²(x)>: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
