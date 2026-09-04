#pragma once

/**
 * @file entanglement.hpp
 * @brief Spatial entanglement density from backbone kink-angle analysis.
 *
 * @details
 * Identifies backbone "kinks" — bond angles θ below a threshold — and bins
 * them along the x-axis.  This is the standard post-processing step after a
 * Primitive Path Analysis (PPA): kinks in the primitive path correspond to
 * entanglement points.
 *
 * Two binning modes are available via EntanglementConfig::mode:
 *
 *   "chain_com"  (default)
 *     Each chain is assigned to exactly one bin based on its CoM x-position.
 *     All kinks of that chain accumulate in that bin.
 *     Avoids double-counting; gives avg_kinks_per_chain directly.
 *     Output: x_center  chains_in_bin  total_kinks  avg_kinks_per_chain
 *
 *   "monomer"
 *     Each kink is placed in the bin of the middle monomer p1.
 *     Chains crossing bin boundaries contribute to multiple bins.
 *     Also tracks how many distinct chains visit each bin.
 *     Output: x_center  kink_count  chains_in_bin  ent_per_chain
 *
 * Recommended thresholds:
 *   147.0°  genuine sharp entanglement kinks (Kremer-Grest, default)
 *   180.0°  count every non-straight bond (raw PPA output)
 *
 * MIC is applied on y,z; x is the gradient direction (no wrap).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "utility.hpp"

namespace md {

struct EntanglementConfig {
    std::string filenamePrefix;
    int    frameIndex       {0};
    int    Nm               {500};
    int    Nc               {100};
    int    N_s              {0};
    int    nBins            {100};
    double kinkThresholdDeg {147.0};
    std::string mode        {"chain_com"};
};

namespace detail {

/// Is the bond angle at the triplet (p0,p1,p2) a kink (cos(theta) < cosThresh)?
/// MIC on y,z only; x is the non-periodic gradient direction.
inline bool isKink(const std::vector<float>& rx, const std::vector<float>& ry,
                    const std::vector<float>& rz, int p0, int p1, int p2,
                    float Ly, float Lz, double cosThresh) noexcept {
    float b1x, b1y, b1z, b2x, b2y, b2z;
    minImageVecYZ(rx, ry, rz, p0, p1, Ly, Lz, b1x, b1y, b1z);
    minImageVecYZ(rx, ry, rz, p1, p2, Ly, Lz, b2x, b2y, b2z);
    const double r1 = std::sqrt(b1x*b1x+b1y*b1y+b1z*b1z);
    const double r2 = std::sqrt(b2x*b2x+b2y*b2y+b2z*b2z);
    if (r1==0.0 || r2==0.0) return false;
    const double cosA = (b1x*b2x+b1y*b2y+b1z*b2z)/(r1*r2);
    return cosA < cosThresh;
}

inline int countChainKinks(const std::vector<float>& rx,
                            const std::vector<float>& ry,
                            const std::vector<float>& rz,
                            int base, int Nm,
                            float Ly, float Lz,
                            double cosThresh) noexcept {
    int kinks = 0;
    for (int i = 1; i < Nm - 1; ++i)
        if (isKink(rx, ry, rz, base+i-1, base+i, base+i+1, Ly, Lz, cosThresh))
            ++kinks;
    return kinks;
}

} // namespace detail

// ── chain_com mode ────────────────────────────────────────────────────────────

inline void computeEntanglementChainCoM(const EntanglementConfig& cfg,
                                         const std::vector<float>& rx,
                                         const std::vector<float>& ry,
                                         const std::vector<float>& rz,
                                         int Nc, float Lx, float Ly, float Lz,
                                         const std::string& outputFile) {
    const double binWidth  = Lx / cfg.nBins;
    const double cosThresh = std::cos(cfg.kinkThresholdDeg * M_PI / 180.0);

    std::vector<int> counts(cfg.nBins, 0);
    std::vector<int> kinks (cfg.nBins, 0);

    for (int c = 0; c < Nc; ++c) {
        const int base = c * cfg.Nm;
        double xSum = 0.0;
        for (int i = 0; i < cfg.Nm; ++i) xSum += rx[base+i];
        const double xCoM = xSum / cfg.Nm;
        int bin = static_cast<int>(xCoM / binWidth);
        bin = std::max(0, std::min(bin, cfg.nBins-1));
        ++counts[bin];
        kinks[bin] += detail::countChainKinks(
            rx, ry, rz, base, cfg.Nm, Ly, Lz, cosThresh);
        printProgress("Entanglement", Nc, c);
    }
    std::cout << "\n100%\n";

    std::ofstream fo(outputFile);
    fo << "# x_center\t chains_in_bin\t total_kinks\t avg_kinks_per_chain\n";
    for (int i = 0; i < cfg.nBins; ++i) {
        const double avg = counts[i] > 0
                           ? static_cast<double>(kinks[i]) / counts[i] : 0.0;
        fo << (i+0.5)*binWidth << '\t' << counts[i] << '\t'
           << kinks[i] << '\t' << avg << '\n';
    }
}

// ── monomer mode ──────────────────────────────────────────────────────────────

inline void computeEntanglementMonomer(const EntanglementConfig& cfg,
                                        const std::vector<float>& rx,
                                        const std::vector<float>& ry,
                                        const std::vector<float>& rz,
                                        int Nc, float Lx, float Ly, float Lz,
                                        const std::string& outputFile) {
    const double binWidth  = Lx / cfg.nBins;
    const double cosThresh = std::cos(cfg.kinkThresholdDeg * M_PI / 180.0);

    std::vector<int>                     kinkCount(cfg.nBins, 0);
    std::vector<std::unordered_set<int>> chainsInBin(cfg.nBins);

    for (int c = 0; c < Nc; ++c) {
        const int base = c * cfg.Nm;
        for (int i = 0; i < cfg.Nm; ++i) {
            const int bin = std::min(
                static_cast<int>(rx[base+i] / binWidth), cfg.nBins-1);
            if (bin >= 0) chainsInBin[bin].insert(c);
        }
        for (int i = 1; i < cfg.Nm - 1; ++i) {
            const int p0=base+i-1, p1=base+i, p2=base+i+1;
            if (detail::isKink(rx, ry, rz, p0, p1, p2, Ly, Lz, cosThresh)) {
                const int bin = std::min(
                    static_cast<int>(rx[p1]/binWidth), cfg.nBins-1);
                if (bin >= 0) ++kinkCount[bin];
            }
        }
        printProgress("Entanglement", Nc, c);
    }
    std::cout << "\n100%\n";

    std::ofstream fo(outputFile);
    fo << "# x_center\t kink_count\t chains_in_bin\t ent_per_chain\n";
    for (int i = 0; i < cfg.nBins; ++i) {
        const int    cc  = static_cast<int>(chainsInBin[i].size());
        const double epc = cc > 0 ? static_cast<double>(kinkCount[i]) / cc : 0.0;
        fo << (i+0.5)*binWidth << '\t' << kinkCount[i]
           << '\t' << cc << '\t' << epc << '\n';
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

inline void computeEntanglement(const EntanglementConfig& cfg,
                                 const std::string& outputFile = "entanglement.dat") {
    std::cout << "===== Starting Entanglement Analysis"
              << " (mode=" << cfg.mode
              << ", threshold=" << cfg.kinkThresholdDeg << "°) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    Frame fr = loadFrame(cfg.filenamePrefix, cfg.frameIndex, cfg.Nm, cfg.Nc,
                         UnwrapPolicy::None);
    auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
    const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;
    const int Nc = (fr.N - cfg.N_s) / cfg.Nm;

    if (cfg.mode == "chain_com")
        computeEntanglementChainCoM(cfg, rx, ry, rz, Nc, Lx, Ly, Lz, outputFile);
    else if (cfg.mode == "monomer")
        computeEntanglementMonomer (cfg, rx, ry, rz, Nc, Lx, Ly, Lz, outputFile);
    else
        throw std::invalid_argument(
            "mode must be 'chain_com' or 'monomer', got: " + cfg.mode);

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time Entanglement: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
