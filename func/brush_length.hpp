#pragma once

/**
 * @file brush_length.hpp
 * @brief Brush / penetration length distribution P(L).
 *
 * @details
 * Analyses how far polymer chains extend beyond the solvent front in a
 * symmetric brush geometry.  For each chain anchored near the box centre,
 * the penetration length L is defined as the maximum distance any monomer
 * reaches beyond the solvent front on the corresponding side:
 *
 *   L = max over j { |r_j,x − x_anchor| − |x_front − x_anchor| }
 *       subject to r_j being on the solvent side of the front
 *
 * The solvent front positions (left and right) are estimated from the
 * percentile of solvent x-coordinates, defaulting to the 95th percentile.
 *
 * Chains are considered "anchored near the centre" when their first monomer
 * lies within --centrebuffer σ of Lx/2.
 *
 * Output file: two columns  L  P(L)
 *   P(L) is normalised by the number of valid chains so Σ P(L) ΔL ≈ 1.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "utility.hpp"

namespace md {

/**
 * @brief Configuration for a brush length run.
 */
struct BrushLengthConfig {
    std::string filenamePrefix;       ///< Trajectory file prefix
    int    frameIndex     {0};        ///< Frame to analyse
    int    Nm             {500};      ///< Monomers per chain
    int    Nc             {1000};     ///< Number of chains
    int    N_s            {0};        ///< Number of solvent particles
    double frontPercentile{95.0};     ///< Percentile for solvent front detection
    double centreBuffer   {10.0};     ///< Half-width of centre anchor zone [σ]
    double binWidth       {1.0};      ///< Histogram bin width [σ]
    int    minBin         {2};        ///< First histogram bin to output
    int    maxBin         {90};       ///< Last  histogram bin to output
};

/**
 * @brief Compute P(L) for brush chain penetration lengths.
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination file (default: "brush_length.dat").
 */
inline void computeBrushLength(const BrushLengthConfig& cfg,
                                 const std::string& outputFile = "brush_length.dat") {
    std::cout << "===== Starting Brush Penetration Length P(L) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    // ── Load frame ────────────────────────────────────────────────────────────
    Frame fr = loadFrame(cfg.filenamePrefix, cfg.frameIndex, cfg.Nm, cfg.Nc,
                         UnwrapPolicy::None,
                         cfg.N_s > 0 ? Species::Both : Species::Polymer);
    auto& rx = fr.rx;
    const int N = fr.N;
    const float Lx = fr.Lx;

    const double centreX = Lx * 0.5;
    const int    N_p     = cfg.Nm * cfg.Nc;

    // ── Find solvent front positions (percentile) ─────────────────────────────
    std::vector<double> leftX, rightX;
    for (int i = N_p; i < N_p + cfg.N_s && i < N; ++i) {
        if (rx[i] < centreX) leftX.push_back(rx[i]);
        else                  rightX.push_back(rx[i]);
    }

    double leftFront = 0.0, rightFront = Lx;
    if (!leftX.empty()) {
        std::sort(leftX.begin(), leftX.end());
        const int idx = static_cast<int>(cfg.frontPercentile / 100.0 * leftX.size());
        leftFront  = leftX[std::min(idx, (int)leftX.size()-1)];
    }
    if (!rightX.empty()) {
        std::sort(rightX.begin(), rightX.end());
        const int idx = static_cast<int>((100.0 - cfg.frontPercentile) / 100.0 * rightX.size());
        rightFront = rightX[std::max(0, std::min(idx, (int)rightX.size()-1))];
    }

    std::cout << "  Left solvent front:  x = " << leftFront  << " σ\n"
              << "  Right solvent front: x = " << rightFront << " σ\n";

    // ── Compute penetration length per chain ──────────────────────────────────
    std::map<int,int> hist;
    int validChains = 0;

    for (int c = 0; c < cfg.Nc; ++c) {
        const int base   = c * cfg.Nm;
        const double xAnchor = rx[base]; // anchor = first monomer

        if (std::abs(xAnchor - centreX) > cfg.centreBuffer) continue;

        double maxLen = 0.0;
        for (int j = 0; j < cfg.Nm; ++j) {
            const double xj = rx[base + j];
            double len = 0.0;
            if (xj < leftFront)        len = xAnchor - xj;         // penetrates left
            else if (xj > rightFront)  len = xj      - xAnchor;    // penetrates right
            if (len > maxLen) maxLen = len;
        }

        if (maxLen > 0.0) {
            const int bin = static_cast<int>(maxLen / cfg.binWidth);
            hist[bin]++;
            ++validChains;
        }
    }

    std::cout << "  Valid centre-anchored chains: " << validChains << "\n";

    // ── Write output ──────────────────────────────────────────────────────────
    std::ofstream fo(outputFile);
    fo << "# L [sigma]\t P(L)\n";
    for (int b = cfg.minBin; b <= cfg.maxBin; ++b) {
        const double prob = (validChains > 0 && hist.count(b))
                            ? static_cast<double>(hist.at(b)) / validChains : 0.0;
        fo << (b + 0.5) * cfg.binWidth << '\t' << prob << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time Brush Length: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
