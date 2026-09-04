#pragma once

/**
 * Mean Squared Internal Distance (MSID): <R²(s)> as a function of
 *        the contour-length separation s along the polymer backbone.
 *
 * The result is normalised by s·lb² (lb = bond length) to produce the
 * characteristic ratio C(s) = <R²(s)> / (s·lb²).
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utility.hpp"

namespace md {

/**
 * @brief Configuration for an MSID run.
 */
struct MSIDConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int  frameStart;
    int  frameStop;
    int  frameStep;
    int  Nm;                    ///< Monomers per chain
    int  Nc;                    ///< Number of chains
    double bondLength = 0.964;  ///< Equilibrium bond length lb [σ]
};

/**
 * @brief Compute MSID over a range of trajectory frames.
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination file (default: "msid.dat").
 */
inline void computeMSID(const MSIDConfig& cfg,
                        const std::string& outputFile = "msid.dat") {
    std::cout << "===== Starting MSID =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    int frameCount = 0;

    std::vector<double> sumX(cfg.Nm, 0.0);
    std::vector<double> sumY(cfg.Nm, 0.0);
    std::vector<double> sumZ(cfg.Nm, 0.0);

    for (int smp = cfg.frameStart; smp <= cfg.frameStop; smp += cfg.frameStep) {
        // Unwrapped per chain (ChainOnly) so intra-chain distances are correct
        // across periodic boundaries (a wrapped chain would otherwise give
        // ±L jumps in R(s)).
        Frame fr = loadFrame(cfg.filenamePrefix, smp, cfg.Nm, cfg.Nc,
                             UnwrapPolicy::ChainOnly);
        auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;

        // Iterate exactly the polymer chains (cfg.Nc). Using fr.N/Nm here would
        // miscount when the frame also contains solvent, and would then disagree
        // with the cfg.Nc used in the normalisation below.
        for (int s = 0; s < cfg.Nm; ++s)
            for (int i = 0; i < cfg.Nc; ++i) {
                const int chainBase = i * cfg.Nm;
                for (int j = chainBase; j < chainBase + (cfg.Nm - s); ++j) {
                    const double dx = static_cast<double>(rx[j+s]) - rx[j];
                    const double dy = static_cast<double>(ry[j+s]) - ry[j];
                    const double dz = static_cast<double>(rz[j+s]) - rz[j];
                    sumX[s] += dx*dx; sumY[s] += dy*dy; sumZ[s] += dz*dz;
                }
            }

        ++frameCount;
        printProgress("MSID", cfg.frameStop + 1, smp);
    }
    std::cout << "\n100%\n";

    // ── Normalise and write ───────────────────────────────────────────────────
    std::ofstream fo(outputFile);
    fo << "# s\t C(s) = <R2(s)>/(s*lb^2)\n";

    // denom counts every (chain, pair-at-separation-s, frame) term summed above.
    for (int s = 1; s < cfg.Nm; ++s) {
        const double denom = static_cast<double>(cfg.Nc) * (cfg.Nm - s) * frameCount;
        const double r2    = (sumX[s] + sumY[s] + sumZ[s]) / denom;
        fo << s << '\t' << (r2 / (s * cfg.bondLength * cfg.bondLength)) << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time MSID: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
