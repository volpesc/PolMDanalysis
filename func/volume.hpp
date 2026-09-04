#pragma once

/**
 * @file volume.hpp
 * @brief Equivalent spherical volume from chain radius of gyration.
 *
 * @details
 * Computes the overall radius of gyration Rg of the polymer system and
 * the volume of the equivalent sphere:
 *
 *   Rg²   = (1/N) Σ_i |r_i − r_CoM|²
 *   V_eq  = (4/3) π Rg³
 *
 * Coordinates are unwrapped bond-by-bond before the CoM is computed so
 * the result is correct for chains that span periodic boundaries.
 *
 * Output file: two columns  Rg  V_eq
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
 * @brief Configuration for a volume run.
 */
struct VolumeConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int  frameIndex {0};        ///< Frame to analyse
    int  Nm         {10};       ///< Monomers per chain
    int  Nc         {100};      ///< Number of chains
};

/**
 * @brief Result from a volume computation.
 */
struct VolumeResult {
    double Rg  {}; ///< Radius of gyration [σ]
    double Veq {}; ///< Equivalent spherical volume [σ³]
};

/**
 * @brief Compute Rg and the equivalent spherical volume for one frame.
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination file (default: "volume.dat").
 * @return VolumeResult
 */
inline VolumeResult computeVolume(const VolumeConfig& cfg,
                                   const std::string& outputFile = "volume.dat") {
    std::cout << "===== Starting Gyration Volume =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    // ── Load frame, unwrapped as a contiguous cluster with its CoM ────────────
    // Each chain is unwrapped within itself (reset at chain boundaries) and the
    // chains are imaged together; walking bond-by-bond across chain boundaries
    // would spuriously link unrelated chains. Only the polymer region [0, N_p)
    // is used — solvent slots (if the frame has any) are left untouched.
    Frame fr = loadFrame(cfg.filenamePrefix, cfg.frameIndex, cfg.Nm, cfg.Nc,
                         UnwrapPolicy::ChainCluster);
    auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
    const int N = fr.N;
    const int N_p = std::min(cfg.Nm * cfg.Nc, N);
    const double cx = fr.comX, cy = fr.comY, cz = fr.comZ;

    // ── Rg² (true spread of the imaged cluster; direct distance) ──────────────
    double rg2 = 0.0;
    for (int i = 0; i < N_p; ++i) {
        rg2 += (rx[i]-cx)*(rx[i]-cx)
             + (ry[i]-cy)*(ry[i]-cy)
             + (rz[i]-cz)*(rz[i]-cz);
    }
    rg2 /= N_p;

    const double Rg  = std::sqrt(rg2);
    const double Veq = (4.0/3.0) * M_PI * Rg * Rg * Rg;

    // ── Output ────────────────────────────────────────────────────────────────
    std::ofstream fo(outputFile);
    fo << "# Rg [sigma]\t V_eq [sigma^3]\n";
    fo << Rg << '\t' << Veq << '\n';

    std::cout << "  Rg  = " << Rg  << " σ\n"
              << "  Veq = " << Veq << " σ³\n";

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time Volume: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";

    return {Rg, Veq};
}

} // namespace md
