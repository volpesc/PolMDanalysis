#pragma once

/**
 * @file backbone.hpp
 * @brief FENE bond-force profile along the backbone.
 *
 * @details
 * Computes the FENE (Finitely Extensible Non-linear Elastic) bond force
 * projected onto a user-defined stretch direction, as a function of bond
 * index along the chain:
 *
 *   F_FENE(r) = -K r / (1 - (r/R0)²)
 *   F_proj(i) = F_FENE(r_i) · (b̂_i · ê_stretch)
 *
 * where b̂_i is the unit bond vector for bond i and ê_stretch is the
 * normalised stretch direction (default: x̂ = [1,0,0]).
 *
 * Two modes are available via BackboneConfig::mode:
 *
 *   "full"   Chain-major loop over all chains and bonds.  Each bond index j
 *            accumulates contributions from every chain; the result is
 *            averaged per bond index.  Use for bulk equilibrium analysis.
 *
 *   "single" Read particles in ID order (one chain).  Stops at the first
 *            bond that exceeds R0.  Use for a single tagged chain or
 *            stretched-chain analysis.
 *
 * FENE parameters (from Kremer-Grest model):
 *   K  = 30 ε/σ²
 *   R0 = 1.5 σ
 *
 * Output file: columns  bond_index  F_proj
 */

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utility.hpp"
#include "forces.hpp"

namespace md {

// ── FENE model parameters (see func/forces.hpp for the shared force model) ────
static constexpr double FENE_K  = kg::FENE_K;  ///< Spring constant [ε/σ²]
static constexpr double FENE_R0 = kg::FENE_R0; ///< Maximum bond extension [σ]

/**
 * @brief Configuration for a backbone stress run.
 */
struct BackboneConfig {
    std::string filenamePrefix;         ///< Trajectory file prefix
    int    frameIndex        {0};       ///< Frame to analyse
    int    Nm                {10};      ///< Monomers per chain
    int    Nc                {100};     ///< Number of chains
    std::string mode         {"full"};  ///< "full" (all chains) or "single"
    std::array<double,3> stretchDir{1.0, 0.0, 0.0}; ///< Stretch direction
};

/**
 * @brief Compute the projected FENE bond-force profile along the backbone.
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination file (default: "backbone.dat").
 */
inline void computeBackbone(const BackboneConfig& cfg,
                             const std::string& outputFile = "backbone.dat") {
    std::cout << "===== Starting Backbone Bond-Force Profile =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    // ── Normalise stretch direction ───────────────────────────────────────────
    const auto& d = cfg.stretchDir;
    const double norm = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    const std::array<double,3> eStr{d[0]/norm, d[1]/norm, d[2]/norm};

    // ── Load frame ────────────────────────────────────────────────────────────
    Frame fr = loadFrame(cfg.filenamePrefix, cfg.frameIndex, cfg.Nm, cfg.Nc,
                         UnwrapPolicy::None);
    auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
    const int N = fr.N;
    const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;
    const int Nc = N / cfg.Nm;

    std::vector<double> Fproj(cfg.Nm - 1, 0.0);
    std::vector<int>    count (cfg.Nm - 1, 0);

    if (cfg.mode == "full") {
        // All chains, average per bond index
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < cfg.Nm - 1; ++j) {
                const int p0 = i*cfg.Nm + j;
                const int p1 = p0 + 1;
                float dx, dy, dz;
                minImageVec(rx, ry, rz, p0, p1, Lx, Ly, Lz, dx, dy, dz);
                const double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (r == 0.0 || r >= FENE_R0) continue;

                const Vec3 Fb = feneBondForce({double(dx), double(dy), double(dz)},
                                              FENE_K, FENE_R0);
                Fproj[j] += Fb[0]*eStr[0] + Fb[1]*eStr[1] + Fb[2]*eStr[2];
                ++count[j];
            }
        }
    } else {
        // Single-chain (ID order), stop at first broken bond
        for (int i = 0; i < N - 1; ++i) {
            float dx, dy, dz;
            minImageVec(rx, ry, rz, i, i+1, Lx, Ly, Lz, dx, dy, dz);
            const double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (r >= FENE_R0) break;

            const int j = i % (cfg.Nm - 1);
            const Vec3 Fb = feneBondForce({double(dx), double(dy), double(dz)},
                                          FENE_K, FENE_R0);
            Fproj[j] += Fb[0]*eStr[0] + Fb[1]*eStr[1] + Fb[2]*eStr[2];
            ++count[j];
        }
    }

    // ── Write output ──────────────────────────────────────────────────────────
    std::ofstream fo(outputFile);
    fo << "# bond_index\t F_proj [epsilon/sigma]\n";
    for (int j = 0; j < cfg.Nm - 1; ++j) {
        const double avg = count[j] > 0 ? Fproj[j] / count[j] : 0.0;
        fo << j << '\t' << avg << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time Backbone: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
