#pragma once

/**
 * @file energy.hpp
 * @brief Polymer–solvent interaction energy.
 *
 * @details
 * Computes the total non-bonded interaction energy between all polymer
 * monomers and all solvent particles for a given coupling parameter λ.
 *
 * The pair potential consists of two parts:
 *
 *   WCA repulsion (r < r_cut):
 *     U_WCA(r) = 4ε [(σ/r)¹² − (σ/r)⁶]
 *     r_cut = 2^(1/6) σ
 *
 *   Cosine-attractive tail (r_cut ≤ r < √2 r_cut):
 *     U_att(r) = α [cos(π (r/r_cut)²) − 1]
 *     α = 0.5145 λ
 *
 * where λ ∈ [0,1] controls the attractive strength (λ=0: purely repulsive,
 * λ=1: full attraction).
 *
 * Output: printed to stdout as  lambda  E_total
 *         and optionally written to a file when --out is provided.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utility.hpp"

namespace md {

// ── Pair potential functions ──────────────────────────────────────────────────

/**
 * @brief WCA (shifted Lennard-Jones) pair energy.
 */
inline double wcaPotential(double r, double epsilon, double sigma) noexcept {
    const double s  = sigma / r;
    const double s6 = s*s*s * s*s*s;
    return 4.0 * epsilon * (s6*s6 - s6);
}

/**
 * @brief Cosine-attractive tail energy.
 */
inline double attractivePotential(double r, double alpha, double rCut) noexcept {
    return alpha * (std::cos(M_PI * (r/rCut) * (r/rCut)) - 1.0);
}

/**
 * @brief Total pair energy between two particles.
 */
inline double pairEnergy(double r, double epsilon, double sigma,
                          double alpha, double rCut) noexcept {
    if (r <= 0.0)         return 0.0;
    if (r < rCut)         return wcaPotential(r, epsilon, sigma);
    const double rCutA = rCut * std::sqrt(2.0);
    if (r < rCutA)        return attractivePotential(r, alpha, rCut);
    return 0.0;
}

// ── Configuration ─────────────────────────────────────────────────────────────

/**
 * @brief Configuration for an energy run.
 */
struct EnergyConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int    frameIndex  {0};     ///< Frame to analyse
    int    Nm          {10};    ///< Monomers per chain
    int    Nc          {100};   ///< Number of chains
    int    N_s         {0};     ///< Number of solvent particles
    double lambda      {1.0};   ///< Attractive coupling  λ ∈ [0,1]
    double epsilon     {1.0};   ///< LJ energy scale [ε]
    double sigma       {1.0};   ///< LJ length scale [σ]
};

/**
 * @brief Total polymer–solvent pair energy (pure kernel, OpenMP-parallel).
 *
 * Separated from I/O so it can be unit-tested directly. Each thread accumulates
 * into a private partial sum (reduction), so the loop is race-free.
 */
inline double polymerSolventEnergy(const std::vector<float>& rx,
                                   const std::vector<float>& ry,
                                   const std::vector<float>& rz,
                                   int N, int N_p, int N_s,
                                   float Lx, float Ly, float Lz,
                                   double epsilon, double sigma,
                                   double alpha, double rCut) {
    double totalEnergy = 0.0;
    const int iMax = (N_p < N) ? N_p : N;   // canonical bound for OpenMP
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:totalEnergy)
#endif
    for (int i = 0; i < iMax; ++i) {
        for (int j = N_p; j < N_p + N_s && j < N; ++j) {
            float dx = rx[j]-rx[i], dy = ry[j]-ry[i], dz = rz[j]-rz[i];
            applyMinimumImage(dx, Lx);
            applyMinimumImage(dy, Ly);
            applyMinimumImage(dz, Lz);
            const double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            totalEnergy += pairEnergy(r, epsilon, sigma, alpha, rCut);
        }
    }
    return totalEnergy;
}

/**
 * @brief Compute total polymer–solvent interaction energy.
 *
 * @param cfg        Run parameters.
 * @param outputFile Optional output file path (empty = stdout only).
 * @return Total interaction energy [ε].
 */
inline double computeEnergy(const EnergyConfig& cfg,
                              const std::string& outputFile = "") {
    std::cout << "===== Starting Interaction Energy =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    // ── Load frame ────────────────────────────────────────────────────────────
    const std::string fn = makeInputFilename(cfg.filenamePrefix, cfg.frameIndex);
    int N; float Lx{}, Ly{}, Lz{};
    std::vector<float> rx, ry, rz;
    readFrame(fn, cfg.Nm, cfg.Nc, rx, ry, rz, N, Lx, Ly, Lz,
              cfg.N_s > 0 ? Species::Both : Species::Polymer);

    const int N_p   = cfg.Nm * cfg.Nc;
    const double alpha = 0.5145 * cfg.lambda;
    const double rCut  = std::pow(2.0, 1.0/6.0) * cfg.sigma;

    // ── All polymer–solvent pairs (parallel kernel) ───────────────────────────
    const double totalEnergy = polymerSolventEnergy(
        rx, ry, rz, N, N_p, cfg.N_s, Lx, Ly, Lz,
        cfg.epsilon, cfg.sigma, alpha, rCut);

    // ── Output ────────────────────────────────────────────────────────────────
    std::cout << "  λ = " << cfg.lambda << "  E = " << totalEnergy << " ε\n";

    if (!outputFile.empty()) {
        std::ofstream fo(outputFile, std::ios::app);
        fo << cfg.lambda << '\t' << totalEnergy << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time Energy: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";

    return totalEnergy;
}

} // namespace md
