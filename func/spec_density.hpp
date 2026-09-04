#pragma once

/**
 * @file spec_density.hpp
 * @brief Species-resolved radial density profile ρ(r) around the polymer CoM.
 *
 * @details
 * Computes the local volume fraction occupied by polymer monomers and solvent
 * particles in concentric shells around the polymer centre of mass:
 *
 *   ρ(r) = Σ V_particle(r) / V_shell(r)
 *
 * where V_particle is the hard-sphere volume of each species and V_shell the
 * volume of the spherical shell [r, r+Δr].
 *
 * Hard-sphere diameters used:
 *   σ_monomer = 1.00 σ
 *   σ_solvent = 0.75 σ
 *
 * Output file: three columns  r  rho_polymer  rho_solvent
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
 * @brief Configuration for a species density run.
 */
struct SpecDensityConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int    frameIndex  {0};     ///< Frame to analyse
    int    Nm          {10};    ///< Monomers per chain
    int    Nc          {100};   ///< Number of chains
    int    N_s         {0};     ///< Number of solvent particles
    int    nShells     {18};    ///< Number of radial shells
    double shellWidth  {1.0};   ///< Shell width [σ]
    double sigmaMonomer{1.00};  ///< Monomer hard-sphere diameter [σ]
    double sigmaSolvent{0.75};  ///< Solvent hard-sphere diameter [σ]
};

/**
 * @brief Compute species-resolved radial density profile.
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination file (default: "spec_density.dat").
 */
inline void computeSpeciesDensity(const SpecDensityConfig& cfg,
                                   const std::string& outputFile = "spec_density.dat") {
    std::cout << "===== Starting Species Radial Density rho(r) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    // ── Load frame, unwrapped as a contiguous cluster with its CoM ────────────
    // (per-chain unwrap + inter-chain imaging; see unwrapPolymerCluster).
    Frame fr = loadFrame(cfg.filenamePrefix, cfg.frameIndex, cfg.Nm, cfg.Nc,
                         UnwrapPolicy::ChainCluster,
                         cfg.N_s > 0 ? Species::Both : Species::Polymer);
    auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
    const int N = fr.N;
    const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;
    const double cx = fr.comX, cy = fr.comY, cz = fr.comZ;

    const int N_p = cfg.Nm * cfg.Nc; // polymer particles

    // Minimum-image distance from the CoM (all axes periodic: globule geometry).
    auto distFromCoM = [&](int i) {
        double dx = rx[i]-cx, dy = ry[i]-cy, dz = rz[i]-cz;
        dx -= Lx*std::round(dx/Lx);
        dy -= Ly*std::round(dy/Ly);
        dz -= Lz*std::round(dz/Lz);
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    };

    // ── Particle volumes ──────────────────────────────────────────────────────
    const double volMono = (4.0/3.0) * M_PI * std::pow(cfg.sigmaMonomer*0.5, 3);
    const double volSolv = (4.0/3.0) * M_PI * std::pow(cfg.sigmaSolvent*0.5, 3);

    std::vector<double> rhoPoly(cfg.nShells, 0.0);
    std::vector<double> rhoSolv(cfg.nShells, 0.0);

    // ── Bin monomers ──────────────────────────────────────────────────────────
    for (int i = 0; i < N_p && i < N; ++i) {
        const int s = static_cast<int>(distFromCoM(i) / cfg.shellWidth);
        if (s < cfg.nShells) rhoPoly[s] += volMono;
    }

    // ── Bin solvent (imaged to the CoM via minimum image) ─────────────────────
    for (int i = N_p; i < N_p + cfg.N_s && i < N; ++i) {
        const int s = static_cast<int>(distFromCoM(i) / cfg.shellWidth);
        if (s < cfg.nShells) rhoSolv[s] += volSolv;
    }

    // ── Normalise by shell volume ─────────────────────────────────────────────
    std::ofstream fo(outputFile);
    fo << "# r [sigma]\t rho_polymer\t rho_solvent\n";

    for (int s = 0; s < cfg.nShells; ++s) {
        const double rIn  = s       * cfg.shellWidth;
        const double rOut = (s + 1) * cfg.shellWidth;
        const double vShell = (4.0/3.0) * M_PI * (rOut*rOut*rOut - rIn*rIn*rIn);
        const double rCenter = (rIn + rOut) * 0.5;

        rhoPoly[s] /= vShell;
        rhoSolv[s] /= vShell;

        fo << rCenter << '\t' << rhoPoly[s] << '\t' << rhoSolv[s] << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time Species Density: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
