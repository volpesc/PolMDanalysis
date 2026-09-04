#pragma once

/**
 * @file gds_diffusion.hpp
 * @brief Gibbs Dividing Surface (GDS) diffusion front analysis.
 *
 * @details
 * Tracks the diffusion front of polymer and solvent species across a series
 * of trajectory frames, computing:
 *
 *   Front position x_GDS(t)
 *     The Gibbs dividing surface: the x-position where the cumulative
 *     concentration profile reaches 50% of the total.  Equivalent to the
 *     median particle position along x.
 *
 *   Uptake(t)
 *     Total number of particles in the concentration profile (integrated
 *     density).
 *
 *   Polymer uptake count
 *     Number of polymer monomers that have crossed a user-defined threshold
 *     x_threshold into the solvent region.
 *
 *   Induction time t_ind
 *     The first frame at which the GDS velocity exceeds a threshold,
 *     indicating the onset of diffusion.
 *
 * Output file: CSV with columns
 *   time, solvent_front, polymer_front, solvent_uptake, polymer_uptake
 *
 * Usage note:
 *   This tool analyses a RANGE of frames (--start to --stop), not a single
 *   frame.  Each frame is read independently so memory use is O(1) in frames.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "utility.hpp"

namespace md {

/**
 * @brief Configuration for a GDS diffusion run.
 */
struct GDSConfig {
    std::string filenamePrefix;         ///< Trajectory file prefix
    int    frameStart         {0};      ///< First frame index
    int    frameStop          {100};    ///< Last  frame index
    int    Nm                 {500};    ///< Monomers per chain
    int    Nc                 {1000};   ///< Number of chains
    int    N_s                {0};      ///< Number of solvent particles
    double xMin               {0.0};   ///< Profile x range start [σ]
    double xMax               {100.0}; ///< Profile x range end [σ]
    double binWidth           {3.0};   ///< Concentration profile bin width [σ]
    double deltaT             {1000.0};///< Physical time between frames [τ]
    double solventRegionStart {45.0};  ///< x threshold for polymer uptake count [σ]
    double velocityThreshold  {0.01};  ///< GDS velocity threshold for induction [σ/τ]
};

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace detail {

/**
 * @brief Find the Gibbs dividing surface (median position) of a 1D profile.
 * @return x-position of the GDS, or -1 if not found.
 */
inline double gibbsDividingSurface(const std::vector<double>& profile,
                                    double binWidth, double xMin) noexcept {
    const double total = std::accumulate(profile.begin(), profile.end(), 0.0);
    if (total <= 0.0) return -1.0;
    const double half = total * 0.5;
    double cumulative = 0.0;
    for (size_t i = 0; i < profile.size(); ++i) {
        cumulative += profile[i];
        if (cumulative >= half) {
            const double excess = cumulative - half;
            const double frac   = (profile[i] > 0.0) ? excess / profile[i] : 0.0;
            return xMin + (static_cast<double>(i) + 1.0 - frac) * binWidth;
        }
    }
    return -1.0;
}

/**
 * @brief Find first frame index where GDS velocity exceeds threshold.
 * @return Frame index, or -1 if never triggered.
 */
inline int inductionFrame(const std::vector<double>& fronts,
                           double deltaT, double velocityThreshold) noexcept {
    for (size_t i = 1; i < fronts.size(); ++i) {
        if ((fronts[i] - fronts[i-1]) / deltaT > velocityThreshold)
            return static_cast<int>(i);
    }
    return -1;
}

} // namespace detail

/**
 * @brief Run GDS diffusion front analysis over a range of frames.
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination CSV file (default: "gds_diffusion.csv").
 */
inline void computeGDSDiffusion(const GDSConfig& cfg,
                                  const std::string& outputFile = "gds_diffusion.csv") {
    std::cout << "===== Starting GDS Diffusion Front Analysis =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    const int nBins = static_cast<int>((cfg.xMax - cfg.xMin) / cfg.binWidth);
    const int N_p   = cfg.Nm * cfg.Nc;

    std::vector<double> times;
    std::vector<double> solventFronts, polymerFronts;
    std::vector<double> solventUptakes, polymerUptakes;

    for (int frame = cfg.frameStart; frame <= cfg.frameStop; ++frame) {
        Frame fr;
        try {
            fr = loadFrame(cfg.filenamePrefix, frame, cfg.Nm, cfg.Nc, UnwrapPolicy::None,
                           cfg.N_s > 0 ? Species::Both : Species::Polymer);
        } catch (const std::exception& e) {
            std::cerr << "  Skipping frame " << frame << ": " << e.what() << "\n";
            continue;
        }
        auto& rx = fr.rx;
        const int N = fr.N;

        std::vector<double> polyProfile(nBins, 0.0);
        std::vector<double> solvProfile(nBins, 0.0);
        int polyUptakeCount = 0;

        // Polymer monomers
        for (int i = 0; i < N_p && i < N; ++i) {
            const int bin = static_cast<int>((rx[i] - cfg.xMin) / cfg.binWidth);
            if (bin >= 0 && bin < nBins) polyProfile[bin] += 1.0;
            if (rx[i] > cfg.solventRegionStart) ++polyUptakeCount;
        }
        // Solvent particles
        for (int i = N_p; i < N_p + cfg.N_s && i < N; ++i) {
            const int bin = static_cast<int>((rx[i] - cfg.xMin) / cfg.binWidth);
            if (bin >= 0 && bin < nBins) solvProfile[bin] += 1.0;
        }

        const double time = (frame - cfg.frameStart) * cfg.deltaT;
        times.push_back(time);
        solventFronts.push_back(
            detail::gibbsDividingSurface(solvProfile, cfg.binWidth, cfg.xMin));
        polymerFronts.push_back(
            detail::gibbsDividingSurface(polyProfile, cfg.binWidth, cfg.xMin));
        solventUptakes.push_back(
            std::accumulate(solvProfile.begin(), solvProfile.end(), 0.0));
        polymerUptakes.push_back(static_cast<double>(polyUptakeCount));

        printProgress("GDS", cfg.frameStop + 1, frame);
    }
    std::cout << "\n100%\n";

    // ── Write CSV ─────────────────────────────────────────────────────────────
    std::ofstream fo(outputFile);
    fo << "time,solvent_front,polymer_front,solvent_uptake,polymer_uptake\n";
    for (size_t i = 0; i < times.size(); ++i) {
        fo << times[i]         << ','
           << solventFronts[i] << ','
           << polymerFronts[i] << ','
           << solventUptakes[i] << ','
           << polymerUptakes[i] << '\n';
    }

    // ── Induction times ───────────────────────────────────────────────────────
    const int indS = detail::inductionFrame(solventFronts, cfg.deltaT, cfg.velocityThreshold);
    const int indP = detail::inductionFrame(polymerFronts, cfg.deltaT, cfg.velocityThreshold);
    if (indS >= 0)
        std::cout << "  Solvent induction time: t = " << times[indS] << " τ\n";
    else
        std::cout << "  No solvent induction regime detected.\n";
    if (indP >= 0)
        std::cout << "  Polymer induction time: t = " << times[indP] << " τ\n";
    else
        std::cout << "  No polymer induction regime detected.\n";

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time GDS: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
