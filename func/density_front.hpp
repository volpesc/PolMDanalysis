#pragma once

/**
 * Solvent density-profile snapshots at several specified frames, together
 * with the GDS front position (with an error bar) measured at each of the
 * same times.
 *
 * Two outputs from one run, sharing the same frame selection:
 *
 *   <stem>_profiles.dat   z[sigma]  rho_s(t=t1)  rho_s(t=t2) ...
 *     One column per requested frame: a literal single-frame snapshot of
 *     the solvent number density rho_s(z), binned over the full box
 *     [0, Lx) with nBins bins -- same normalisation convention as
 *     density.hpp: rho = <N> / (A * binWidth). Raw, unshifted.
 *
 *   <stem>_front.dat      t[tau]  z_f[sigma]  err
 *     For each requested frame, the GDS solvent front (median of that same
 *     kind of density histogram) averaged over a short window of
 *     --frontwindow consecutive frames starting at that frame, with the
 *     standard error of the mean across that window as the uncertainty.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "utility.hpp"
#include "gds_diffusion.hpp"   // md::detail::gibbsDividingSurface

namespace md {

struct DensityFrontConfig {
    std::string filenamePrefix;
    int    Nm{10}, Nc{100}, Ns{0};
    int    nBins{200};
    float  deltaT{1000.0f};        ///< Physical time per frame [tau]
    int    frame0{0};              ///< Reference frame for t=0 labelling
    std::vector<int> atFrames;     ///< Target frame indices for snapshots
    int    frontWindow{5};         ///< Frames averaged for the front-position error bar
};

namespace detail {

/// Solvent number-density histogram for one frame, binned over [0, Lx).
/// Returns the histogram and the bin width actually used (Lx/nBins).
inline std::pair<std::vector<double>, double>
solventHistogram(const std::string& prefix, int frameIdx, int Nm, int Nc, int Ns, int nBins,
                  float& outLy, float& outLz) {
    Frame fr = loadFrame(prefix, frameIdx, Nm, Nc, UnwrapPolicy::None, Species::Solvent);
    const int Np = Nm * Nc;
    const float Lx = fr.Lx;
    outLy = fr.Ly; outLz = fr.Lz;
    const double binWidth = Lx / nBins;

    std::vector<double> hist(nBins, 0.0);
    for (int i = Np; i < Np + Ns && i < fr.N; ++i) {
        float x = fr.rx[i] - Lx * std::floor(fr.rx[i] / Lx);   // wrap into [0, Lx)
        const int bin = static_cast<int>(x / binWidth);
        if (bin >= 0 && bin < nBins) hist[bin] += 1.0;
    }
    return {hist, binWidth};
}

} // namespace detail

inline void computeDensityFront(const DensityFrontConfig& cfg,
                                 const std::string& outputStem = "densityfront") {
    std::cout << "===== Starting Density Profile Snapshots + Front Position =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    const int nTimes = static_cast<int>(cfg.atFrames.size());
    if (nTimes == 0)
        throw std::invalid_argument("computeDensityFront(): --atframes gave no frame indices");

    // ── Density snapshots: one column per requested frame, raw (no averaging) ──
    std::vector<std::vector<double>> profiles(nTimes);
    double binWidthUsed = 0.0;

    for (int k = 0; k < nTimes; ++k) {
        float Ly{}, Lz{};
        auto [hist, binWidth] = detail::solventHistogram(
            cfg.filenamePrefix, cfg.atFrames[k], cfg.Nm, cfg.Nc, cfg.Ns, cfg.nBins, Ly, Lz);
        const double norm = static_cast<double>(Ly) * Lz * binWidth;
        profiles[k].resize(cfg.nBins);
        for (int b = 0; b < cfg.nBins; ++b) profiles[k][b] = hist[b] / norm;
        binWidthUsed = binWidth;   // for z labels; assumes ~constant box size across frames
        printProgress("Density snapshots", nTimes, k + 1);
    }
    std::cout << "\n";

    std::ofstream fd(outputStem + "_profiles.dat");
    fd << "# z[sigma]";
    for (int k = 0; k < nTimes; ++k)
        fd << "\trho_s(t=" << (cfg.atFrames[k] - cfg.frame0) * cfg.deltaT << ")";
    fd << '\n';
    for (int b = 0; b < cfg.nBins; ++b) {
        fd << (b + 0.5) * binWidthUsed;
        for (int k = 0; k < nTimes; ++k) fd << '\t' << profiles[k][b];
        fd << '\n';
    }

    // ── Front position, averaged (with SEM) over a short window per time ───────
    std::ofstream ff(outputStem + "_front.dat");
    ff << "# t[tau]\tz_f[sigma]\terr\n";

    for (int k = 0; k < nTimes; ++k) {
        std::vector<double> fronts;
        fronts.reserve(cfg.frontWindow);
        for (int w = 0; w < cfg.frontWindow; ++w) {
            const int frameIdx = cfg.atFrames[k] + w;
            float Ly{}, Lz{};
            auto [hist, binWidth] = detail::solventHistogram(
                cfg.filenamePrefix, frameIdx, cfg.Nm, cfg.Nc, cfg.Ns, cfg.nBins, Ly, Lz);
            const double zf = detail::gibbsDividingSurface(hist, binWidth, 0.0);
            if (zf >= 0.0) fronts.push_back(zf);
        }

        const double time = (cfg.atFrames[k] - cfg.frame0) * cfg.deltaT;
        if (fronts.empty()) {
            ff << time << "\tnan\tnan\n";
            continue;
        }
        const double mean = std::accumulate(fronts.begin(), fronts.end(), 0.0) / fronts.size();
        double err = 0.0;
        if (fronts.size() > 1) {
            double sq = 0.0;
            for (double v : fronts) sq += (v - mean) * (v - mean);
            err = std::sqrt(sq / (fronts.size() - 1) / fronts.size());
        }
        ff << time << '\t' << mean << '\t' << err << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time DensityFront: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd - wallStart).count()
              << " ms\n";
}

} // namespace md
