#pragma once

/**
 * @file bond_angle.hpp
 * @brief Bond angle distribution P(θ).
 *
 * @details
 * Computes the probability distribution of backbone bond angles θ, defined
 * by three consecutive monomers i, i+1, i+2 along each chain:
 *
 *   cos θ = (r_{i,i+1} · r_{i+1,i+2}) / (|r_{i,i+1}| · |r_{i+1,i+2}|)
 *
 * The raw distribution P(θ) is normalised over the angle so that
 *   ∫ P(θ) dθ = 1.
 *
 * Dividing by sin(θ) removes the geometric solid-angle factor: for isotropic
 * (freely-jointed) orientation P(θ) ∝ sin(θ), so P(θ)/sin(θ) is flat.
 *
 * Minimum image convention is applied to both bond vectors.
 *
 * Output file: columns  theta[deg]  P(theta)  P(theta)/sin(theta)
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
 * @brief Configuration for a bond angle distribution run.
 */
struct BondAngleConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int    frameStart   {0};
    int    frameStop    {100};
    int    frameStep    {1};
    int    Nm           {10};   ///< Monomers per chain
    int    Nc           {100};  ///< Number of chains
    int    nBins        {180};  ///< Histogram bins over [0°, 180°]
};

/**
 * @brief Compute P(θ) averaged over a trajectory range.
 *
 * @param cfg        Run parameters.
 * @param outputFile Destination file (default: "bond_angle.dat").
 */
inline void computeBondAngle(const BondAngleConfig& cfg,
                              const std::string& outputFile = "bond_angle.dat") {
    std::cout << "===== Starting Bond Angle Distribution P(theta) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    const double dTheta = M_PI / cfg.nBins;  // bin width in radians
    std::vector<double> hist(cfg.nBins, 0.0);
    long totalAngles = 0;

    for (int smp = cfg.frameStart; smp <= cfg.frameStop; smp += cfg.frameStep) {
        Frame fr = loadFrame(cfg.filenamePrefix, smp, cfg.Nm, cfg.Nc, UnwrapPolicy::None);
        auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
        const int N = fr.N;
        const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;

        const int Nc_actual = N / cfg.Nm;

        for (int i = 0; i < Nc_actual; ++i) {
            for (int j = 0; j < cfg.Nm - 2; ++j) {
                const int p0 = i*cfg.Nm + j;
                const int p1 = p0 + 1;
                const int p2 = p0 + 2;

                // Bond vector i→i+1
                float b1x, b1y, b1z;
                minImageVec(rx, ry, rz, p0, p1, Lx, Ly, Lz, b1x, b1y, b1z);

                // Bond vector i+1→i+2
                float b2x, b2y, b2z;
                minImageVec(rx, ry, rz, p1, p2, Lx, Ly, Lz, b2x, b2y, b2z);

                const double r1 = std::sqrt(b1x*b1x + b1y*b1y + b1z*b1z);
                const double r2 = std::sqrt(b2x*b2x + b2y*b2y + b2z*b2z);
                if (r1 == 0.0 || r2 == 0.0) continue;

                double cosTheta = (b1x*b2x + b1y*b2y + b1z*b2z) / (r1 * r2);
                cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
                const double theta = std::acos(cosTheta);

                const int bin = static_cast<int>(theta / dTheta);
                if (bin >= 0 && bin < cfg.nBins) {
                    hist[bin] += 1.0;
                    ++totalAngles;
                }
            }
        }

        printProgress("BondAngle", cfg.frameStop + 1, smp);
    }
    std::cout << "\n100%\n";

    // ── Normalise: divide by dθ for a density, and by sin(θ) to remove the
    //    geometric solid-angle factor (third column). ──────────────────────────
    std::ofstream fo(outputFile);
    fo << "# theta [deg]\t P(theta)\t P(theta)/sin(theta)\n";

    for (int b = 0; b < cfg.nBins; ++b) {
        const double theta  = (b + 0.5) * dTheta;
        const double sinT   = std::sin(theta);
        // Raw probability density
        const double p      = hist[b] / (totalAngles * dTheta);
        // Solid-angle weighted
        const double pSin   = (sinT > 1e-12) ? p / sinT : 0.0;

        fo << theta * 180.0 / M_PI << '\t' << p << '\t' << pSin << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time Bond Angle: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd - wallStart).count()
              << " ms\n";
}

} // namespace md
