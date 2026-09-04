#pragma once

/**
 * Static structure factor S(q) for molecular systems.
 *
 * Two wavevector sampling strategies are provided as policy classes and
 * selected at compile-time via a template parameter (Strategy Pattern):
 *
 *   CartesianSampling   – regular grid in reciprocal space (sq_sc)
 *   SphericalSampling   – isotropic shell sampling (sq_new)
 *
 * Both strategies share the same accumulation and output logic.
 */

#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utility.hpp"

namespace md {

// ── Wavevector sampling policies ─────────────────────────────────────────────

/**
 * @brief Cartesian (grid) sampling: q = (nx·Δkx, ny·Δky, nz·Δkz).
 *
 * Controls the k-grid resolution via dkMultiplier and the k-range via nkBound.
 */
struct CartesianSampling {
    float dkMultiplier{1.0f};  ///< Multiplier on 2π/L spacing
    int   nkBound{10};         ///< Maximum integer index along each axis
    float binTolerance{0.1f};  ///< Binning tolerance for radial averaging

    std::vector<WaveVector> generate(float Lx, float Ly, float Lz) const {
        const float dkx = 2.f * static_cast<float>(PI) / Lx * dkMultiplier;
        const float dky = 2.f * static_cast<float>(PI) / Ly * dkMultiplier;
        const float dkz = 2.f * static_cast<float>(PI) / Lz * dkMultiplier;

        std::vector<WaveVector> wv;
        for (int nx = 0; nx <= nkBound; ++nx)
        for (int ny = 0; ny <= nkBound; ++ny)
        for (int nz = 0; nz <= nkBound; ++nz)
            wv.push_back({nx*dkx, ny*dky, nz*dkz});
        return wv;
    }
};

/**
 * @brief Isotropic (spherical) sampling: uniformly distributed orientations
 *        on concentric shells with logarithmically spaced radii.
 */
struct SphericalSampling {
    int   nkBound{100};          ///< Number of radial shells
    float binTolerance{0.1f};    ///< Binning tolerance for radial averaging
    int   nWalk{8};              ///< Angular subdivisions per axis
    float bondLength{0.97f};     ///< Reference bond length [σ]

    std::vector<WaveVector> generate(float /*Lx*/, float /*Ly*/, float /*Lz*/) const {
        const int N_walk_2 = nWalk * nWalk;
        std::vector<WaveVector> wv;
        wv.reserve(nkBound * N_walk_2);

        for (int nr = 0; nr < nkBound; ++nr) {
            const float k_r = std::pow(2.f * static_cast<float>(PI) / bondLength,
                                        ((nr+1)*4.f/nkBound) - 3.f);
            for (int ny = 0; ny < nWalk; ++ny) {
                const float theta = ny * static_cast<float>(PI) / nWalk;
                for (int nz = 0; nz < nWalk; ++nz) {
                    const float phi = nz * 2.f * static_cast<float>(PI) / nWalk;
                    wv.push_back({
                        k_r * std::sin(theta) * std::cos(phi),
                        k_r * std::sin(theta) * std::sin(phi),
                        k_r * std::cos(theta)
                    });
                }
            }
        }
        return wv;
    }
};

// ── S(q) accumulation ─────────────────────────────────────────────────────────

/**
 * @brief Accumulate the single-chain structure factor Sc(q).
 *
 * @param wv     Wavevector list.
 * @param rx/ry/rz  Positions.
 * @param Nm     Monomers per chain.
 * @param Nc     Number of chains.
 * @return Per-wavevector Sc(q) values.
 */
inline std::vector<float> accumulateSc(const std::vector<WaveVector>& wv,
                                        const std::vector<float>& rx,
                                        const std::vector<float>& ry,
                                        const std::vector<float>& rz,
                                        int Nm, int Nc) {
    const int nk = static_cast<int>(wv.size());
    std::vector<float> Sk(nk, 0.f);

    // Each wavevector is independent (writes only Sk[k]), so parallelise over k.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < Nc; ++i) {
            float chainCos{}, chainSin{};
            for (int j = 0; j < Nm; ++j) {
                const int pid = i*Nm + j;
                const float phase = wv[k].x*rx[pid] + wv[k].y*ry[pid] + wv[k].z*rz[pid];
                chainCos += std::cos(phase);
                chainSin += std::sin(phase);
            }
            Sk[k] += (chainCos*chainCos + chainSin*chainSin) / Nm;
        }
        Sk[k] /= Nc;
    }
    return Sk;
}

/**
 * @brief Accumulate the total structure factor S(q).
 */
inline std::vector<float> accumulateStot(const std::vector<WaveVector>& wv,
                                          const std::vector<float>& rx,
                                          const std::vector<float>& ry,
                                          const std::vector<float>& rz,
                                          int N) {
    const int nk = static_cast<int>(wv.size());
    std::vector<float> Sk(nk, 0.f);

    // Each wavevector is independent (writes only Sk[k]), so parallelise over k.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < nk; ++k) {
        float c{}, s{};
        for (int pid = 0; pid < N; ++pid) {
            const float phase = wv[k].x*rx[pid] + wv[k].y*ry[pid] + wv[k].z*rz[pid];
            c += std::cos(phase);
            s += std::sin(phase);
        }
        Sk[k] = (c*c + s*s) / N;
    }
    return Sk;
}

// ── Radial binning and I/O ────────────────────────────────────────────────────

/**
 * @brief Sort wavevectors by |q|², then radially average S(q).
 *
 * @param wv         Wavevectors (sorted in place).
 * @param Sk         S(k) values (reordered together with wv).
 * @param tolerance  Binning tolerance: merge wavevectors if Δ|q|² < tolerance.
 * @param outputFile Output path.
 */
inline void radialAverageAndWrite(std::vector<WaveVector>& wv,
                                   std::vector<float>&      Sk,
                                   float                    tolerance,
                                   const std::string&       outputFile) {
    // Sort by |q|²
    const int nk = static_cast<int>(wv.size());
    std::vector<int> idx(nk);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        auto sq = [&](int i){ return wv[i].x*wv[i].x+wv[i].y*wv[i].y+wv[i].z*wv[i].z; };
        return sq(a) < sq(b);
    });

    std::ofstream fo(outputFile);
    fo << "# q\t S(q)\n";

    int    count = 0;
    double knorm = 0.0, skmean = 0.0;

    for (int ii = 0; ii < nk; ++ii) {
        const int i = idx[ii];
        const float q2 = wv[i].x*wv[i].x + wv[i].y*wv[i].y + wv[i].z*wv[i].z;
        ++count;
        knorm  += std::sqrt(q2);
        skmean += Sk[i];

        const bool lastOne = (ii + 1 == nk);
        float q2_next{};
        if (!lastOne) {
            const int j = idx[ii+1];
            q2_next = wv[j].x*wv[j].x + wv[j].y*wv[j].y + wv[j].z*wv[j].z;
        }

        if (lastOne || std::fabs(q2_next - q2) > tolerance) {
            fo << knorm / count << '\t' << skmean / count << '\n';
            count = 0; knorm = 0.0; skmean = 0.0;
        }
    }
}

// ── High-level entry points ───────────────────────────────────────────────────

/**
 * @brief Compute single-chain S(q) for one snapshot.
 *
 * @tparam Sampling  CartesianSampling or SphericalSampling.
 */
template<typename Sampling>
void computeSc(const std::string& filenamePrefix,
               int                frameIndex,
               const Sampling&    sampling,
               int Nm, int Nc,
               const std::string& outputFile) {
    std::cout << "===== Starting Sc(q) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    Frame fr = loadFrame(filenamePrefix, frameIndex, Nm, Nc, UnwrapPolicy::None);
    auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
    const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;
    Nc = fr.N / Nm;

    auto wv = sampling.generate(Lx, Ly, Lz);
    std::cout << "Wavevectors: " << wv.size() << '\n';

    auto Sk = accumulateSc(wv, rx, ry, rz, Nm, Nc);
    radialAverageAndWrite(wv, Sk, sampling.binTolerance, outputFile);

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time Sc(q): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

/**
 * @brief Compute total S(q) averaged over several snapshots.
 *
 * @tparam Sampling  CartesianSampling or SphericalSampling.
 */
template<typename Sampling>
void computeStot(const std::string& filenamePrefix,
                 int                frameStart,
                 int                frameStop,
                 const Sampling&    sampling,
                 int Nm, int Nc,
                 const std::string& outputFile) {
    std::cout << "===== Starting S(q) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    // Generate wavevectors from first frame dimensions
    const Frame fr0 = loadFrame(filenamePrefix, frameStart, Nm, Nc, UnwrapPolicy::None);
    auto wv = sampling.generate(fr0.Lx, fr0.Ly, fr0.Lz);
    std::cout << "Wavevectors: " << wv.size() << '\n';

    const int nFrames = frameStop - frameStart + 1;
    std::vector<float> SkSum(wv.size(), 0.f);

    for (int snap = frameStart; snap <= frameStop; ++snap) {
        Frame fr = loadFrame(filenamePrefix, snap, Nm, Nc, UnwrapPolicy::None);
        auto Sk = accumulateStot(wv, fr.rx, fr.ry, fr.rz, fr.N);
        for (size_t k = 0; k < wv.size(); ++k) SkSum[k] += Sk[k];
    }
    for (auto& s : SkSum) s /= nFrames;

    radialAverageAndWrite(wv, SkSum, sampling.binTolerance, outputFile);

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time S(q): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
