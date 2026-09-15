#pragma once

/**
 * MPI-parallelized front-resolved monomer MSD, g1(t), for two populations
 * classified by their location relative to the instantaneous solvent
 * diffusion front:
 *
 *   g1_swollen(t) - monomers behind the front (solvent-penetrated region)
 *   g1_dry(t)     - monomers ahead of the front (unpenetrated / glassy region)
 *
 * Population membership is decided fresh at each time origin t0 from that
 * monomer's unwrapped x-position vs. the GDS solvent front at t0 (a monomer
 * within +-frontBuffer of the front is dropped from both populations to
 * avoid ambiguous assignment); the same monomer is then followed forward
 * over a log-spaced set of lags. Error bars are the standard error of the
 * per-origin population-mean g1, across time origins.
 *
 * As with msd_mpi.hpp, trajectories are unwrapped before any displacement is
 * computed (chain-unwrap on the first frame, temporal minimum-image unwrap
 * on every subsequent frame) so a periodic-boundary crossing never shows up
 * as a spurious +-L jump. The solvent front itself is computed from raw,
 * per-frame wrapped coordinates (same convention as gds_diffusion.hpp).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>

#include "utility.hpp"
#include "gds_diffusion.hpp"   // md::detail::gibbsDividingSurface

namespace md {

/**
 * @brief Configuration for a front-resolved MSD run.
 */
struct MSDFrontConfig {
    std::string filenamePrefix;   ///< Trajectory file prefix (e.g. "requil_")
    int    frameStart;            ///< First frame index
    int    frameStop;             ///< Last  frame index
    int    frameStep;             ///< Frame stride
    float  timeStep;              ///< Physical time between frames [tau]
    int    Nm;                    ///< Monomers per chain
    int    Nc;                    ///< Number of chains
    int    Ns{0};                 ///< Solvent particles
    double xMin       {0.0};      ///< Solvent-profile range start [sigma]
    double xMax       {100.0};    ///< Solvent-profile range end   [sigma]
    double binWidth   {3.0};      ///< Solvent-profile bin width for GDS front [sigma]
    double frontBuffer{2.0};      ///< Half-width of the exclusion zone straddling the front [sigma]
    double tMax       {300.0};    ///< Max lag time reported [tau] (stay below the tube-constraint crossover)
    int    nLogPoints {40};       ///< Number of log-spaced output lag points
};

namespace detail {

/// Unique, sorted, log-spaced integer lags in [1, maxLag].
inline std::vector<int> logSpacedLags(int maxLag, int nPoints) {
    std::vector<int> lags;
    if (maxLag < 1) return lags;
    nPoints = std::max(nPoints, 1);
    const double logMax = std::log(static_cast<double>(maxLag));
    for (int k = 0; k < nPoints; ++k) {
        const double frac   = (nPoints == 1) ? 1.0 : static_cast<double>(k) / (nPoints - 1);
        const int    lag    = std::clamp(static_cast<int>(std::round(std::exp(frac * logMax))), 1, maxLag);
        lags.push_back(lag);
    }
    std::sort(lags.begin(), lags.end());
    lags.erase(std::unique(lags.begin(), lags.end()), lags.end());
    return lags;
}

} // namespace detail

/**
 * @brief Compute front-resolved g1(t) using all available MPI ranks.
 *
 * Must be called after MPI_Init and before MPI_Finalize.
 */
inline void computeMSDFront(const MSDFrontConfig& cfg,
                             const std::string& outputFile = "msd_front.dat") {

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int Np    = cfg.Nm * cfg.Nc;
    const int M     = (cfg.frameStop - cfg.frameStart) / cfg.frameStep + 1;
    const int nBins = static_cast<int>((cfg.xMax - cfg.xMin) / cfg.binWidth);

    if (world_rank == 0)
    // Raw (still PBC-wrapped) polymer positions, per-frame box lengths, and
    // the GDS front, all indexed by frame t. Frame t is read from disk by
    // exactly one rank (round-robin ownership) and then broadcast to the
    // rest, so the disk-I/O / text-parsing cost -- the real bottleneck for
    // large solvent-laden frames -- runs in parallel instead of falling on
    // rank 0 alone.
    std::vector<std::vector<float>> rx(M, std::vector<float>(Np));
    std::vector<float> LxArr(M, 0.f), LyArr(M, 0.f), LzArr(M, 0.f);
    std::vector<float> frontPos(M, -1.f);

    if (world_rank == 0)
        std::cout << "Reading " << M << " frames across " << world_size << " ranks...\n";

    for (int t = 0; t < M; ++t) {
        const int owner = t % world_size;
        if (world_rank != owner) continue;

        const int frameIdx = cfg.frameStart + t * cfg.frameStep;
        int Ntot;
        float frameLx{}, frameLy{}, frameLz{};
        std::vector<float> fx, fy, fz;
        const std::string fn = makeInputFilename(cfg.filenamePrefix, frameIdx);
        readFrame(fn, cfg.Nm, cfg.Nc, fx, fy, fz, Ntot, frameLx, frameLy, frameLz, Species::Both);

        for (int p = 0; p < Np; ++p) { rx[t][p] = fx[p]; ry[t][p] = fy[p]; rz[t][p] = fz[p]; }
        LxArr[t] = frameLx; LyArr[t] = frameLy; LzArr[t] = frameLz;

        // Instantaneous solvent GDS front from raw (still PBC-wrapped)
        // positions -- same convention as gds_diffusion.hpp.
        std::vector<double> solvProfile(nBins, 0.0);
        for (int p = Np; p < Np + cfg.Ns && p < Ntot; ++p) {
            const int bin = static_cast<int>((fx[p] - cfg.xMin) / cfg.binWidth);
            if (bin >= 0 && bin < nBins) solvProfile[bin] += 1.0;
        }
        frontPos[t] = static_cast<float>(
            detail::gibbsDividingSurface(solvProfile, cfg.binWidth, cfg.xMin));
    }

    // ── Broadcast each frame from whichever rank actually read it ────────────
    for (int t = 0; t < M; ++t) {
        const int owner = t % world_size;
        MPI_Bcast(rx[t].data(), Np, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(ry[t].data(), Np, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(rz[t].data(), Np, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&LxArr[t], 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&LyArr[t], 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&LzArr[t], 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&frontPos[t], 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
    }

    if (world_rank == 0) std::cout << "Read complete. Unwrapping...\n";

    // ── Sequential unwrap pass ────────────────────────────────────────────────
    // Cheap (O(Np) per frame) compared to disk I/O, so every rank just redoes
    // it locally on the now-fully-populated raw data.
    unwrapChains(rx[0], ry[0], rz[0], cfg.Nm, cfg.Nc, Np, LxArr[0], LyArr[0], LzArr[0]);
    for (int t = 1; t < M; ++t) {
        for (int p = 0; p < Np; ++p) {
            rx[t][p] -= LxArr[t] * std::round((rx[t][p]-rx[t-1][p]) / LxArr[t]);
            ry[t][p] -= LyArr[t] * std::round((ry[t][p]-ry[t-1][p]) / LyArr[t]);
            rz[t][p] -= LzArr[t] * std::round((rz[t][p]-rz[t-1][p]) / LzArr[t]);
        }
    }
    // ── Log-spaced lags, bounded by both available frames and --tmax ─────────
    const int maxLagByFrames = M - 1;
    const int maxLagByTime   = static_cast<int>(cfg.tMax / (cfg.timeStep * cfg.frameStep));
    const int maxLag         = std::max(1, std::min(maxLagByFrames, maxLagByTime));
    const std::vector<int> lags = detail::logSpacedLags(maxLag, cfg.nLogPoints);
    const int L = static_cast<int>(lags.size());

    const int innerStart = cfg.Nm / 4;           // central 50 % of chain, matches msd_mpi.hpp
    const int innerEnd   = cfg.Nm - cfg.Nm / 4;

    // ── Distribute time origins across ranks ──────────────────────────────────
    const int chunk    = M / world_size;
    const int startIdx = world_rank * chunk;
    const int endIdx   = (world_rank == world_size - 1) ? M : startIdx + chunk;

    enum Pop { SWOLLEN = 0, DRY = 1, NPOP = 2 };
    // Per-lag, per-population accumulators over ORIGINS (not monomers): each
    // origin contributes one population-mean g1 value, so these give the
    // mean-of-means and its standard error across origins.
    std::vector<double> sumM (L*NPOP, 0.0), sumM2(L*NPOP, 0.0), nOrig(L*NPOP, 0.0);

    for (int t0 = startIdx; t0 < endIdx; ++t0) {
        if (frontPos[t0] < 0.f) continue;   // GDS front not found at this origin
        const float xf = frontPos[t0];

        for (int li = 0; li < L; ++li) {
            const int dt = lags[li];
            if (t0 + dt >= M) continue;

            double sum[NPOP] = {0.0, 0.0};
            long   cnt[NPOP] = {0, 0};

            for (int j = 0; j < cfg.Nc; ++j)
                for (int i = innerStart; i < innerEnd; ++i) {
                    const int pid = j*cfg.Nm + i;
                    const float x0 = rx[t0][pid];

                    int pop;
                    if      (x0 < xf - cfg.frontBuffer) pop = SWOLLEN;
                    else if (x0 > xf + cfg.frontBuffer) pop = DRY;
                    else continue;   // inside the buffer straddling the front: ambiguous

                    const double dx = static_cast<double>(rx[t0+dt][pid]) - rx[t0][pid];
                    const double dy = static_cast<double>(ry[t0+dt][pid]) - ry[t0][pid];
                    const double dz = static_cast<double>(rz[t0+dt][pid]) - rz[t0][pid];
                    sum[pop] += dx*dx + dy*dy + dz*dz;
                    ++cnt[pop];
                }

            for (int p = 0; p < NPOP; ++p) {
                if (cnt[p] == 0) continue;
                const double m = sum[p] / static_cast<double>(cnt[p]);
                sumM [li*NPOP+p] += m;
                sumM2[li*NPOP+p] += m*m;
                nOrig[li*NPOP+p] += 1.0;
            }
        }
    }

    // ── Reduce to rank 0 ──────────────────────────────────────────────────────
    auto reduceD = [&](std::vector<double>& v) {
        MPI_Reduce(world_rank == 0 ? MPI_IN_PLACE : v.data(), v.data(),
                   static_cast<int>(v.size()), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    };
    reduceD(sumM); reduceD(sumM2); reduceD(nOrig);

    // ── Output (rank 0 only) ──────────────────────────────────────────────────
    if (world_rank == 0) {
        std::ofstream fo(outputFile);
        fo      << "# t[tau]\tg1_swollen[sigma^2]\terr\tg1_dry[sigma^2]\terr\n";
        std::cout << "# t[tau]\tg1_swollen[sigma^2]\terr\tg1_dry[sigma^2]\terr\n";

        for (int li = 0; li < L; ++li) {
            const double time = lags[li] * cfg.frameStep * cfg.timeStep;

            double g1[NPOP], err[NPOP];
            for (int p = 0; p < NPOP; ++p) {
                const double n = nOrig[li*NPOP+p];
                if (n < 0.5) { g1[p] = 0.0; err[p] = 0.0; continue; }
                const double mean = sumM[li*NPOP+p] / n;
                g1[p] = mean;
                if (n > 1.5) {
                    const double var = std::max((sumM2[li*NPOP+p] - n*mean*mean) / (n - 1.0), 0.0);
                    err[p] = std::sqrt(var / n);
                } else {
                    err[p] = 0.0;
                }
            }

            fo      << time << '\t' << g1[SWOLLEN] << '\t' << err[SWOLLEN]
                             << '\t' << g1[DRY]     << '\t' << err[DRY] << '\n';
            std::cout << time << '\t' << g1[SWOLLEN] << '\t' << err[SWOLLEN]
                               << '\t' << g1[DRY]     << '\t' << err[DRY] << '\n';
        }

        const auto wallEnd = std::chrono::steady_clock::now();
        std::cout << "Wall time MSD-front: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
                  << " ms\n";
    }
}

} // namespace md
