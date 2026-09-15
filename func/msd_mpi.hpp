#pragma once

/**
 * MPI-parallelized Mean Squared Displacement (MSD) analysis.
 *
 * Computes three standard polymer MSD observables:
 *   g1(t) – monomer MSD relative to lab frame (inner monomers only)
 *   g2(t) – monomer MSD relative to chain center of mass
 *   g3(t) – chain center-of-mass MSD
 *
 * Trajectories are unwrapped before any displacement is computed: the first
 * frame is unwrapped chain-by-chain (md::unwrapChains) and every subsequent
 * frame is unwrapped in time relative to the previous one (minimum-image
 * per particle), so a particle crossing a periodic boundary never shows up
 * as a spurious +-L jump in g1/g2/g3.
 *
 * Parallelization strategy: the time-origin loop is distributed across MPI
 * ranks. Results are reduced to rank 0 for output.
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>

#include "utility.hpp"

namespace md {

/**
 * @brief Configuration for an MSD run.
 */
struct MSDConfig {
    std::string filenamePrefix; ///< Trajectory file prefix (e.g. "requil_")
    int    frameStart;          ///< First frame index
    int    frameStop;           ///< Last  frame index
    int    frameStep;           ///< Frame stride
    float  timeStep;            ///< Physical time between frames [simulation units]
    int    Nm;                  ///< Monomers per chain
    int    Nc;                  ///< Number of chains
};

/**
 * @brief Compute MSD using all available MPI ranks.
 *
 * Must be called after MPI_Init and before MPI_Finalize.
 *
 * @param cfg  Run parameters.
 * @param outputFile  Path for the output data file.
 */
inline void computeMSD(const MSDConfig& cfg, const std::string& outputFile = "msd.dat") {

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int N  = cfg.Nm * cfg.Nc;
    const int M  = (cfg.frameStop - cfg.frameStart) / cfg.frameStep + 1;


    if (world_rank == 0)
        std::cout << "===== Starting MSD (MPI ranks: " << world_size << ") =====\n";

    const auto wallStart = std::chrono::steady_clock::now();

    // Raw (still PBC-wrapped) positions and per-frame box lengths, indexed
    // by frame t. Frame t is read from disk by exactly one rank
    // (round-robin ownership) and then broadcast to the rest, so the
    // disk-I/O / text-parsing cost runs in parallel instead of falling on
    // rank 0 alone.
    std::vector<std::vector<float>> rx(M, std::vector<float>(N));
    std::vector<std::vector<float>> ry(M, std::vector<float>(N));
    std::vector<std::vector<float>> rz(M, std::vector<float>(N));
    std::vector<float> LxArr(M, 0.f), LyArr(M, 0.f), LzArr(M, 0.f);

    if (world_rank == 0)
        std::cout << "Reading " << M << " frames across " << world_size << " ranks...\n";

    for (int t = 0; t < M; ++t) {
        const int owner = t % world_size;
        if (world_rank != owner) continue;

        const int frameIdx = cfg.frameStart + t * cfg.frameStep;
        int Ntmp;
        float frameLx{}, frameLy{}, frameLz{};
        const std::string fn = makeInputFilename(cfg.filenamePrefix, frameIdx);
        readFrame(fn, cfg.Nm, cfg.Nc, rx[t], ry[t], rz[t], Ntmp, frameLx, frameLy, frameLz);
        LxArr[t] = frameLx; LyArr[t] = frameLy; LzArr[t] = frameLz;
    }

    // ── Broadcast each frame from whichever rank actually read it ────────────
    for (int t = 0; t < M; ++t) {
        const int owner = t % world_size;
        MPI_Bcast(rx[t].data(), N, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(ry[t].data(), N, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(rz[t].data(), N, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&LxArr[t], 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&LyArr[t], 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&LzArr[t], 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
    }

    if (world_rank == 0) std::cout << "Read complete. Unwrapping...\n";

    // ── Sequential unwrap pass, plus per-frame CoMs ───────────────────────────
    // Unwrap is O(N) per frame (cheap vs. disk I/O), so every rank just
    // redoes it locally on the now-fully-populated raw data instead of
    // broadcasting pre-unwrapped results from one rank. CoM has no
    // cross-frame dependency either, so it's computed redundantly per rank
    // here too, saving what used to be a separate broadcast round.
    std::vector<std::vector<float>> cx(M, std::vector<float>(cfg.Nc, 0.f));
    std::vector<std::vector<float>> cy(M, std::vector<float>(cfg.Nc, 0.f));
    std::vector<std::vector<float>> cz(M, std::vector<float>(cfg.Nc, 0.f));
    std::vector<float> scx(M, 0.f), scy(M, 0.f), scz(M, 0.f);

    unwrapChains(rx[0], ry[0], rz[0], cfg.Nm, cfg.Nc, N, LxArr[0], LyArr[0], LzArr[0]);
    for (int t = 1; t < M; ++t) {
        for (int p = 0; p < N; ++p) {
            rx[t][p] -= LxArr[t] * std::round((rx[t][p]-rx[t-1][p]) / LxArr[t]);
            ry[t][p] -= LyArr[t] * std::round((ry[t][p]-ry[t-1][p]) / LyArr[t]);
            rz[t][p] -= LzArr[t] * std::round((rz[t][p]-rz[t-1][p]) / LzArr[t]);
        }
    }
    for (int t = 0; t < M; ++t) {
        computeSystemCoM(cfg.Nm, cfg.Nc, rx[t], ry[t], rz[t], N, scx[t], scy[t], scz[t]);
        computeCoM      (cfg.Nm, cfg.Nc, rx[t], ry[t], rz[t], cx[t], cy[t], cz[t]);
    }


    // ── Distribute time origins across ranks ──────────────────────────────────
    const int chunk     = M / world_size;
    const int startIdx  = world_rank * chunk;
    const int endIdx    = (world_rank == world_size - 1) ? M : startIdx + chunk;

    // Per-lag-time accumulators (double to avoid precision loss over long sums)
    std::vector<double> g1x(M,0.), g1y(M,0.), g1z(M,0.); // inner monomer MSD
    std::vector<double> g2x(M,0.), g2y(M,0.), g2z(M,0.); // monomer-CoM MSD
    std::vector<double> g3x(M,0.), g3y(M,0.), g3z(M,0.); // CoM MSD
    std::vector<int>   cnt(M, 0);

    const int innerStart = cfg.Nm / 4;           // central 50 % of chain
    const int innerEnd   = cfg.Nm - cfg.Nm / 4;

    for (int t = startIdx; t < endIdx; ++t) {
        for (int dt = 1; (t + dt) < M; ++dt) {
            ++cnt[dt];

            // g1: inner monomers vs. lab frame
            for (int j = 0; j < cfg.Nc; ++j)
                for (int i = innerStart; i < innerEnd; ++i) {
                    const int pid = j*cfg.Nm + i;
                    const double dx = static_cast<double>(rx[t+dt][pid]) - rx[t][pid];
                    const double dy = static_cast<double>(ry[t+dt][pid]) - ry[t][pid];
                    const double dz = static_cast<double>(rz[t+dt][pid]) - rz[t][pid];
                    g1x[dt] += dx*dx; g1y[dt] += dy*dy; g1z[dt] += dz*dz;
                }

            for (int j = 0; j < cfg.Nc; ++j) {
                // g3: chain CoM MSD, with the overall system-CoM drift removed.
                const double d3x = (static_cast<double>(cx[t+dt][j]) - scx[t+dt])
                                 - (static_cast<double>(cx[t][j])   - scx[t]);
                const double d3y = (static_cast<double>(cy[t+dt][j]) - scy[t+dt])
                                 - (static_cast<double>(cy[t][j])   - scy[t]);
                const double d3z = (static_cast<double>(cz[t+dt][j]) - scz[t+dt])
                                 - (static_cast<double>(cz[t][j])   - scz[t]);
                g3x[dt] += d3x*d3x; g3y[dt] += d3y*d3y; g3z[dt] += d3z*d3z;

                // g2: monomer displacement in its own chain-CoM frame (drift-free).
                //     u_i = r_i - R_cm,chain; subtracting the chain CoM already
                //     removes all translational motion, so no system-CoM term.
                for (int i = 0; i < cfg.Nm; ++i) {
                    const int pid = j*cfg.Nm + i;
                    const double dx = (static_cast<double>(rx[t+dt][pid]) - cx[t+dt][j])
                                    - (static_cast<double>(rx[t][pid])    - cx[t][j]);
                    const double dy = (static_cast<double>(ry[t+dt][pid]) - cy[t+dt][j])
                                    - (static_cast<double>(ry[t][pid])    - cy[t][j]);
                    const double dz = (static_cast<double>(rz[t+dt][pid]) - cz[t+dt][j])
                                    - (static_cast<double>(rz[t][pid])    - cz[t][j]);
                    g2x[dt] += dx*dx; g2y[dt] += dy*dy; g2z[dt] += dz*dz;
                }
            }
        }
    }

    // ── Reduce to rank 0 ──────────────────────────────────────────────────────
    auto reduce = [&](std::vector<double>& v) {
        MPI_Reduce(world_rank == 0 ? MPI_IN_PLACE : v.data(),
                   v.data(), M, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    };
    reduce(g1x); reduce(g1y); reduce(g1z);
    reduce(g2x); reduce(g2y); reduce(g2z);
    reduce(g3x); reduce(g3y); reduce(g3z);
    MPI_Reduce(world_rank == 0 ? MPI_IN_PLACE : cnt.data(),
               cnt.data(), M, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // ── Output (rank 0 only) ──────────────────────────────────────────────────
    if (world_rank == 0) {
        const int innerCount = cfg.Nc * (innerEnd - innerStart);

        std::ofstream fo(outputFile);
        fo << "# t\t g1(t)\t g2(t)\t g3(t)\n";
        std::cout << "# t\t g1(t)\t g2(t)\t g3(t)\n";

        for (int t = 1; t < M; ++t) {
            if (cnt[t] == 0) continue;
            const double g1 = (g1x[t]+g1y[t]+g1z[t]) / (static_cast<double>(innerCount) * cnt[t]);
            const double g2 = (g2x[t]+g2y[t]+g2z[t]) / (static_cast<double>(N)          * cnt[t]);
            const double g3 = (g3x[t]+g3y[t]+g3z[t]) / (static_cast<double>(cfg.Nc)     * cnt[t]);
            const float time = t * cfg.frameStep * cfg.timeStep;

            fo << time << '\t' << g1 << '\t' << g2 << '\t' << g3 << '\n';
            std::cout << time << '\t' << g1 << '\t' << g2 << '\t' << g3 << '\n';
        }

        const auto wallEnd = std::chrono::steady_clock::now();
        std::cout << "Wall time MSD: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
                  << " ms\n";
    }
}

} // namespace md
