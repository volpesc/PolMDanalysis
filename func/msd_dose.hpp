#pragma once

/**
 * MPI-parallelized dose-response monomer mobility: g1,par(t) = <dy^2+dz^2>
 * binned by the LOCAL solvent density sampled at a monomer's own position,
 * tested at both ends of the lag window.
 *
 * Rationale (see the accompanying brief):
 *  - Only the in-plane displacement (y,z, i.e. the axes normal to the film's
 *    diffusion direction x) is used. The film dilates along x as it absorbs
 *    solvent and is uniform in y/z, so dy,dz carry no convective term and
 *    need no drift correction, unlike dx.
 *  - Population membership (which concentration bin a monomer belongs to)
 *    is re-decided at every requested time origin t0 -- never fixed at the
 *    start of the run, when the film is completely dry.
 *  - A monomer only counts toward concentration bin k if its local solvent
 *    density falls inside bin k's range BOTH at t0 and at t0+dt. This is
 *    what makes the assignment robust to the front having advanced during
 *    the lag window, without ever referring to a front position.
 *  - Output is deliberately NOT pre-averaged into one number per bin: it is
 *    a per-chain, per-concentration-bin sum (sum of dy^2+dz^2, a count, and
 *    a sum of the local density actually sampled). That is exactly the
 *    sufficient statistic a chain-level block bootstrap needs downstream
 *    (monomers of one chain are not independent draws; the chain is), so
 *    all binning/error-bar/curve-fitting choices are left to the Python
 *    post-processing step instead of being baked in here.
 *
 * Population/task model: each requested (t0, dt) pair is an independent,
 * embarrassingly parallel unit of work -- unlike msd_front.hpp/msd_mpi.hpp,
 * there is no cross-origin reduction, so tasks are simply round-robined
 * across MPI ranks and each rank writes its own output files directly.
 *
 * Trajectory reading/unwrapping follows msd_mpi.hpp/msd_front.hpp exactly:
 * frame t is read from disk by one rank (round robin) and broadcast, then
 * every rank redoes the (cheap) sequential unwrap locally. Solvent positions
 * are needed only to build a density profile at each frame (not tracked
 * over time), so -- as in msd_front.hpp -- only the resulting per-frame
 * histogram is retained, not the raw solvent coordinates.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <mpi.h>

#include "utility.hpp"

namespace md {

/**
 * @brief Configuration for a dose-response monomer-mobility run.
 */
struct MSDDoseConfig {
    std::string filenamePrefix;        ///< Trajectory file prefix (e.g. "requil_")
    int    frameStart;                 ///< First frame index
    int    frameStop;                  ///< Last  frame index
    int    frameStep;                  ///< Frame stride
    float  timeStep;                   ///< Physical time between frames [tau]
    int    Nm;                         ///< Monomers per chain
    int    Nc;                         ///< Number of chains
    int    Ns{0};                      ///< Solvent particles

    double xMin       {0.0};           ///< Density-profile range start [sigma] (film-normal axis)
    double xMax       {282.0};         ///< Density-profile range end   [sigma]
    double densityBinWidth{0.25};      ///< Local solvent-density grid [sigma] -- fine, NOT the
                                        ///< 1.413 sigma grid used for the coarse six-time profiles.

    std::vector<int>    t0Frames;      ///< Explicit origin frame indices. If empty, every frame
                                        ///< whose physical time >= t0MinPhys is used as an origin.
    double t0MinPhys   {20000.0};      ///< Refuse (warn + skip) any origin earlier than this [tau].
                                        ///< Physical reasoning: at t=0 the film is completely dry,
                                        ///< so "wet" concentration bins are empty by construction;
                                        ///< an origin chosen before the front has advanced meaningfully
                                        ///< reproduces that pathology in a milder form.

    std::vector<double> concEdges;     ///< Ascending bin edges [sigma^-3] for local rho_s, e.g.
                                        ///< {0.0, 0.02, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3}.
                                        ///< The first bin (edges[0],edges[1]) is the "solvent-free"
                                        ///< bin and is what the stress-vs-solvent control reads.

    int    tailTrim    {10};           ///< Monomers excluded from EACH end of every chain (chain
                                        ///< ends are intrinsically more mobile at fixed friction).

    double tMaxLag     {30000.0};      ///< Largest lag attempted from any origin [tau].
    int    nLogPoints  {15};           ///< Log-spaced lag points per origin, before clamping to
                                        ///< available frames / tMaxLag.
};

namespace dosedetail {

/// Unique, sorted, log-spaced integer lags in [1, maxLag].
/// (Deliberately NOT named/namespaced the same as msd_front.hpp's
/// detail::logSpacedLags: both headers are pulled into the same
/// translation unit by tools.hpp, so a shared name would collide.)
inline std::vector<int> doseLogSpacedLags(int maxLag, int nPoints) {
    std::vector<int> lags;
    if (maxLag < 1) return lags;
    nPoints = std::max(nPoints, 1);
    const double logMax = std::log(static_cast<double>(maxLag));
    for (int k = 0; k < nPoints; ++k) {
        const double frac = (nPoints == 1) ? 1.0 : static_cast<double>(k) / (nPoints - 1);
        const int    lag  = std::clamp(static_cast<int>(std::round(std::exp(frac * logMax))), 1, maxLag);
        lags.push_back(lag);
    }
    std::sort(lags.begin(), lags.end());
    lags.erase(std::unique(lags.begin(), lags.end()), lags.end());
    return lags;
}

/// Which concentration bin (index into [0, edges.size()-2]) rho falls in,
/// or -1 if rho is outside [edges.front(), edges.back()).
inline int concBinOf(double rho, const std::vector<double>& edges) {
    if (edges.size() < 2) return -1;
    if (rho < edges.front() || rho >= edges.back()) return -1;
    // edges.size() is small (a handful of bins): linear scan is fine and
    // keeps this readable; this is not the hot loop (the monomer loop is).
    for (std::size_t k = 0; k + 1 < edges.size(); ++k)
        if (rho >= edges[k] && rho < edges[k+1]) return static_cast<int>(k);
    return -1;
}

} // namespace dosedetail

/**
 * @brief Compute dose-response in-plane monomer MSD using all MPI ranks.
 *
 * Must be called after MPI_Init and before MPI_Finalize.
 *
 * Writes one file per (t0, dt) task: "<outputPrefix>_t0<T0>_dt<DT>.dat",
 * each with a header documenting the bin edges and columns
 *   chain   concbin   n   sum_dyz2   sum_rho0
 * (mean g1,par for that chain/bin = sum_dyz2/n; mean local rho_s sampled =
 * sum_rho0/n). Aggregate/bootstrap/fit these in the Python post-processor.
 */
inline void computeMSDDose(const MSDDoseConfig& cfg,
                            const std::string& outputPrefix = "msd_dose") {

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int Np    = cfg.Nm * cfg.Nc;
    const int M     = (cfg.frameStop - cfg.frameStart) / cfg.frameStep + 1;
    const int nBins = std::max(1, static_cast<int>((cfg.xMax - cfg.xMin) / cfg.densityBinWidth));

    if (world_rank == 0) std::cout.setf(std::ios::unitbuf);
    if (world_rank == 0)
        std::cout << "===== Starting dose-response monomer MSD (MPI ranks: " << world_size << ") =====\n";
    if (cfg.concEdges.size() < 2) {
        if (world_rank == 0)
            std::cerr << "computeMSDDose(): need at least 2 concentration bin edges; aborting.\n";
        return;
    }
    for (std::size_t k = 1; k < cfg.concEdges.size(); ++k)
        if (cfg.concEdges[k] <= cfg.concEdges[k-1] && world_rank == 0)
            std::cerr << "computeMSDDose(): warning -- concEdges not strictly ascending at index " << k << "\n";

    const auto wallStart = std::chrono::steady_clock::now();

    // ── Read + broadcast every frame's polymer positions, and reduce each
    //    frame's solvent block to a density histogram (never retained raw) ──
    std::vector<std::vector<float>> rx(M, std::vector<float>(Np));
    std::vector<std::vector<float>> ry(M, std::vector<float>(Np));
    std::vector<std::vector<float>> rz(M, std::vector<float>(Np));
    std::vector<float> LxArr(M, 0.f), LyArr(M, 0.f), LzArr(M, 0.f);
    std::vector<std::vector<float>> solvProfile(M, std::vector<float>(nBins, 0.f)); // number density

    if (world_rank == 0) std::cout << "Reading " << M << " frames across " << world_size << " ranks...\n";

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

        // Solvent number-density profile, same wrapped-x convention as density.hpp.
        std::vector<double> hist(nBins, 0.0);
        for (int p = Np; p < Np + cfg.Ns && p < Ntot; ++p) {
            const float xw  = fx[p] - frameLx * std::floor(fx[p] / frameLx);
            const int   bin = static_cast<int>((xw - cfg.xMin) / cfg.densityBinWidth);
            if (bin >= 0 && bin < nBins) hist[bin] += 1.0;
        }
        const double A    = static_cast<double>(frameLy) * frameLz;
        const double norm = A * cfg.densityBinWidth;
        for (int b = 0; b < nBins; ++b) solvProfile[t][b] = static_cast<float>(hist[b] / norm);
    }

    for (int t = 0; t < M; ++t) {
        const int owner = t % world_size;
        MPI_Bcast(rx[t].data(), Np, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(ry[t].data(), Np, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(rz[t].data(), Np, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&LxArr[t], 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&LyArr[t], 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&LzArr[t], 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(solvProfile[t].data(), nBins, MPI_FLOAT, owner, MPI_COMM_WORLD);
    }

    if (world_rank == 0) std::cout << "Read complete. Unwrapping polymer...\n";

    // ── Sequential unwrap pass (polymer only; y,z need this for correct
    //    displacements, x needs it too -- for locating the monomer in the
    //    density profile we wrap it straight back with fmod, see below) ────
    unwrapChains(rx[0], ry[0], rz[0], cfg.Nm, cfg.Nc, Np, LxArr[0], LyArr[0], LzArr[0]);
    for (int t = 1; t < M; ++t) {
        for (int p = 0; p < Np; ++p) {
            rx[t][p] -= LxArr[t] * std::round((rx[t][p]-rx[t-1][p]) / LxArr[t]);
            ry[t][p] -= LyArr[t] * std::round((ry[t][p]-ry[t-1][p]) / LyArr[t]);
            rz[t][p] -= LzArr[t] * std::round((rz[t][p]-rz[t-1][p]) / LzArr[t]);
        }
    }

    // ── Build the origin list ─────────────────────────────────────────────
    std::vector<int> origins;
    if (!cfg.t0Frames.empty()) {
        for (int f0 : cfg.t0Frames) {
            const int t = (f0 - cfg.frameStart) / cfg.frameStep;
            if (t < 0 || t >= M) { if (world_rank==0) std::cerr << "t0 frame " << f0 << " out of range, skipping\n"; continue; }
            const double phys = t * cfg.frameStep * cfg.timeStep;
            if (phys < cfg.t0MinPhys && world_rank == 0)
                std::cerr << "WARNING: requested t0=" << phys << " tau is below t0MinPhys="
                          << cfg.t0MinPhys << " tau -- concentration groups may be near-empty "
                          << "(dry-film pathology). Proceeding anyway since it was explicit.\n";
            origins.push_back(t);
        }
    } else {
        for (int t = 0; t < M; ++t) {
            const double phys = t * cfg.frameStep * cfg.timeStep;
            if (phys >= cfg.t0MinPhys) origins.push_back(t);
        }
        if (world_rank == 0)
            std::cout << "Auto-selected " << origins.size() << " origins with t0 >= "
                      << cfg.t0MinPhys << " tau\n";
    }

    // ── Build the (t0, dt) task list ──────────────────────────────────────
    struct Task { int t0; int dt; };
    std::vector<Task> tasks;
    const int maxLagByTime = static_cast<int>(cfg.tMaxLag / (cfg.timeStep * cfg.frameStep));
    for (int t0 : origins) {
        const int maxLagByFrames = M - 1 - t0;
        const int maxLag = std::max(0, std::min(maxLagByFrames, maxLagByTime));
        for (int dt : dosedetail::doseLogSpacedLags(maxLag, cfg.nLogPoints))
            tasks.push_back({t0, dt});
    }
    if (world_rank == 0)
        std::cout << "Total (t0,dt) tasks: " << tasks.size() << " across " << world_size << " ranks\n";

    const int nConc = static_cast<int>(cfg.concEdges.size()) - 1;
    const int tailStart = cfg.tailTrim;
    const int tailEnd   = cfg.Nm - cfg.tailTrim;
    if (tailEnd <= tailStart && world_rank == 0)
        std::cerr << "computeMSDDose(): tailTrim=" << cfg.tailTrim
                  << " leaves no monomers per chain (Nm=" << cfg.Nm << ") -- reduce it.\n";

    // ── Each rank owns and fully computes its round-robined tasks ──────────
    for (std::size_t ti = world_rank; ti < tasks.size(); ti += world_size) {
        const int t0 = tasks[ti].t0;
        const int dt = tasks[ti].dt;
        const int t1 = t0 + dt;

        // sumDyz2[concbin*Nc + chain], count[...], sumRho0[...]
        std::vector<double> sumDyz2(static_cast<std::size_t>(nConc)*cfg.Nc, 0.0);
        std::vector<double> sumRho0(static_cast<std::size_t>(nConc)*cfg.Nc, 0.0);
        std::vector<long>   count  (static_cast<std::size_t>(nConc)*cfg.Nc, 0);

        for (int j = 0; j < cfg.Nc; ++j) {
            for (int i = tailStart; i < tailEnd; ++i) {
                const int pid = j*cfg.Nm + i;

                const float x0w = rx[t0][pid] - LxArr[t0]*std::floor(rx[t0][pid]/LxArr[t0]);
                const float x1w = rx[t1][pid] - LxArr[t1]*std::floor(rx[t1][pid]/LxArr[t1]);
                const int   b0  = static_cast<int>((x0w - cfg.xMin) / cfg.densityBinWidth);
                const int   b1  = static_cast<int>((x1w - cfg.xMin) / cfg.densityBinWidth);
                if (b0 < 0 || b0 >= nBins || b1 < 0 || b1 >= nBins) continue;

                const double rho0 = solvProfile[t0][b0];
                const double rho1 = solvProfile[t1][b1];

                const int cb0 = dosedetail::concBinOf(rho0, cfg.concEdges);
                const int cb1 = dosedetail::concBinOf(rho1, cfg.concEdges);
                if (cb0 < 0 || cb0 != cb1) continue;   // must hold at BOTH ends, in the SAME bin

                const double dy = static_cast<double>(ry[t1][pid]) - ry[t0][pid];
                const double dz = static_cast<double>(rz[t1][pid]) - rz[t0][pid];
                const double dyz2 = dy*dy + dz*dz;

                const std::size_t idx = static_cast<std::size_t>(cb0)*cfg.Nc + j;
                sumDyz2[idx] += dyz2;
                sumRho0[idx] += rho0;
                count  [idx] += 1;
            }
        }

        const double t0Phys = t0 * cfg.frameStep * cfg.timeStep;
        const double dtPhys = dt * cfg.frameStep * cfg.timeStep;
        std::ostringstream fnss;
        fnss << outputPrefix << "_t0" << static_cast<long>(t0Phys)
             << "_dt" << static_cast<long>(dtPhys) << ".dat";
        std::ofstream fo(fnss.str());
        fo << "# t0[tau]=" << t0Phys << "  dt[tau]=" << dtPhys << "  tailTrim=" << cfg.tailTrim << "\n";
        fo << "# concEdges[sigma^-3]:";
        for (double e : cfg.concEdges) fo << ' ' << e;
        fo << "\n";
        fo << "# chain\tconcbin\tn\tsum_dyz2\tsum_rho0\n";
        for (int cb = 0; cb < nConc; ++cb)
            for (int j = 0; j < cfg.Nc; ++j) {
                const std::size_t idx = static_cast<std::size_t>(cb)*cfg.Nc + j;
                if (count[idx] == 0) continue;
                fo << j << '\t' << cb << '\t' << count[idx] << '\t'
                   << sumDyz2[idx] << '\t' << sumRho0[idx] << '\n';
            }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        const auto wallEnd = std::chrono::steady_clock::now();
        std::cout << "Wall time MSD-dose: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
                  << " ms\n";
    }
}

} // namespace md
