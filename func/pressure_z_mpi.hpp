#pragma once

/**
 * @file pressure_z_mpi.hpp
 * @brief Irving–Kirkwood layer pressure tensor P(x), MPI-parallel.
 *
 * Computes the full symmetric stress tensor (Voigt order xx yy zz xy xz yz) as a
 * function of position x along the layer-normal direction. Contributions:
 *
 *   Kinetic (Eq. 1):    P^K_ab(x)   = 1/(A dX) * sum_{i in slab} v_i,a v_i,b
 *   Pairwise (Eq. 2):   P^Upair_ab  = 1/A * sum_{i!=j} f_ij,a r_ij,b H(x_i,x_j,x)
 *   Bending  (Eq. 5):   P^Ubend_ab  = 1/A * sum_triplets [ f_i,a r_ij,b H(x_i,x_j,x)
 *                                                        + f_k,a r_kj,b H(x_k,x_j,x) ]
 *
 * with H(x_i,x_j,x) = 1/|x_i-x_j| when the plane x lies between x_i and x_j
 * (see md::heaviside). The pairwise term includes both non-bonded (WCA + cosine
 * attraction) and FENE bond forces; both are two-body. Bending is decomposed
 * onto its two bonds using the actual three-body forces (f_i, f_k), which
 * reproduces the exact bending virial (sum over layers of Tr P^Ubend * A * dX
 * equals sum_triplets f_i.r_ij + f_k.r_kj).
 *
 * The layer-normal (x) direction is treated as non-periodic (interfaces / walls),
 * matching the rest of the suite; the minimum-image convention is applied only in
 * y and z. The wall contribution (needs wall positions/forces not present in the
 * trajectory) is not included.
 *
 * Parallelisation: pairwise/bond/bending work is split across MPI ranks (particles
 * for pairs, chains for bonded terms); each rank accumulates full-length local
 * profiles that are reduced onto rank 0. The kinetic term is cheap and computed on
 * rank 0.
 *
 * Output files (columns: x  Pxx Pyy Pzz Pxy Pxz Pyz):
 *   <stem>_PK_p.txt   kinetic, polymer
 *   <stem>_PK_s.txt   kinetic, solvent
 *   <stem>_Upair.txt  pairwise virial (non-bonded + FENE)
 *   <stem>_Ubend.txt  bending virial
 *   <stem>_total.txt  sum of all contributions
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>

#include "utility.hpp"
#include "forces.hpp"

namespace md {

/**
 * @brief Configuration for a layer-pressure run.
 */
struct PressureConfig {
    std::string filenamePrefix; ///< Trajectory file prefix
    int   frameIndex;           ///< Frame to analyse
    int   Nm;                   ///< Monomers per chain
    int   Nc;                   ///< Number of chains
    int   N_s;                  ///< Number of solvent particles
    int   nLayer;               ///< Number of layers along x
};

// ── Voigt index map: {alpha, beta} -> Voigt index ────────────────────────────
static constexpr std::array<std::array<int,2>, 6> VOIGT_IDX = {{
    {0,0},{1,1},{2,2},{0,1},{0,2},{1,2}
}};

namespace detail {

/// Minimum image in y,z only (x is the non-periodic layer-normal direction).
inline void micYZ(double& dy, double& dz, double Ly, double Lz) noexcept {
    if      (dy >  0.5*Ly) dy -= Ly;
    else if (dy < -0.5*Ly) dy += Ly;
    if      (dz >  0.5*Lz) dz -= Lz;
    else if (dz < -0.5*Lz) dz += Lz;
}

/**
 * @brief Linked-cell list for O(N) neighbour search.
 *
 * y and z are periodic (cells wrap); x is non-periodic (cells clamp, no wrap).
 * The 27-cell stencil is de-duplicated so small boxes (< 3 cells on a periodic
 * axis) stay correct.
 */
class PressureCellList {
public:
    void build(const std::vector<Particle>& P, double Lx, double Ly, double Lz,
               double cutoff) {
        cs_ = cutoff;
        nx_ = std::max(1, static_cast<int>(Lx / cs_));
        ny_ = std::max(1, static_cast<int>(Ly / cs_));
        nz_ = std::max(1, static_cast<int>(Lz / cs_));
        head_.assign(static_cast<std::size_t>(nx_) * ny_ * nz_, -1);
        next_.assign(P.size(), -1);
        for (std::size_t i = 0; i < P.size(); ++i) {
            const int c = cellOf(P[i]);
            next_[i] = head_[c];
            head_[c] = static_cast<int>(i);
        }
    }

    template <class Fn>
    void forEachNeighbour(const Particle& q, Fn&& fn) const {
        const int ix = cxClamp(q.pos[0]);
        const int iy = wrap(cfloor(q.pos[1]), ny_);
        const int iz = wrap(cfloor(q.pos[2]), nz_);
        int visited[27]; int nv = 0;
        for (int dx = -1; dx <= 1; ++dx) {
            const int jx = ix + dx;
            if (jx < 0 || jx >= nx_) continue;          // no wrap in x
            for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz) {
                const int c = cellIndex(jx, wrap(iy+dy, ny_), wrap(iz+dz, nz_));
                bool seen = false;
                for (int v = 0; v < nv; ++v) if (visited[v]==c){ seen=true; break; }
                if (seen) continue;
                visited[nv++] = c;
                for (int j = head_[c]; j != -1; j = next_[j]) fn(j);
            }
        }
    }

private:
    static int wrap(int i, int n) { i %= n; return i < 0 ? i + n : i; }
    int cfloor(double v) const { return static_cast<int>(std::floor(v / cs_)); }
    int cxClamp(double x) const { int i = cfloor(x); return i<0?0:(i>=nx_?nx_-1:i); }
    int cellIndex(int ix, int iy, int iz) const { return (ix*ny_ + iy)*nz_ + iz; }
    int cellOf(const Particle& p) const {
        return cellIndex(cxClamp(p.pos[0]), wrap(cfloor(p.pos[1]),ny_),
                         wrap(cfloor(p.pos[2]),nz_));
    }
    double cs_ {1.0};
    int nx_ {1}, ny_ {1}, nz_ {1};
    std::vector<int> head_, next_;
};

/// Add a two-body stress contribution f (on the first endpoint) with bond
/// vector r = (first - second) to every layer plane the bond crosses.
inline void addBondStress(std::vector<Tensor>& V, const Vec3& f,
                          double rx, double ry, double rz,
                          double xa, double xb,
                          double dX, int nLayer, double invA) noexcept {
    const double adx = std::fabs(xa - xb);
    if (adx < 1e-12) return;
    const double w = invA / adx;                        // (1/A) * 1/|x_ij|
    const double lo = std::min(xa, xb), hi = std::max(xa, xb);
    const double rb[3] = {rx, ry, rz};
    int lstart = static_cast<int>(std::floor(lo / dX)) + 1;
    int lend   = static_cast<int>(std::ceil (hi / dX)) - 1;
    lstart = std::max(lstart, 0);
    lend   = std::min(lend, nLayer - 1);
    for (int l = lstart; l <= lend; ++l) {
        const double xp = l * dX;
        if (xp <= lo || xp >= hi) continue;             // strictly between
        for (int vi = 0; vi < 6; ++vi)
            V[l].ij[vi] += f[VOIGT_IDX[vi][0]] * rb[VOIGT_IDX[vi][1]] * w;
    }
}

/**
 * @brief Serial accumulation of the pairwise + bending virial profiles.
 *
 * Ranges [iBegin,iEnd) (particles, for non-bonded pairs) and [cBegin,cEnd)
 * (chains, for FENE + bending) allow an MPI partition; pass the full ranges for
 * a serial computation. Non-bonded pairs are counted once (j > i).
 */
inline void accumulateVirial(const std::vector<Particle>& P, int N_p,
                             int Nm, double Lx, double Ly, double Lz, int nLayer,
                             const PressureCellList& cl,
                             int iBegin, int iEnd, int cBegin, int cEnd,
                             std::vector<Tensor>& virPair,
                             std::vector<Tensor>& virBend) {
    const double dX   = Lx / nLayer;
    const double invA = 1.0 / (Ly * Lz);

    auto sigmaOf = [&](int a, int b) {
        const bool pa = a < N_p, pb = b < N_p;
        if (pa && pb) return kg::SIGMA_PP;
        if (!pa && !pb) return kg::SIGMA_SS;
        return kg::SIGMA_PS;
    };

    // ── Non-bonded pairs (WCA + cosine attraction) ────────────────────────────
    for (int i = iBegin; i < iEnd; ++i) {
        const Particle& pi = P[i];
        cl.forEachNeighbour(pi, [&](int j) {
            if (j <= i) return;                         // each unordered pair once
            double rx = pi.pos[0] - P[j].pos[0];        // r_ij = r_i - r_j (x raw)
            double ry = pi.pos[1] - P[j].pos[1];
            double rz = pi.pos[2] - P[j].pos[2];
            micYZ(ry, rz, Ly, Lz);
            const double sig = sigmaOf(i, j);
            const double rca = wcaCutoff(sig) * std::sqrt(2.0);
            const double r2  = rx*rx + ry*ry + rz*rz;
            if (r2 >= rca*rca || r2 < 1e-24) return;
            // force on i by j = nonbondedForce(dr = r_j - r_i) = nonbondedForce(-r_ij)
            const Vec3 fi = nonbondedForce({-rx,-ry,-rz}, kg::EPSILON, sig, kg::ALPHA);
            addBondStress(virPair, fi, rx, ry, rz, pi.pos[0], P[j].pos[0],
                          dX, nLayer, invA);
        });
    }

    // ── Bonded FENE (two-body) and bending (three-body) along chains ──────────
    for (int c = cBegin; c < cEnd; ++c) {
        const int base = c * Nm;

        for (int b = 0; b < Nm - 1; ++b) {
            const int i = base + b, j = base + b + 1;
            double rx = P[i].pos[0] - P[j].pos[0];
            double ry = P[i].pos[1] - P[j].pos[1];
            double rz = P[i].pos[2] - P[j].pos[2];
            micYZ(ry, rz, Ly, Lz);
            const Vec3 fi = feneBondForce({-rx,-ry,-rz});   // force on i
            addBondStress(virPair, fi, rx, ry, rz, P[i].pos[0], P[j].pos[0],
                          dX, nLayer, invA);
        }

        for (int b = 1; b < Nm - 1; ++b) {
            const int i = base + b - 1, j = base + b, k = base + b + 1;
            // Build MIC-consistent positions relative to apex j (x raw).
            double rijx = P[i].pos[0]-P[j].pos[0], rijy = P[i].pos[1]-P[j].pos[1], rijz = P[i].pos[2]-P[j].pos[2];
            double rkjx = P[k].pos[0]-P[j].pos[0], rkjy = P[k].pos[1]-P[j].pos[1], rkjz = P[k].pos[2]-P[j].pos[2];
            micYZ(rijy, rijz, Ly, Lz);
            micYZ(rkjy, rkjz, Ly, Lz);
            const Vec3 rj{P[j].pos[0], P[j].pos[1], P[j].pos[2]};
            const Vec3 ri{rj[0]+rijx, rj[1]+rijy, rj[2]+rijz};
            const Vec3 rk{rj[0]+rkjx, rj[1]+rkjy, rj[2]+rkjz};
            const BendForces bf = bendingForces(ri, rj, rk);
            // bond i-j carries f_i; bond k-j carries f_k
            addBondStress(virBend, bf.fi, rijx, rijy, rijz, ri[0], rj[0], dX, nLayer, invA);
            addBondStress(virBend, bf.fk, rkjx, rkjy, rkjz, rk[0], rj[0], dX, nLayer, invA);
        }
    }
}

} // namespace detail

/**
 * @brief Compute and write the layer pressure tensor P(x) using MPI.
 *
 * @param cfg         Run parameters.
 * @param outputStem  File stem; outputs are "<stem>_PK_p.txt" etc.
 */
inline void computeLayerPressure(const PressureConfig& cfg,
                                  const std::string& outputStem = "pressure") {
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (rank == 0) std::cout << "===== Starting P(x) =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

    const int N_p = cfg.Nc * cfg.Nm;

    float Lx{}, Ly{}, Lz{};
    std::vector<Particle> particles;

    if (rank == 0) {
        const std::string fn = makeInputFilename(cfg.filenamePrefix, cfg.frameIndex);
        int Ntmp;
        readHeader(fn, Ntmp, Lx, Ly, Lz);
        particles = readParticles(fn);
    }

    MPI_Bcast(&Lx, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ly, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lz, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int nParticles = static_cast<int>(particles.size());
    MPI_Bcast(&nParticles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) particles.resize(nParticles);
    MPI_Bcast(particles.data(), nParticles * static_cast<int>(sizeof(Particle)),
              MPI_BYTE, 0, MPI_COMM_WORLD);

    const int    N      = nParticles;
    const double deltaX = static_cast<double>(Lx) / cfg.nLayer;
    const double A      = static_cast<double>(Ly) * Lz;

    // ── Virial: cell list + partitioned accumulation, reduced to rank 0 ───────
    detail::PressureCellList cl;
    cl.build(particles, Lx, Ly, Lz, wcaCutoff(kg::SIGMA_PP) * std::sqrt(2.0));

    const int iBegin = static_cast<int>(static_cast<long>(rank)   * N / nproc);
    const int iEnd   = static_cast<int>(static_cast<long>(rank+1) * N / nproc);
    const int cBegin = static_cast<int>(static_cast<long>(rank)   * cfg.Nc / nproc);
    const int cEnd   = static_cast<int>(static_cast<long>(rank+1) * cfg.Nc / nproc);

    std::vector<Tensor> virPair(cfg.nLayer), virBend(cfg.nLayer);
    detail::accumulateVirial(particles, N_p, cfg.Nm, Lx, Ly, Lz, cfg.nLayer,
                             cl, iBegin, iEnd, cBegin, cEnd, virPair, virBend);

    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : virPair.data(), virPair.data(),
               cfg.nLayer * 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : virBend.data(), virBend.data(),
               cfg.nLayer * 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank != 0) return;   // non-root ranks are done after the reductions

    // ── Kinetic term (rank 0; cheap) ──────────────────────────────────────────
    std::vector<Tensor> kinP(cfg.nLayer), kinS(cfg.nLayer);
    for (int l = 0; l < cfg.nLayer; ++l) {
        const double x = l * deltaX;
        for (int vi = 0; vi < 6; ++vi) {
            const int a = VOIGT_IDX[vi][0], b = VOIGT_IDX[vi][1];
            kinP[l].ij[vi] = kineticPressure_pp(A, x, deltaX, particles, N_p, a, b);
            kinS[l].ij[vi] = kineticPressure_ss(A, x, deltaX, particles, N_p, cfg.N_s, a, b);
        }
    }

    // ── Output ────────────────────────────────────────────────────────────────
    const std::string header = "# x\t Pxx\t Pyy\t Pzz\t Pxy\t Pxz\t Pyz\n";
    auto writeFile = [&](const std::string& suffix, const std::vector<Tensor>& data) {
        std::ofstream fo(outputStem + "_" + suffix + ".txt");
        fo << header;
        for (int l = 0; l < cfg.nLayer; ++l) {
            fo << l * deltaX;
            for (int vi = 0; vi < 6; ++vi) fo << '\t' << data[l].ij[vi];
            fo << '\n';
        }
    };

    std::vector<Tensor> total(cfg.nLayer);
    for (int l = 0; l < cfg.nLayer; ++l)
        for (int vi = 0; vi < 6; ++vi)
            total[l].ij[vi] = kinP[l].ij[vi] + kinS[l].ij[vi]
                            + virPair[l].ij[vi] + virBend[l].ij[vi];

    writeFile("PK_p",  kinP);
    writeFile("PK_s",  kinS);
    writeFile("Upair", virPair);
    writeFile("Ubend", virBend);
    writeFile("total", total);

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Wall time P(x): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
