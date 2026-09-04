#pragma once

/**
 * @file force_ellipsoid.hpp
 * @brief Per-monomer force ellipsoid analysis.
 *
 * @details
 * For every polymer monomer we compute the net instantaneous force f each
 * frame, then accumulate the time-averaged force-covariance tensor
 *
 *   T_ab = < f_a f_b >_frames      (a, b in {x, y, z})
 *
 * Note that a single frame's outer product f (x) f is rank-1 (its eigenvalues
 * are (|f|^2, 0, 0)), so the anisotropy only becomes meaningful once T is
 * averaged over frames. We therefore accumulate the raw tensor per frame and
 * diagonalise the *time-averaged* tensor once, at the end.
 *
 * Diagonalising the averaged T gives eigenvalues lambda1 >= lambda2 >= lambda3
 * (squared force magnitudes along the principal axes) and, from the leading
 * eigenvector, the preferred force direction (dir_x, dir_y, dir_z):
 *   - anisotropy  = sqrt(lambda1) / sqrt(lambda3)   (aspect ratio)
 *   - prolateness = (lambda1 - lambda2) / (lambda1 - lambda3)  in [0, 1]
 *
 * The force model (WCA + cosine attraction + bond + bending) lives in
 * func/forces.hpp, shared with the rest of the suite. This tool uses a harmonic
 * bond (kg::BOND_*) to match its original calibration; switch the single call
 * below to feneBondForce() to use the FENE bond instead.
 *
 * Neighbour search uses a flat linked-cell list, so the cost is O(N) per frame
 * with no per-query heap allocation. Parallelism (OpenMP): the non-bonded loop
 * is one-sided (each particle sums forces from its neighbours, writing only to
 * itself) so it is data-race-free; bonded forces are parallelised over chains,
 * which are disjoint; the final per-particle diagonalisation is independent.
 *
 * Output CSV columns:
 *   particle_id, fx_avg, fy_avg, fz_avg, fmag_avg,
 *   lambda1, lambda2, lambda3, anisotropy, prolateness, dir_x, dir_y, dir_z
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utility.hpp"
#include "forces.hpp"

namespace md {

// ═══════════════════════════════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════════════════════════════

struct ForceEllipsoidConfig {
    std::string filenamePrefix;
    int    frameStart        {0};
    int    frameStop         {10};
    int    frameStride       {1};
    int    Nm                {500};
    int    Nc                {1000};
    int    N_s               {0};

    // Force-model parameters (Kremer-Grest defaults, see func/forces.hpp)
    double epsilon_pp        {kg::EPSILON};
    double sigma_pp          {kg::SIGMA_PP};
    double alpha_pp          {kg::ALPHA};
    double epsilon_ps        {kg::EPSILON};
    double sigma_ps          {kg::SIGMA_PS};
    double alpha_ps          {kg::ALPHA};
    double k_bond            {kg::BOND_K};
    double r0_bond           {kg::BOND_R0};
    double a_theta           {kg::BEND_A};
    double b_theta           {kg::BEND_B};
    int    threads           {1};
    std::string outputStem   {"interval"};
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Internal data types
// ═══════════════════════════════════════════════════════════════════════════════

namespace detail {

struct FParticle {
    double x, y, z;
    double fx{}, fy{}, fz{};
};

/**
 * @brief Flat linked-cell list for O(N) neighbour search.
 *
 * Cells are stored contiguously; membership is a head[cell] -> next[particle]
 * singly linked list, so building and querying allocate nothing per particle
 * and stay cache-friendly. The list is built once per frame and only read
 * during force evaluation, so concurrent queries are thread-safe.
 */
class CellList {
public:
    void build(const std::vector<FParticle>& p, int begin, int end,
               double Lx, double Ly, double Lz, double cutoff) {
        p_  = &p;
        cs_ = cutoff;
        nx_ = std::max(1, static_cast<int>(Lx / cs_));
        ny_ = std::max(1, static_cast<int>(Ly / cs_));
        nz_ = std::max(1, static_cast<int>(Lz / cs_));
        head_.assign(static_cast<std::size_t>(nx_) * ny_ * nz_, -1);
        next_.assign(p.size(), -1);
        for (int i = begin; i < end; ++i) {
            const int c = cellOf(p[i]);
            next_[i] = head_[c];
            head_[c] = i;
        }
    }

    /// Visit every neighbour particle index exactly once (dedups the 27-cell
    /// stencil so small boxes with < 3 cells per axis are still correct).
    template <class Fn>
    void forEachNeighbour(const FParticle& q, Fn&& fn) const {
        const int ix = cellCoord(q.x, nx_);
        const int iy = cellCoord(q.y, ny_);
        const int iz = cellCoord(q.z, nz_);

        int visited[27];
        int nv = 0;
        for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
            const int c = cellIndex(wrap(ix+dx, nx_), wrap(iy+dy, ny_), wrap(iz+dz, nz_));
            bool seen = false;
            for (int v = 0; v < nv; ++v) if (visited[v] == c) { seen = true; break; }
            if (seen) continue;
            visited[nv++] = c;
            for (int j = head_[c]; j != -1; j = next_[j]) fn(j);
        }
    }

private:
    static int wrap(int i, int n) { i %= n; return i < 0 ? i + n : i; }
    int cellCoord(double x, int n) const { return wrap(static_cast<int>(std::floor(x/cs_)), n); }
    int cellIndex(int ix, int iy, int iz) const { return (ix*ny_ + iy)*nz_ + iz; }
    int cellOf(const FParticle& q) const {
        return cellIndex(cellCoord(q.x, nx_), cellCoord(q.y, ny_), cellCoord(q.z, nz_));
    }

    const std::vector<FParticle>* p_ {nullptr};
    double cs_ {1.0};
    int nx_ {1}, ny_ {1}, nz_ {1};
    std::vector<int> head_;
    std::vector<int> next_;
};

// ── Per-frame net forces on the polymer monomers ──────────────────────────────

inline std::vector<Vec3> processFrame(
    const std::vector<float>& rx,
    const std::vector<float>& ry,
    const std::vector<float>& rz,
    int N, float Lx, float Ly, float Lz,
    const ForceEllipsoidConfig& cfg)
{
    const int N_p   = cfg.Nm * cfg.Nc;
    const int NpEff = std::min(N_p, N);

    std::vector<FParticle> P(N);
    for (int i = 0; i < N; ++i) { P[i].x = rx[i]; P[i].y = ry[i]; P[i].z = rz[i]; }

    const double rCut_pp = wcaCutoff(cfg.sigma_pp);
    const double rCut_ps = wcaCutoff(cfg.sigma_ps);
    const double maxCut  = std::max(rCut_pp, rCut_ps) * std::sqrt(2.0);

    CellList clPoly, clSolv;
    clPoly.build(P, 0,   N_p, Lx, Ly, Lz, maxCut);
    clSolv.build(P, N_p, N,   Lx, Ly, Lz, maxCut);

    // ── Non-bonded forces (one-sided ⇒ race-free, assigned first) ─────────────
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int i = 0; i < NpEff; ++i) {
        const FParticle& pi = P[i];
        double fx = 0.0, fy = 0.0, fz = 0.0;

        clPoly.forEachNeighbour(pi, [&](int j) {              // polymer-polymer
            if (j == i) return;
            Vec3 dr{P[j].x - pi.x, P[j].y - pi.y, P[j].z - pi.z};
            minimumImage(dr, Lx, Ly, Lz);
            const Vec3 f = nonbondedForce(dr, cfg.epsilon_pp, cfg.sigma_pp, cfg.alpha_pp);
            fx += f[0]; fy += f[1]; fz += f[2];
        });
        clSolv.forEachNeighbour(pi, [&](int j) {              // polymer-solvent
            Vec3 dr{P[j].x - pi.x, P[j].y - pi.y, P[j].z - pi.z};
            minimumImage(dr, Lx, Ly, Lz);
            const Vec3 f = nonbondedForce(dr, cfg.epsilon_ps, cfg.sigma_ps, cfg.alpha_ps);
            fx += f[0]; fy += f[1]; fz += f[2];
        });
        P[i].fx = fx; P[i].fy = fy; P[i].fz = fz;
    }

    // ── Bonded forces: parallelise over chains (disjoint ⇒ race-free) ─────────
    //  Uses the harmonic bond to match this tool's original calibration; swap
    //  harmonicBondForce -> feneBondForce here to use the FENE bond instead.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < cfg.Nc; ++c) {
        const int base = c * cfg.Nm;
        for (int j = 0; j < cfg.Nm - 1; ++j) {
            FParticle& a = P[base + j];
            FParticle& b = P[base + j + 1];
            const Vec3 f = harmonicBondForce({b.x-a.x, b.y-a.y, b.z-a.z},
                                             cfg.k_bond, cfg.r0_bond);
            a.fx += f[0]; a.fy += f[1]; a.fz += f[2];
            b.fx -= f[0]; b.fy -= f[1]; b.fz -= f[2];
        }
        for (int j = 1; j < cfg.Nm - 1; ++j) {
            FParticle& pi = P[base + j - 1];
            FParticle& pj = P[base + j];
            FParticle& pk = P[base + j + 1];
            const BendForces bf = bendingForces({pi.x,pi.y,pi.z}, {pj.x,pj.y,pj.z},
                                                {pk.x,pk.y,pk.z}, cfg.a_theta, cfg.b_theta);
            pi.fx += bf.fi[0]; pi.fy += bf.fi[1]; pi.fz += bf.fi[2];
            pj.fx += bf.fj[0]; pj.fy += bf.fj[1]; pj.fz += bf.fj[2];
            pk.fx += bf.fk[0]; pk.fy += bf.fk[1]; pk.fz += bf.fk[2];
        }
    }

    std::vector<Vec3> forces(N_p, Vec3{0.0, 0.0, 0.0});
    for (int i = 0; i < NpEff; ++i) forces[i] = {P[i].fx, P[i].fy, P[i].fz};
    return forces;
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════════════
//  Public entry point
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Compute per-monomer force ellipsoids over a range of frames.
 * @param cfg  Run parameters.
 */
inline void computeForceEllipsoid(const ForceEllipsoidConfig& cfg) {
    std::cout << "===== Starting Force Ellipsoid Analysis =====\n";
    const auto wallStart = std::chrono::steady_clock::now();

#ifdef _OPENMP
    omp_set_num_threads(cfg.threads);
    std::cout << "  OpenMP threads: " << cfg.threads << "\n";
#endif

    const int N_p = cfg.Nm * cfg.Nc;

    std::vector<double> sumFx(N_p, 0.0), sumFy(N_p, 0.0), sumFz(N_p, 0.0);
    std::vector<double> sumT[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            sumT[a][b].assign(N_p, 0.0);

    int frameCount = 0;

    for (int frame = cfg.frameStart; frame <= cfg.frameStop; frame += cfg.frameStride) {
        Frame fr;
        try {
            fr = loadFrame(cfg.filenamePrefix, frame, cfg.Nm, cfg.Nc, UnwrapPolicy::None,
                           cfg.N_s > 0 ? Species::Both : Species::Polymer);
        } catch (const std::exception& e) {
            std::cerr << "  Skipping frame " << frame << ": " << e.what() << "\n";
            continue;
        }
        auto& rx = fr.rx; auto& ry = fr.ry; auto& rz = fr.rz;
        const int N = fr.N;
        const float Lx = fr.Lx, Ly = fr.Ly, Lz = fr.Lz;

        const auto forces = detail::processFrame(rx, ry, rz, N, Lx, Ly, Lz, cfg);

        for (int i = 0; i < N_p; ++i) {
            const Vec3& f = forces[i];
            sumFx[i] += f[0]; sumFy[i] += f[1]; sumFz[i] += f[2];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    sumT[a][b][i] += f[a] * f[b];
        }
        ++frameCount;
        printProgress("ForceEllipsoid", cfg.frameStop + 1, frame);
    }
    std::cout << "\n100%  (" << frameCount << " frames)\n";
    if (frameCount == 0) { std::cerr << "No frames processed.\n"; return; }

    // ── Diagonalise the time-averaged tensor and write output ─────────────────
    const std::string outFile = cfg.outputStem + "_ellipsoid.csv";
    std::ofstream fo(outFile);
    fo << "particle_id,fx_avg,fy_avg,fz_avg,fmag_avg,"
          "lambda1,lambda2,lambda3,anisotropy,prolateness,dir_x,dir_y,dir_z\n";

    for (int i = 0; i < N_p; ++i) {
        const double fxA = sumFx[i] / frameCount;
        const double fyA = sumFy[i] / frameCount;
        const double fzA = sumFz[i] / frameCount;
        const double fmag = std::sqrt(fxA*fxA + fyA*fyA + fzA*fzA);

        double T[3][3];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                T[a][b] = sumT[a][b][i] / frameCount;

        const Vec3 ev  = eigenvalues3(T);
        const double l1 = std::max(0.0, ev[0]);
        const double l2 = std::max(0.0, ev[1]);
        const double l3 = std::max(0.0, ev[2]);
        const Vec3 dir  = eigenvector3(T, ev[0]);       // preferred force direction

        const double anis = (l3 > 1e-30)      ? std::sqrt(l1) / std::sqrt(l3) : 0.0;
        const double prol = (l1 - l3 > 1e-30) ? (l1 - l2) / (l1 - l3)         : 0.0;

        fo << i << ',' << fxA << ',' << fyA << ',' << fzA << ',' << fmag << ','
           << l1 << ',' << l2 << ',' << l3 << ','
           << anis << ',' << prol << ','
           << dir[0] << ',' << dir[1] << ',' << dir[2] << '\n';
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    std::cout << "Output: " << outFile << "\n"
              << "Wall time Force Ellipsoid: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd-wallStart).count()
              << " ms\n";
}

} // namespace md
