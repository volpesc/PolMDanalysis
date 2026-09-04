#pragma once

/**
 * Core data structures, I/O utilities, and physics helper functions
 *        for molecular dynamics trajectory analysis.
 *
 * Design notes:
 *  - No raw-pointer ownership: callers use std::vector<> or smart pointers.
 *  - No "using namespace std" to avoid name collisions in larger projects.
 *  - Helper free functions are placed in the `md` namespace.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── Compile-time constants ────────────────────────────────────────────────────
static constexpr double PI              = 3.14159265358979323846;

// ── Convenience macros (kept for legacy call-sites; prefer inline functions) ──
#define SQR(x) ((x)*(x))
#define CUB(x) ((x)*(x)*(x))

namespace md {

// ── Data structures ───────────────────────────────────────────────────────────

/// Symmetric 3×3 stress tensor stored in Voigt order: xx yy zz xy xz yz
struct Tensor {
    std::array<double, 6> ij{};
};

/// Point particle carrying position and velocity.
struct Particle {
    std::array<double, 3> pos{};
    std::array<double, 3> vel{};
};

/// Wave-vector (single-precision to match FFT conventions).
struct WaveVector {
    float x{}, y{}, z{};
};

// ── Geometry helpers ──────────────────────────────────────────────────────────

/**
 * @brief Apply minimum-image convention to a displacement component.
 * @param dx   Displacement (modified in place).
 * @param box  Periodic box length along that axis.
 */
inline void applyMinimumImage(float& dx, float box) noexcept {
    if      (dx >  0.5f * box) dx -= box;
    else if (dx < -0.5f * box) dx += box;
}

/**
 * @brief Minimum-imaged displacement vector r[b] - r[a], all three axes periodic.
 *
 * Shared by every tool that needs a bond vector or other short displacement
 * between two particle indices in the same frame (bulk geometry: x, y, and z
 * all periodic).
 */
inline void minImageVec(const std::vector<float>& rx, const std::vector<float>& ry,
                        const std::vector<float>& rz, int a, int b,
                        float Lx, float Ly, float Lz,
                        float& dx, float& dy, float& dz) noexcept {
    dx = rx[b]-rx[a]; applyMinimumImage(dx, Lx);
    dy = ry[b]-ry[a]; applyMinimumImage(dy, Ly);
    dz = rz[b]-rz[a]; applyMinimumImage(dz, Lz);
}

/**
 * @brief Minimum-imaged displacement vector r[b] - r[a], y/z periodic, x raw.
 *
 * For tools that treat x as the non-periodic layer-normal direction
 * (interface/slab geometry, matching pressure_z_mpi.hpp's convention).
 */
inline void minImageVecYZ(const std::vector<float>& rx, const std::vector<float>& ry,
                          const std::vector<float>& rz, int a, int b,
                          float Ly, float Lz,
                          float& dx, float& dy, float& dz) noexcept {
    dx = rx[b]-rx[a];
    dy = ry[b]-ry[a]; applyMinimumImage(dy, Ly);
    dz = rz[b]-rz[a]; applyMinimumImage(dz, Lz);
}

/// Squared magnitude of a wave-vector.
inline float wavevectorNormSq(const WaveVector* wv, int i) noexcept {
    return wv[i].x * wv[i].x + wv[i].y * wv[i].y + wv[i].z * wv[i].z;
}

/**
 * @brief Irving–Kirkwood contour weight H(x_i, x_j, x).
 *
 * Following the layer-pressure definition (H = (1/|x_ij|)[θ(x_j-x)θ(x-x_i) +
 * θ(x_i-x)θ(x-x_j)]), this returns 1/|x_i - x_j| when the plane at x lies
 * strictly between x_i and x_j (the i–j segment crosses the plane) and 0
 * otherwise. x is the layer-normal direction and is not minimum-imaged.
 */
inline double heaviside(double xi, double xj, double x) noexcept {
    if ((xj > x && x > xi) || (xi > x && x > xj))
        return 1.0 / std::fabs(xj - xi);
    return 0.0;
}

// ── Filename helpers ──────────────────────────────────────────────────────────

inline std::string makeInputFilename(const std::string& prefix, int idx) {
    return prefix + std::to_string(idx) + ".xyz";
}

inline std::string makeOutputFilename(const std::string& prefix, int idx) {
    return prefix + std::to_string(idx) + ".dat";
}

// ── Progress reporting ────────────────────────────────────────────────────────

inline void printProgress(const std::string& tag, int total, int current) {
    const int pct = static_cast<int>(current * 100.0 / total);
    if (pct % 5 == 0) {
        std::cout << tag << " progress: " << pct << " %\r";
        std::cout.flush();
    }
}

// ── Pressure-tensor contributions ────────────────────────────────────────────

/// Kinetic (Irving–Kirkwood) contribution for polymer particles.
inline double kineticPressure_pp(double A, double x, double deltaX,
                                  const std::vector<Particle>& p,
                                  int N_p, int alpha, int beta) noexcept {
    double sum = 0.0;
    for (int i = 0; i < N_p; ++i)
        if (p[i].pos[0] >= x - deltaX*0.5 && p[i].pos[0] <= x + deltaX*0.5)
            sum += p[i].vel[alpha] * p[i].vel[beta];
    return sum / (A * deltaX);
}

/// Kinetic contribution for solvent particles.
inline double kineticPressure_ss(double A, double x, double deltaX,
                                  const std::vector<Particle>& p,
                                  int N_p, int N_s, int alpha, int beta) noexcept {
    double sum = 0.0;
    for (int i = N_p; i < N_p + N_s; ++i)
        if (p[i].pos[0] >= x - deltaX*0.5 && p[i].pos[0] <= x + deltaX*0.5)
            sum += p[i].vel[alpha] * p[i].vel[beta];
    return sum / (A * deltaX);
}

// ── I/O ───────────────────────────────────────────────────────────────────────

/**
 * @brief Which atoms to load from a frame.
 *
 * Atoms always keep their absolute file indices: polymer occupies
 * [0, Nm*Nc) and solvent occupies [Nm*Nc, N), regardless of the scope
 * requested, so callers index the two blocks the same way in every case.
 */
enum class Species { Polymer, Solvent, Both };

/**
 * @brief Read positions from an XYZ-format trajectory frame.
 *
 * File layout: a header "N Lx Ly Lz", then N atom lines
 * "id type x y z vx vy vz", with the Nm*Nc polymer atoms first and any
 * solvent atoms after them.
 *
 * The position arrays are always sized to the full frame (N) and atoms are
 * stored at their absolute indices; @p which only selects which block(s) are
 * actually read from disk. Unrequested atoms are left at the origin, so a
 * polymer-only tool never pays to parse a large solvent block, while a
 * solvent-aware tool can request Species::Both and index solvent at [Nm*Nc, N).
 *
 * @param which  Polymer (default), Solvent, or Both.
 * @throws std::runtime_error if the file cannot be opened, has a bad header,
 *         or ends before the requested atoms are read.
 */
inline void readFrame(const std::string& filename,
                      int Nm, int Nc,
                      std::vector<float>& rx, std::vector<float>& ry, std::vector<float>& rz,
                      int& N, float& Lx, float& Ly, float& Lz,
                      Species which = Species::Polymer) {
    std::ifstream fi(filename);
    if (!fi) throw std::runtime_error("readFrame(): cannot open " + filename);

    fi >> N >> Lx >> Ly >> Lz;
    if (!fi || N < 0)
        throw std::runtime_error("readFrame(): bad header in " + filename);

    rx.assign(N, 0.f); ry.assign(N, 0.f); rz.assign(N, 0.f);

    const int Np    = std::min(Nm * Nc, N);                  // polymer atoms present
    const int begin = (which == Species::Solvent) ? Np : 0;  // first atom to store
    const int end   = (which == Species::Polymer) ? Np : N;  // one past the last

    std::string skip;
    for (int i = 0; i < begin; ++i)                          // advance past unread atoms
        fi >> skip >> skip >> skip >> skip >> skip >> skip >> skip >> skip;

    for (int i = begin; i < end; ++i) {
        fi >> skip >> skip >> rx[i] >> ry[i] >> rz[i] >> skip >> skip >> skip;
        if (!fi)
            throw std::runtime_error("readFrame(): unexpected end of " + filename);
    }
}

/// Read only the header of an XYZ file (N, box dimensions).
inline void readHeader(const std::string& filename,
                       int& N, float& Lx, float& Ly, float& Lz) {
    std::ifstream fi(filename);
    if (!fi) throw std::runtime_error("readHeader(): cannot open " + filename);
    fi >> N >> Lx >> Ly >> Lz;
}

/// Read particle positions and velocities (7-column format: id type x y z vx vy vz).
/// Reads every atom line (polymer first, then solvent) after the single header line.
inline std::vector<Particle> readParticles(const std::string& filename) {
    std::ifstream fi(filename);
    if (!fi) throw std::runtime_error("readParticles(): cannot open " + filename);

    std::vector<Particle> particles;
    std::string line;
    std::getline(fi, line);            // skip the single "N Lx Ly Lz" header line
    while (std::getline(fi, line)) {
        std::istringstream iss(line);
        int id, type;
        Particle p;
        if (iss >> id >> type
                >> p.pos[0] >> p.pos[1] >> p.pos[2]
                >> p.vel[0] >> p.vel[1] >> p.vel[2])
            particles.push_back(p);    // ignore blank / malformed trailing lines
    }
    return particles;
}

// ── Center-of-mass ────────────────────────────────────────────────────────────

/// Per-chain center of mass (no PBC).
inline void computeCoM(int Nm, int Nc,
                        const std::vector<float>& rx,
                        const std::vector<float>& ry,
                        const std::vector<float>& rz,
                        std::vector<float>& cx,
                        std::vector<float>& cy,
                        std::vector<float>& cz) {
    cx.assign(Nc, 0.f); cy.assign(Nc, 0.f); cz.assign(Nc, 0.f);
    for (int j = 0; j < Nc; ++j) {
        for (int i = 0; i < Nm; ++i) {
            const int pid = j*Nm+i;
            cx[j] += rx[pid]; cy[j] += ry[pid]; cz[j] += rz[pid];
        }
        cx[j] /= Nm; cy[j] /= Nm; cz[j] /= Nm;
    }
}

/// Per-chain center of mass with periodic boundary conditions.
///
/// Each chain is unwrapped bond-by-bond starting from its first monomer: the
/// minimum-image bond vector is added to a running position, so a chain that
/// straddles a periodic boundary yields a contiguous CoM instead of one smeared
/// across the box. All Nm monomers (including monomer 0) contribute, and the
/// final CoM is wrapped back into the primary cell [0, L) so it is a valid
/// in-box coordinate.
inline void computeCoMPBC(int Nm, int Nc,
                           const std::vector<float>& rx,
                           const std::vector<float>& ry,
                           const std::vector<float>& rz,
                           std::vector<float>& cx,
                           std::vector<float>& cy,
                           std::vector<float>& cz,
                           float Lx, float Ly, float Lz) {
    cx.assign(Nc, 0.f); cy.assign(Nc, 0.f); cz.assign(Nc, 0.f);
    for (int j = 0; j < Nc; ++j) {
        const int base = j*Nm;
        // Running unwrapped position, seeded at monomer 0, which is also the
        // first term of the sum.
        double ux = rx[base], uy = ry[base], uz = rz[base];
        double sx = ux, sy = uy, sz = uz;
        for (int i = 1; i < Nm; ++i) {
            const int pid = base+i;
            float dx = rx[pid]-rx[pid-1], dy = ry[pid]-ry[pid-1], dz = rz[pid]-rz[pid-1];
            applyMinimumImage(dx, Lx);
            applyMinimumImage(dy, Ly);
            applyMinimumImage(dz, Lz);
            ux += dx; uy += dy; uz += dz;   // running unwrapped position of monomer i
            sx += ux; sy += uy; sz += uz;
        }
        sx /= Nm; sy /= Nm; sz /= Nm;
        // Wrap the CoM back into the primary cell [0, L).
        cx[j] = static_cast<float>(sx - Lx*std::floor(sx/Lx));
        cy[j] = static_cast<float>(sy - Ly*std::floor(sy/Ly));
        cz[j] = static_cast<float>(sz - Lz*std::floor(sz/Lz));
    }
}

/// System-level center of mass.
inline void computeSystemCoM(int Nm, int Nc,
                              const std::vector<float>& rx,
                              const std::vector<float>& ry,
                              const std::vector<float>& rz,
                              int N,
                              float& scx, float& scy, float& scz) {
    scx = scy = scz = 0.f;
    for (int j = 0; j < Nc; ++j)
        for (int i = 0; i < Nm; ++i) {
            const int pid = j*Nm+i;
            scx += rx[pid]; scy += ry[pid]; scz += rz[pid];
        }
    scx /= N; scy /= N; scz /= N;
}

/**
 * @brief Unwrap each chain independently (bond-by-bond within the chain).
 *
 * The polymer occupies [0, Nm*Nc). The walk resets at every chain boundary, so
 * the last monomer of one chain is never linked to the first monomer of the
 * next. After this call, intra-chain displacements r[j+s]-r[j] are correct
 * (no minimum-image wrap needed) up to the chain's true extent. Solvent, if any,
 * is untouched. rx,ry,rz are modified in place.
 */
inline void unwrapChains(std::vector<float>& rx, std::vector<float>& ry,
                         std::vector<float>& rz,
                         int Nm, int Nc, int N, float Lx, float Ly, float Lz) {
    const int Np = std::min(Nm * Nc, N);
    for (int c = 0; c < Nc; ++c) {
        const int base = c * Nm;
        if (base >= Np) break;
        const int last = std::min(base + Nm, Np);
        for (int i = base + 1; i < last; ++i) {
            // Shift by whatever whole number of boxes brings monomer i next to
            // its already-unwrapped predecessor. A single ±L step is not enough
            // once a chain has unwrapped past one box length (its coordinate can
            // then sit two or more boxes away from the previous monomer).
            rx[i] -= Lx * std::round((rx[i] - rx[i-1]) / Lx);
            ry[i] -= Ly * std::round((ry[i] - ry[i-1]) / Ly);
            rz[i] -= Lz * std::round((rz[i] - rz[i-1]) / Lz);
        }
    }
}

/**
 * @brief Make a multi-chain polymer contiguous under PBC and return its CoM.
 *
 * Each chain is unwrapped within itself (see unwrapChains), then shifted by whole
 * box vectors so its centre of mass is the minimum image of the first chain's
 * centre of mass, giving a single contiguous cluster whose centre of mass is well
 * defined even when chains sit in different periodic images.
 *
 * rx,ry,rz are modified in place over the polymer region only; solvent (if any,
 * at [Nm*Nc, N)) is untouched. All three axes are treated as periodic, which is
 * the appropriate convention for a compact cluster/globule (valid up to a
 * cluster radius of ~L/2).
 *
 * @return the polymer centre of mass via comX,comY,comZ.
 */
inline void unwrapPolymerCluster(std::vector<float>& rx, std::vector<float>& ry,
                                 std::vector<float>& rz,
                                 int Nm, int Nc, int N,
                                 float Lx, float Ly, float Lz,
                                 double& comX, double& comY, double& comZ) {
    const int Np = std::min(Nm * Nc, N);
    auto boxRound = [](double d, double L) { return L * std::round(d / L); };

    unwrapChains(rx, ry, rz, Nm, Nc, N, Lx, Ly, Lz);   // make each chain contiguous

    double refX = 0.0, refY = 0.0, refZ = 0.0;
    bool   haveRef = false;
    double gX = 0.0, gY = 0.0, gZ = 0.0;
    int    counted = 0;

    for (int c = 0; c < Nc; ++c) {
        const int base = c * Nm;
        if (base >= Np) break;
        const int last = std::min(base + Nm, Np);

        double ccx = 0.0, ccy = 0.0, ccz = 0.0;
        const int n = last - base;
        for (int i = base; i < last; ++i) { ccx += rx[i]; ccy += ry[i]; ccz += rz[i]; }
        ccx /= n; ccy /= n; ccz /= n;

        if (!haveRef) { refX = ccx; refY = ccy; refZ = ccz; haveRef = true; }

        const double sx = boxRound(ccx - refX, Lx);
        const double sy = boxRound(ccy - refY, Ly);
        const double sz = boxRound(ccz - refZ, Lz);
        for (int i = base; i < last; ++i) {
            rx[i] -= static_cast<float>(sx);
            ry[i] -= static_cast<float>(sy);
            rz[i] -= static_cast<float>(sz);
            gX += rx[i]; gY += ry[i]; gZ += rz[i];
        }
        counted += n;
    }

    if (counted > 0) { gX /= counted; gY /= counted; gZ /= counted; }
    comX = gX; comY = gY; comZ = gZ;
}

// ── Frame loading with an explicit unwrap policy ──────────────────────────────

/**
 * @brief How a loaded frame's polymer coordinates should be treated for PBC.
 *
 * Picking one of these (instead of hand-calling readFrame + an unwrap helper
 * at every call site) makes the unwrap treatment an explicit, visible part of
 * each tool's frame-loading call, rather than something a tool can silently
 * omit. See the corresponding free functions above for what each one does.
 */
enum class UnwrapPolicy {
    None,          ///< Raw coordinates as read from the file (still PBC-wrapped).
    ChainOnly,     ///< unwrapChains(): contiguous per chain, chains not imaged
                   ///< against each other. For intra-chain quantities (bond
                   ///< vectors, internal distances) at bin/chain-boundary scale.
    ChainCluster,  ///< unwrapPolymerCluster(): contiguous per chain AND all
                   ///< chains imaged onto one cluster with a well-defined CoM.
                   ///< For whole-polymer quantities (Rg, radial density) in a
                   ///< compact-globule geometry.
};

/// One loaded (and, per @p policy, PBC-unwrapped) trajectory frame.
struct Frame {
    int   N{};
    float Lx{}, Ly{}, Lz{};
    std::vector<float> rx, ry, rz;
    // Populated only by UnwrapPolicy::ChainCluster; zero otherwise.
    double comX{}, comY{}, comZ{};
};

/**
 * @brief Read one frame and apply the requested unwrap policy.
 *
 * Equivalent to readFrame() followed by the unwrap helper @p policy names;
 * centralising it here means a tool's frame-loading call site states its PBC
 * treatment once, instead of a raw readFrame() that silently defaults to "no
 * unwrap" (the mistake behind an early MSD bug: a tool loaded raw, still
 * PBC-wrapped coordinates and never unwrapped them at all).
 *
 * @param prefix,idx,Nm,Nc,which  Forwarded to readFrame().
 * @param policy  Unwrap treatment to apply after reading (default: none).
 */
inline Frame loadFrame(const std::string& prefix, int idx, int Nm, int Nc,
                        UnwrapPolicy policy = UnwrapPolicy::None,
                        Species which = Species::Polymer) {
    Frame f;
    readFrame(makeInputFilename(prefix, idx), Nm, Nc,
              f.rx, f.ry, f.rz, f.N, f.Lx, f.Ly, f.Lz, which);

    switch (policy) {
        case UnwrapPolicy::None:
            break;
        case UnwrapPolicy::ChainOnly:
            unwrapChains(f.rx, f.ry, f.rz, Nm, Nc, f.N, f.Lx, f.Ly, f.Lz);
            break;
        case UnwrapPolicy::ChainCluster:
            unwrapPolymerCluster(f.rx, f.ry, f.rz, Nm, Nc, f.N, f.Lx, f.Ly, f.Lz,
                                  f.comX, f.comY, f.comZ);
            break;
    }
    return f;
}

} // namespace md
