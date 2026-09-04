/**
 * @file analysis.cpp
 * @brief MD Trajectory Analysis Suite - command-line entry point.
 *
 * Usage:
 *   mpirun -np <N> ./analysis --tool <name> [options]
 *   ./analysis --help
 *   ./analysis --help --tool <name>
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j$(nproc)
 *
 * Architecture:
 *   Tools implement the md::Analysis interface (func/analysis_base.hpp) and
 *   self-register with the md::Registry (func/tools.hpp). main() parses the
 *   arguments, looks the requested tool up by name, and runs it - no
 *   compile-time dependency on any concrete tool, so adding a tool needs no
 *   edits here.
 */

#include <iostream>
#include <map>
#include <string>
#include <mpi.h>

#include "func/args.hpp"
#include "func/analysis_base.hpp"
#include "func/tools.hpp"   // defines and registers every tool

// ═══════════════════════════════════════════════════════════════════════════════
//  Help text
// ═══════════════════════════════════════════════════════════════════════════════

static const std::string GLOBAL_HELP = R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║             MD Analysis Suite  —  version 1.2.0                            ║
║             C++17 · MPI · OpenMP                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE
  mpirun -np <N_ranks> ./analysis --tool <name> [options]

GLOBAL OPTIONS
  --tool <name>     Select analysis tool (required)
  --help            Print this message
  --help --tool <n> Tool-specific help

STRUCTURAL / CONFORMATIONAL
  msd               Mean Squared Displacement  g1, g2, g3
  gyr               Gyration radius profile  <Rg²(x)>
  msid              Mean Squared Internal Distance  C(s)
  endtoend          End-to-end distance  <Re²> and P(Re²)
  bondangle         Bond angle distribution  P(theta)
  volume            Gyration volume  Rg, V_eq

SCATTERING / DENSITY
  sq                Static structure factor  S(q)
  rdf               Radial distribution function  g(r)
  density           Spatial density profile  rho(x)
  specdensity       Species radial density  rho(r)

MECHANICS / ENERGY
  backbone          FENE bond-force profile along backbone
  energy            Polymer-solvent interaction energy  E(lambda)
  pressure          Irving-Kirkwood layer pressure tensor  P(x)

ENTANGLEMENT
  ppa               Primitive Path Analysis  bpp, app, Ne
  entanglement      Spatial kink/entanglement density from PPA output

COMMON OPTIONS
  --prefix  <str>   Trajectory prefix          [default: requil_]
  --Nm      <int>   Monomers per chain          [default: 10]
  --Nc      <int>   Number of chains            [default: 10000]
  --start   <int>   First frame                 [default: 0]
  --stop    <int>   Last frame                  [default: 400]
  --step    <int>   Frame stride                [default: 1]
  --frame   <int>   Single frame index          [default: 0]
  --out     <str>   Output file                 [default: tool-specific]
)";

static const std::map<std::string, std::string> TOOL_HELP = {

{"msd", R"(
TOOL: msd  —  Mean Squared Displacement
  g1(t): monomer MSD (lab frame, inner 50% of chain)
  g2(t): monomer MSD relative to chain CoM
  g3(t): chain CoM MSD
  MPI-parallelized over time origins.
OPTIONS: --prefix --Nm --Nc --start --stop --step --out
  --dt <float>   Time between frames [tau]   [default: 50.0]
OUTPUT:  t  g1  g2  g3
EXAMPLE: mpirun -np 4 ./analysis --tool msd --Nm 50 --Nc 200 --start 0 --stop 1000
)"},

{"gyr", R"(
TOOL: gyr  —  Gyration Radius Profile <Rg²(x)>
  Chains binned by PBC-corrected CoM x-position.
OPTIONS: --prefix --Nm --Nc --frame --out
  --binw <float>  Bin width [sigma]   [default: 3.0]
  --nbins <int>   Number of bins      [default: 62]
OUTPUT:  x  Rg2_x  Rg2_y  Rg2_z  Rg2_total
)"},

{"msid", R"(
TOOL: msid  —  Mean Squared Internal Distance C(s)
  C(s) = <R2(s)> / (s * lb^2).  FJC limit: C(s) = 1.
OPTIONS: --prefix --Nm --Nc --start --stop --step --out
  --lb <float>  Bond length [sigma]  [default: 0.964]
OUTPUT:  s  C(s)
)"},

{"endtoend", R"(
TOOL: endtoend  —  End-to-End Distance
  Bond-by-bond PBC-safe reconstruction of Re.
  Outputs time series <Re2> and histogram P(Re2).
OPTIONS: --prefix --Nm --Nc --start --stop --step --out
  --nbins <int>  Histogram bins  [default: 100]
OUTPUT FILES:  <stem>_time.dat   <stem>_hist.dat
)"},

{"bondangle", R"(
TOOL: bondangle  —  Bond Angle Distribution P(theta)
  Triplets i,i+1,i+2. Reports P(theta) and P(theta)*sin(theta).
OPTIONS: --prefix --Nm --Nc --start --stop --step --out
  --nbins <int>  Bins over [0,180 deg]  [default: 180]
OUTPUT:  theta[deg]  P(theta)  P(theta)*sin(theta)
)"},

{"volume", R"(
TOOL: volume  —  Gyration Volume
  Rg of the whole system and equivalent sphere volume V = (4/3) pi Rg^3.
  Coordinates unwrapped bond-by-bond before CoM.
OPTIONS: --prefix --Nm --Nc --frame --out
OUTPUT:  Rg  V_eq
EXAMPLE: ./analysis --tool volume --frame 100 --Nm 50 --Nc 200
)"},

{"sq", R"(
TOOL: sq  —  Static Structure Factor S(q)
  Modes: --mode sc (single-chain) or tot (total)
  Sampling: --sampling spherical (default) or cartesian
OPTIONS: --prefix --Nm --Nc --frame --start --stop --out
  --mode <str>      sc | tot           [default: sc]
  --sampling <str>  spherical|cartesian [default: spherical]
  --nkbound <int>   shells/grid bound  [default: 100]
  --dkmult  <float> dk multiplier      [default: 1.0]
  --bintol  <float> binning tolerance  [default: 0.1]
OUTPUT:  q  S(q)
)"},

{"rdf", R"(
TOOL: rdf  —  Radial Distribution Function g(r)
  All-pairs with MIC. g(r) -> 1 at large r.
OPTIONS: --prefix --Nm --Nc --start --stop --step --out
  --rcut  <float>  Max distance [sigma]  [default: 5.0]
  --nbins <int>    Histogram bins        [default: 200]
OUTPUT:  r  g(r)
)"},

{"density", R"(
TOOL: density  —  Density Profile rho(x)
  rho(x) = <N(x)> / (A * dx). Separate polymer/solvent when --Ns > 0.
OPTIONS: --prefix --Nm --Nc --start --stop --step --out
  --Ns    <int>  Solvent particles  [default: 0]
  --nbins <int>  Bins along x       [default: 100]
OUTPUT:  x  rho_polymer  [rho_solvent]
)"},

{"specdensity", R"(
TOOL: specdensity  —  Species Radial Density rho(r)
  Volume-fraction in shells around polymer CoM.
  sigma_monomer=1.0, sigma_solvent=0.75.
OPTIONS: --prefix --Nm --Nc --frame --out
  --Ns      <int>    Solvent particles  [default: 0]
  --nshells <int>    Radial shells      [default: 18]
  --shellw  <float>  Shell width [sigma] [default: 1.0]
OUTPUT:  r  rho_polymer  rho_solvent
EXAMPLE: ./analysis --tool specdensity --frame 50 --Nm 50 --Nc 200 --Ns 5000
)"},

{"backbone", R"(
TOOL: backbone  —  FENE Bond-Force Profile
  Projected FENE force per bond index along the backbone.
  FENE: K=30, R0=1.5 (Kremer-Grest).
OPTIONS: --prefix --Nm --Nc --frame --out
  --bbmode <str>  full (all chains, averaged) | single  [default: full]
  --stretchx/y/z <float>  Stretch direction components  [default: 1 0 0]
OUTPUT:  bond_index  F_proj
EXAMPLE: ./analysis --tool backbone --frame 0 --Nm 50 --Nc 200 --bbmode full
)"},

{"energy", R"(
TOOL: energy  —  Polymer-Solvent Interaction Energy
  WCA + cosine-attractive tail between all polymer-solvent pairs.
  U_att amplitude alpha = 0.5145 * lambda.
OPTIONS: --prefix --Nm --Nc --frame --out
  --Ns     <int>    Solvent particles   [default: 0]
  --lambda <float>  Coupling lambda     [default: 1.0]
OUTPUT:  lambda  E_total  (appended to file if --out given)
EXAMPLE: ./analysis --tool energy --frame 100 --Nm 50 --Nc 200 --Ns 500 --lambda 0.5
)"},

{"pressure", R"(
TOOL: pressure  —  Irving-Kirkwood Layer Pressure Tensor P(x)
  Full symmetric stress tensor in Voigt notation. MPI over layers.
  Surface tension: gamma = 0.5 * integral(PN - PT) dx
OPTIONS: --prefix --Nm --Nc --frame --out
  --Ns     <int>  Solvent particles  [default: 0]
  --nlayer <int>  Layers along x     [default: 120]
OUTPUT FILES:  <stem>_PK_p.txt  <stem>_PK_s.txt
OUTPUT COLS:   x  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz
)"},

{"ppa", R"(
TOOL: ppa  —  Primitive Path Analysis
  Computes bpp (avg bond length), app (Kuhn length), Ne (entanglement length)
  by treating each chain as its own primitive path.
  Ne = app / bpp.  Related to plateau modulus: G_e ~ rho kT / Ne.
OPTIONS: --prefix --Nm --Nc --frame --out
OUTPUT (key=value):  bpp  app  Ne
EXAMPLE: ./analysis --tool ppa --frame 0 --Nm 500 --Nc 200 --out ppa.dat
)"},

{"entanglement", R"(
TOOL: entanglement  —  Spatial Kink / Entanglement Density
  Bins backbone kink angles along x. A kink is a bond angle < threshold.
  Use after PPA to get spatial entanglement density.
OPTIONS: --prefix --Nm --Nc --frame --out
  --Ns      <int>    Solvent particles to skip  [default: 0]
  --nbins   <int>    Spatial bins along x       [default: 100]
  --kinkdeg <float>  Kink angle threshold [deg] [default: 147.0]
  --entmode <str>    chain_com | monomer        [default: chain_com]
                     chain_com: one bin per chain (by CoM x); no double-count.
                     monomer:   each kink binned at its middle monomer.
OUTPUT (chain_com):  x_center  chains_in_bin  total_kinks  avg_kinks_per_chain
OUTPUT (monomer):    x_center  kink_count     chains_in_bin  ent_per_chain
EXAMPLE: ./analysis --tool entanglement --frame 0 --Nm 500 --Nc 200 \
           --nbins 250 --kinkdeg 150.0 --entmode chain_com --out entanglement.dat
)"}

,
{"nematic", R"(
TOOL: nematic  —  Nematic Order Parameter S(x) and Director n̂(x)
  Builds Q-tensor from bond vectors binned along x.
  S=0: isotropic, S=1: perfectly aligned, S~0.5: weakly nematic.
  Chain end-bonds excluded. MIC on y,z.
OPTIONS: --prefix --Nm --Nc --frame --out
  --Ns    <int>  Solvent particles to skip  [default: 0]
  --nbins <int>  Spatial bins along x       [default: 85]
OUTPUT:  x  S  nx  ny  nz
EXAMPLE: ./analysis --tool nematic --frame 0 --Nm 500 --Nc 1000 --nbins 85
)"},
{"gds", R"(
TOOL: gds  —  Gibbs Dividing Surface Diffusion Front
  Tracks GDS front positions of polymer and solvent across frames.
  Also computes uptake and detects induction time.
OPTIONS: --prefix --Nm --Nc --start --stop --out
  --Ns      <int>    Solvent particles      [default: 0]
  --xmin    <float>  Profile x start [sigma] [default: 0.0]
  --xmax    <float>  Profile x end   [sigma] [default: 100.0]
  --binw    <float>  Profile bin width       [default: 3.0]
  --dt      <float>  Time between frames [tau] [default: 1000.0]
  --xthresh <float>  Polymer uptake threshold [default: 45.0]
  --vthresh <float>  Induction velocity thresh [default: 0.01]
OUTPUT:  CSV: time, solvent_front, polymer_front, solvent_uptake, polymer_uptake
EXAMPLE: ./analysis --tool gds --prefix requil_ --Nm 500 --Nc 1000 --Ns 2000000 \
           --start 70 --stop 120 --dt 1000 --out gds.csv
)"},
{"brushlength", R"(
TOOL: brushlength  —  Brush Penetration Length Distribution P(L)
  Finds solvent front (percentile of solvent x-positions), then measures
  how far chains anchored near the box centre extend beyond the front.
OPTIONS: --prefix --Nm --Nc --frame --out
  --Ns         <int>    Solvent particles          [default: 0]
  --percentile <float>  Front percentile           [default: 95.0]
  --cbuffer    <float>  Centre anchor half-width [sigma] [default: 10.0]
  --binw       <float>  Histogram bin width [sigma] [default: 1.0]
  --minbin     <int>    First output bin            [default: 2]
  --maxbin     <int>    Last  output bin            [default: 90]
OUTPUT:  L  P(L)
EXAMPLE: ./analysis --tool brushlength --frame 100 --Nm 500 --Nc 1000 \
           --Ns 2000000 --percentile 95 --out brush_length.dat
)"}
,
{"fellipsoid", R"(
TOOL: fellipsoid  —  Per-Monomer Force Ellipsoid
  Recomputes all forces (WCA, cosine-attractive, bonds, bending) and builds
  the force covariance tensor T_ab = <f_a f_b> for each monomer.
  Diagonalising T gives the force ellipsoid principal axes (lambda1>=lambda2>=lambda3):
    anisotropy  = sqrt(lambda1) / sqrt(lambda3)  (aspect ratio)
    prolateness = (lambda1 - lambda2) / (lambda1 - lambda3)  in [0,1]
  Uses a cell list for O(N) neighbour search. OpenMP over frames.
  No Eigen dependency — eigenvalues computed analytically (Cardano).
OPTIONS: --prefix --Nm --Nc --start --stop --out
  --Ns       <int>    Solvent particles      [default: 0]
  --step     <int>    Frame stride           [default: 1]
  --threads  <int>    OpenMP threads         [default: 1]
  --alpha_pp <float>  pp attractive amplitude [default: 0.5145]
  --alpha_ps <float>  ps attractive amplitude [default: 0.5145]
OUTPUT:  CSV: particle_id, fx_avg, fy_avg, fz_avg, fmag_avg,
              lambda1, lambda2, lambda3, anisotropy, prolateness
EXAMPLE: ./analysis --tool fellipsoid --prefix requil_ --Nm 500 --Nc 1000 \
           --Ns 2000000 --start 0 --stop 10 --step 1 --threads 8
)"}
};

// ═══════════════════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exitCode = 0;

    try {
        md::Args a = md::parseArgs(argc, argv);

        if (a.help) {
            if (rank == 0) {
                if (!a.tool.empty()) {
                    auto it = TOOL_HELP.find(a.tool);
                    std::cout << (it != TOOL_HELP.end()
                        ? it->second : "Unknown tool. Run --help for list.\n") << '\n';
                } else { std::cout << GLOBAL_HELP; }
            }
            MPI_Finalize(); return 0;
        }
        if (a.tool.empty()) {
            if (rank == 0) std::cerr << "Error: --tool required. Run --help.\n";
            MPI_Finalize(); return 1;
        }

        auto tool = md::Registry::instance().create(a.tool);
        if (!tool)
            throw std::invalid_argument("Unknown tool '" + a.tool + "'. Run --help.");

        // MPI-parallel tools execute on every rank; the rest run on rank 0 only.
        if (tool->runsOnAllRanks() || rank == 0)
            tool->run(a);

    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "\nError: " << e.what() << '\n';
        exitCode = 1;
    }

    MPI_Finalize();
    return exitCode;
}
