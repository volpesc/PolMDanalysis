/**
 * @file tools.hpp
 * @brief Concrete analysis tools.
 *
 * Each tool is a small Analysis subclass whose run() translates the parsed Args
 * into the tool's Config and calls the corresponding (unchanged) compute*
 * function. Every tool self-registers under its CLI name via an inline
 * Register<> object, so the driver never needs to know the concrete types and
 * adding a tool means adding a class here (or in its own header) - no edits to
 * existing code.
 */
#pragma once

#include "args.hpp"
#include "analysis_base.hpp"

#include "utility.hpp"
#include "msd_mpi.hpp"
#include "gyr_endz.hpp"
#include "msid.hpp"
#include "msd_front.hpp"
#include "structure_factor.hpp"
#include "pressure_z_mpi.hpp"
#include "rdf.hpp"
#include "density.hpp"
#include "density_front.hpp"
#include "endtoend.hpp"
#include "bond_angle.hpp"
#include "ppa.hpp"
#include "entanglement.hpp"
#include "spec_density.hpp"
#include "volume.hpp"
#include "backbone.hpp"
#include "energy.hpp"
#include "nematic.hpp"
#include "force_ellipsoid.hpp"
#include "gds_diffusion.hpp"
#include "brush_length.hpp"

namespace md {

// ── Structural / conformational ───────────────────────────────────────────────

class MSDAnalysis : public Analysis {       // MPI-parallel over time origins
public:
    bool runsOnAllRanks() const override { return true; }
    void run(const Args& a) const override {
        MSDConfig c; c.filenamePrefix=a.prefix; c.frameStart=a.start;
        c.frameStop=a.stop; c.frameStep=a.step; c.timeStep=a.dt; c.Nm=a.Nm; c.Nc=a.Nc;
        computeMSD(c, a.out.empty()?"msd.dat":a.out);
    }
};
inline const Register<MSDAnalysis> reg_msd{"msd"};

class GyrAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        computeGyrationProfile(a.prefix,a.frame,
            a.out.empty()?"rg_profile.dat":a.out,a.Nm,a.Nc,a.binWidth,a.nBins);
    }
};

class MSDFrontAnalysis : public Analysis {  // MPI-parallel over time origins
public:
    bool runsOnAllRanks() const override { return true; }
    void run(const Args& a) const override {
        MSDFrontConfig c; c.filenamePrefix=a.prefix; c.frameStart=a.start;
        c.frameStop=a.stop; c.frameStep=a.step; c.timeStep=a.dt;
        c.Nm=a.Nm; c.Nc=a.Nc; c.Ns=a.Ns;
        c.xMin=a.xMin; c.xMax=a.xMax; c.binWidth=a.binWidth;
        c.frontBuffer=a.frontBuffer; c.tMax=a.tMax; c.nLogPoints=a.nLogPoints;
        computeMSDFront(c, a.out.empty()?"msd_front.dat":a.out);
    }
};
inline const Register<MSDFrontAnalysis> reg_msdfront{"msdfront"};

inline const Register<GyrAnalysis> reg_gyr{"gyr"};

class MSIDAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        MSIDConfig c; c.filenamePrefix=a.prefix; c.frameStart=a.start;
        c.frameStop=a.stop; c.frameStep=a.step; c.Nm=a.Nm; c.Nc=a.Nc; c.bondLength=a.lb;
        computeMSID(c, a.out.empty()?"msid.dat":a.out);
    }
};
inline const Register<MSIDAnalysis> reg_msid{"msid"};

class EndToEndAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        EndToEndConfig c; c.filenamePrefix=a.prefix; c.frameStart=a.start;
        c.frameStop=a.stop; c.frameStep=a.step; c.Nm=a.Nm; c.Nc=a.Nc; c.nBins=a.nBins;
        computeEndToEnd(c, a.out.empty()?"endtoend":a.out);
    }
};
inline const Register<EndToEndAnalysis> reg_endtoend{"endtoend"};

class BondAngleAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        BondAngleConfig c; c.filenamePrefix=a.prefix; c.frameStart=a.start;
        c.frameStop=a.stop; c.frameStep=a.step; c.Nm=a.Nm; c.Nc=a.Nc; c.nBins=a.nBins;
        computeBondAngle(c, a.out.empty()?"bond_angle.dat":a.out);
    }
};
inline const Register<BondAngleAnalysis> reg_bondangle{"bondangle"};

class VolumeAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        VolumeConfig c; c.filenamePrefix=a.prefix; c.frameIndex=a.frame;
        c.Nm=a.Nm; c.Nc=a.Nc;
        computeVolume(c, a.out.empty()?"volume.dat":a.out);
    }
};
inline const Register<VolumeAnalysis> reg_volume{"volume"};

// ── Scattering / density ──────────────────────────────────────────────────────

class StructureFactorAnalysis : public Analysis {
public:
    // Strategy pattern: the k-space sampling scheme (spherical vs cartesian) is
    // selected at runtime and threaded through a single generic dispatch lambda.
    void run(const Args& a) const override {
        const std::string out = a.out.empty()?"Sq.dat":a.out;
        auto dispatch = [&](auto& s){
            if      (a.mode=="sc")  computeSc  (a.prefix,a.frame,s,a.Nm,a.Nc,out);
            else if (a.mode=="tot") computeStot(a.prefix,a.start,a.stop,s,a.Nm,a.Nc,out);
            else throw std::invalid_argument("--mode must be sc or tot");
        };
        if (a.sampling=="spherical") {
            SphericalSampling s; s.nkBound=a.nkBound; dispatch(s);
        } else {
            CartesianSampling s; s.nkBound=a.nkBound; s.dkMultiplier=a.dkMult;
            s.binTolerance=a.binTol; dispatch(s);
        }
    }
};
inline const Register<StructureFactorAnalysis> reg_sq{"sq"};

class RDFAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        RDFConfig c; c.filenamePrefix=a.prefix; c.frameStart=a.start;
        c.frameStop=a.stop; c.frameStep=a.step; c.Nm=a.Nm; c.Nc=a.Nc;
        c.nBins=a.nBins; c.rCut=a.rCut;
        computeRDF(c, a.out.empty()?"rdf.dat":a.out);
    }
};
inline const Register<RDFAnalysis> reg_rdf{"rdf"};

class DensityAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        DensityConfig c; c.filenamePrefix=a.prefix; c.frameStart=a.start;
        c.frameStop=a.stop; c.frameStep=a.step; c.Nm=a.Nm; c.Nc=a.Nc;
        c.N_s=a.Ns; c.nBins=a.nBins;
        computeDensity(c, a.out.empty()?"density.dat":a.out);
    }
};
inline const Register<DensityAnalysis> reg_density{"density"};

class SpecDensityAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        SpecDensityConfig c; c.filenamePrefix=a.prefix; c.frameIndex=a.frame;
        c.Nm=a.Nm; c.Nc=a.Nc; c.N_s=a.Ns; c.nShells=a.nShells; c.shellWidth=a.shellW;
        computeSpeciesDensity(c, a.out.empty()?"spec_density.dat":a.out);
    }
};
inline const Register<SpecDensityAnalysis> reg_specdensity{"specdensity"};


class DensityFrontAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        DensityFrontConfig c;
        c.filenamePrefix = a.prefix; c.Nm = a.Nm; c.Nc = a.Nc; c.Ns = a.Ns;
        c.nBins = a.nBins; c.deltaT = a.dt; c.frame0 = a.frame0;
        c.atFrames = a.atFrames; c.frontWindow = a.frontWindow;
        computeDensityFront(c, a.out.empty() ? "densityfront" : a.out);
    }
};
inline const Register<DensityFrontAnalysis> reg_densityfront{"densityfront"};


// ── Mechanics / energy ────────────────────────────────────────────────────────

class BackboneAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        BackboneConfig c; c.filenamePrefix=a.prefix; c.frameIndex=a.frame;
        c.Nm=a.Nm; c.Nc=a.Nc; c.mode=a.bbMode;
        c.stretchDir={a.stretchX,a.stretchY,a.stretchZ};
        computeBackbone(c, a.out.empty()?"backbone.dat":a.out);
    }
};
inline const Register<BackboneAnalysis> reg_backbone{"backbone"};

class EnergyAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        EnergyConfig c; c.filenamePrefix=a.prefix; c.frameIndex=a.frame;
        c.Nm=a.Nm; c.Nc=a.Nc; c.N_s=a.Ns; c.lambda=a.lambda;
        computeEnergy(c, a.out);
    }
};
inline const Register<EnergyAnalysis> reg_energy{"energy"};

class PressureAnalysis : public Analysis {  // MPI-parallel over layers
public:
    bool runsOnAllRanks() const override { return true; }
    void run(const Args& a) const override {
        PressureConfig c; c.filenamePrefix=a.prefix; c.frameIndex=a.frame;
        c.Nm=a.Nm; c.Nc=a.Nc; c.N_s=a.Ns; c.nLayer=a.nLayer;
        computeLayerPressure(c, a.out.empty()?"pressure":a.out);
    }
};
inline const Register<PressureAnalysis> reg_pressure{"pressure"};

// ── Entanglement ──────────────────────────────────────────────────────────────

class PPAAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        PPAConfig c; c.filenamePrefix=a.prefix; c.frameIndex=a.frame;
        c.Nm=a.Nm; c.Nc=a.Nc;
        computePPA(c, a.out.empty()?"ppa.dat":a.out);
    }
};
inline const Register<PPAAnalysis> reg_ppa{"ppa"};

class EntanglementAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        EntanglementConfig c; c.filenamePrefix=a.prefix; c.frameIndex=a.frame;
        c.Nm=a.Nm; c.Nc=a.Nc; c.N_s=a.Ns;
        c.nBins=a.nBins; c.kinkThresholdDeg=a.kinkDeg;
        // Dedicated --entmode (default "chain_com"); fall back to a valid --bbmode
        // for backward compatibility, but never inherit backbone's "full".
        if (a.entModeSet)
            c.mode = a.entMode;
        else if (a.bbMode == "chain_com" || a.bbMode == "monomer")
            c.mode = a.bbMode;
        else
            c.mode = a.entMode;                 // "chain_com"
        computeEntanglement(c, a.out.empty()?"entanglement.dat":a.out);
    }
};
inline const Register<EntanglementAnalysis> reg_entanglement{"entanglement"};

// ── Order / mechanics / diffusion ─────────────────────────────────────────────

class NematicAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        NematicConfig c; c.filenamePrefix=a.prefix; c.frameIndex=a.frame;
        c.Nm=a.Nm; c.Nc=a.Nc; c.N_s=a.Ns; c.nBins=a.nBins;
        computeNematic(c, a.out.empty()?"nematic.dat":a.out);
    }
};
inline const Register<NematicAnalysis> reg_nematic{"nematic"};

class ForceEllipsoidAnalysis : public Analysis {  // OpenMP-parallel per frame
public:
    void run(const Args& a) const override {
        ForceEllipsoidConfig c;
        c.filenamePrefix=a.prefix; c.frameStart=a.start; c.frameStop=a.stop;
        c.frameStride=a.step; c.Nm=a.Nm; c.Nc=a.Nc; c.N_s=a.Ns;
        c.alpha_pp=a.alphaPP; c.alpha_ps=a.alphaPS; c.threads=a.threads;
        c.outputStem=a.out.empty()?"interval":a.out;
        computeForceEllipsoid(c);
    }
};
inline const Register<ForceEllipsoidAnalysis> reg_fellipsoid{"fellipsoid"};

class GDSAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        GDSConfig c; c.filenamePrefix=a.prefix;
        c.frameStart=a.start; c.frameStop=a.stop;
        c.Nm=a.Nm; c.Nc=a.Nc; c.N_s=a.Ns;
        c.xMin=a.xMin; c.xMax=a.xMax; c.binWidth=a.binWidth;
        c.deltaT=a.dt; c.solventRegionStart=a.xThresh;
        c.velocityThreshold=a.vThresh;
        computeGDSDiffusion(c, a.out.empty()?"gds_diffusion.csv":a.out);
    }
};
inline const Register<GDSAnalysis> reg_gds{"gds"};

class BrushLengthAnalysis : public Analysis {
public:
    void run(const Args& a) const override {
        BrushLengthConfig c; c.filenamePrefix=a.prefix; c.frameIndex=a.frame;
        c.Nm=a.Nm; c.Nc=a.Nc; c.N_s=a.Ns;
        c.frontPercentile=a.percentile; c.centreBuffer=a.cBuffer;
        c.binWidth=a.binWidth; c.minBin=a.minBin; c.maxBin=a.maxBin;
        computeBrushLength(c, a.out.empty()?"brush_length.dat":a.out);
    }
};
inline const Register<BrushLengthAnalysis> reg_brushlength{"brushlength"};

} // namespace md
