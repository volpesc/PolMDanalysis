/**
 * @file args.hpp
 * @brief Parsed command-line arguments shared by the driver and every tool.
 *
 * Args is a plain options bag; parseArgs() turns argv into one. Keeping this in
 * its own header lets each tool build its Config from Args without depending on
 * the driver translation unit.
 */
#pragma once

#include <string>
#include <stdexcept>
#include <vector>
#include <sstream>

namespace md {

struct Args {
    std::string tool;
    bool        help     {false};
    std::string prefix   {"requil_"};
    int    Nm    {10};    int    Nc    {10000};
    int    start {0};     int    stop  {400};
    int    step  {1};     int    frame {0};
    std::string out;
    // MSD
    float  dt        {50.0f};
    // Gyration / spec density
    float  binWidth  {3.0f};  int  nBins  {62};
    // MSID
    float  lb        {0.964f};
    // S(q)
    std::string mode{"sc"}, sampling{"spherical"};
    int   nkBound{100};  float dkMult{1.0f};  float binTol{0.1f};
    // Pressure / density / energy
    int    Ns        {0};
    int    nLayer    {120};
    float  rCut      {5.0f};
    double lambda    {1.0};
    // Density-front snapshots
    std::vector<int> atFrames;
    int    frame0      {0};
    int    frontWindow {5};
    // Backbone
    std::string bbMode{"full"};
    double stretchX{1.0}, stretchY{0.0}, stretchZ{0.0};
    // Entanglement
    int    nShells   {18};
    float  shellW    {1.0f};
    double kinkDeg   {147.0};   ///< Kink angle threshold [deg] (PPA/entanglement)
    std::string entMode{"chain_com"};  ///< Entanglement binning: chain_com | monomer
    bool   entModeSet{false};          ///< True once --entmode is given explicitly
    // Force ellipsoid
    double alphaPP   {0.5145};
    double alphaPS   {0.5145};
    int    threads   {1};
    double xMin      {0.0};
    double xMax      {100.0};
    double xThresh   {45.0};
    double vThresh   {0.01};
    // Brush length
    double percentile{95.0};
    double cBuffer   {10.0};
    int    minBin    {2};
    int    maxBin    {90};
    // MSD front-resolved
    double frontBuffer{2.0};
    double tMax       {300.0};
    int    nLogPoints {40};

};

inline std::string nextArg(int& i, int argc, char** argv, const std::string& f) {
    if (++i >= argc) throw std::invalid_argument("Option " + f + " requires a value.");
    return argv[i];
}

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        if      (f=="--help"||f=="-h") a.help     = true;
        else if (f=="--tool")          a.tool      = nextArg(i,argc,argv,f);
        else if (f=="--prefix")        a.prefix    = nextArg(i,argc,argv,f);
        else if (f=="--Nm")     a.Nm       = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--Nc")     a.Nc       = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--start")  a.start    = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--stop")   a.stop     = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--step")   a.step     = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--frame")  a.frame    = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--out")    a.out      = nextArg(i,argc,argv,f);
        else if (f=="--dt")     a.dt       = std::stof(nextArg(i,argc,argv,f));
        else if (f=="--binw")   a.binWidth = std::stof(nextArg(i,argc,argv,f));
        else if (f=="--nbins")  a.nBins    = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--lb")     a.lb       = std::stof(nextArg(i,argc,argv,f));
        else if (f=="--mode")   a.mode     = nextArg(i,argc,argv,f);
        else if (f=="--sampling") a.sampling = nextArg(i,argc,argv,f);
        else if (f=="--nkbound") a.nkBound = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--dkmult") a.dkMult   = std::stof(nextArg(i,argc,argv,f));
        else if (f=="--bintol") a.binTol   = std::stof(nextArg(i,argc,argv,f));
        else if (f=="--Ns")     a.Ns       = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--atframes") {
            std::string list = nextArg(i,argc,argv,f);
            std::stringstream ss(list);
            std::string tok;
            while (std::getline(ss, tok, ',')) a.atFrames.push_back(std::stoi(tok));
        }
        else if (f=="--frame0")      a.frame0      = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--frontwindow") a.frontWindow = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--nlayer") a.nLayer   = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--rcut")   a.rCut     = std::stof(nextArg(i,argc,argv,f));
        else if (f=="--lambda") a.lambda   = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--bbmode") a.bbMode   = nextArg(i,argc,argv,f);
        else if (f=="--entmode"){ a.entMode = nextArg(i,argc,argv,f); a.entModeSet = true; }
        else if (f=="--stretchx") a.stretchX = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--stretchy") a.stretchY = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--stretchz") a.stretchZ = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--nshells") a.nShells  = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--shellw")  a.shellW   = std::stof(nextArg(i,argc,argv,f));
        else if (f=="--kinkdeg") a.kinkDeg  = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--alpha_pp") a.alphaPP  = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--alpha_ps") a.alphaPS  = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--threads")  a.threads  = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--xmin")    a.xMin     = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--xmax")    a.xMax     = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--frontbuf") a.frontBuffer = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--tmax")     a.tMax        = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--nlog")     a.nLogPoints  = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--xthresh") a.xThresh  = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--vthresh") a.vThresh  = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--percentile") a.percentile = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--cbuffer") a.cBuffer  = std::stod(nextArg(i,argc,argv,f));
        else if (f=="--minbin")  a.minBin   = std::stoi(nextArg(i,argc,argv,f));
        else if (f=="--maxbin")  a.maxBin   = std::stoi(nextArg(i,argc,argv,f));
        else throw std::invalid_argument("Unknown option: " + f + "\nRun --help.");
    }
    return a;
}

} // namespace md
