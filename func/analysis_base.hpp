/**
 * @file analysis_base.hpp
 * @brief Polymorphic tool framework for the analysis suite.
 *
 * Design patterns used here:
 *   - Command           : every tool is an Analysis subclass that encapsulates
 *                          one action behind a uniform run() interface, so the
 *                          driver is decoupled from the concrete tools.
 *   - Factory Method     : Registry stores a creator per tool name and builds
 *                          tools on demand, returning them through the base
 *                          interface.
 *   - Self-registration  : the Register<T> helper inserts a tool into the
 *                          Registry at static-initialisation time, so adding a
 *                          tool needs no edits to existing code (open/closed).
 *   - Singleton (Meyers) : a single process-wide Registry instance.
 */
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace md {

struct Args;  // defined in args.hpp; tools receive it by const reference

/**
 * @brief Abstract interface implemented by every analysis tool.
 */
class Analysis {
public:
    virtual ~Analysis() = default;

    /// Execute the analysis using the parsed command-line arguments.
    virtual void run(const Args& args) const = 0;

    /// Tools that must execute on every MPI rank return true (e.g. msd,
    /// pressure). Single-process tools use the default and run on rank 0 only.
    virtual bool runsOnAllRanks() const { return false; }
};

/**
 * @brief Name -> tool factory with self-registration.
 */
class Registry {
public:
    using Creator = std::function<std::unique_ptr<Analysis>()>;

    static Registry& instance() {            // Meyers singleton
        static Registry reg;
        return reg;
    }

    void add(const std::string& name, Creator make) {
        creators_[name] = std::move(make);
    }

    /// Construct the tool registered under @p name, or nullptr if unknown.
    std::unique_ptr<Analysis> create(const std::string& name) const {
        const auto it = creators_.find(name);
        return it == creators_.end() ? nullptr : it->second();
    }

    bool contains(const std::string& name) const {
        return creators_.find(name) != creators_.end();
    }

    /// All registered tool names (sorted, since the backing map is ordered).
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(creators_.size());
        for (const auto& kv : creators_) out.push_back(kv.first);
        return out;
    }

private:
    Registry() = default;
    std::map<std::string, Creator> creators_;
};

/**
 * @brief Static-init helper that registers tool type @c T under a name.
 *
 * Usage in a tool header:
 * @code
 *   inline const md::Register<RDFAnalysis> reg_rdf{"rdf"};
 * @endcode
 *
 * Declared as an @c inline variable so the registration object can live in a
 * header included by several translation units without breaking the ODR.
 */
template <class T>
struct Register {
    explicit Register(const std::string& name) {
        Registry::instance().add(name, [] { return std::make_unique<T>(); });
    }
};

} // namespace md
