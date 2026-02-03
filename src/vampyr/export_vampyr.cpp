/*
 *  \date Mar 08, 2018
 *  \author Magnar Bjørgve <magnar.bjorgve@uit.no> \n
 *          Hylleraas Centre for Quantum Molecular Sciences \n
 *          UiT - The Arctic University of Norway
 */

#include <pybind11/pybind11.h>

#include <string>

#include <MRCPP/constants.h>
#include <MRCPP/version.h>

#include "core/bases.h"
#include "core/filter.h"
#include "functions/functions.h"
#include "operators/convolutions.h"
#include "operators/derivatives.h"
#include "treebuilders/applys.h"
#include "treebuilders/arithmetics.h"
#include "treebuilders/grids.h"
#include "treebuilders/maps.h"
#include "treebuilders/project.h"
#include "trees/trees.h"
#include "trees/world.h"

namespace py = pybind11;
using namespace mrcpp;
using namespace pybind11::literals;

namespace vampyr {

void constants(py::module &m) {
    py::enum_<Traverse>(m, "Traverse")
        .value("TopDown", Traverse::TopDown)
        .value("BottomUp", Traverse::BottomUp)
        .export_values();
    py::enum_<Iterator>(m, "Iterator")
        .value("Lebesgue", Iterator::Lebesgue)
        .value("Hilbert", Iterator::Hilbert)
        .export_values();
}

template <int D> void bind_advanced(py::module &mod) noexcept {
    advanced_applys<D>(mod);
    advanced_arithmetics<D>(mod);
    advanced_project<D>(mod);
    advanced_grids<D>(mod);
    advanced_map<D>(mod);
}

template <int D> void bind_vampyr(py::module &mod) noexcept {
    functions<D>(mod);
    trees<D>(mod);
    world<D>(mod);
    grids<D>(mod);
    applys<D>(mod);
    arithmetics<D>(mod);
    project<D>(mod);
    map<D>(mod);
    derivatives<D>(mod);
    convolutions<D>(mod);
}

PYBIND11_MODULE(_vampyr, m) {
    m.doc() = R"pbdoc(
        VAMPyR
        ------

        VAMPyR makes the MRCPP functionality available through a Python interface.

        .. currentmodule:: vampyr

        .. autosummary::
           :toctree: generate
    )pbdoc";

    m.attr("__version__") = VERSION_INFO;

    m.def("mrcpp_version", &program_version, "Return version of the underlying MRCPP library.");

    // Dimension-independent bindings go in the main module
    constants(m);

    // Dimension-dependent bindings go into the main module
    bind_vampyr<1>(m);
    bind_vampyr<2>(m);
    bind_vampyr<3>(m);

    // Advanced bindings go into a single "advanced" submodule
    py::module advanced_mod = m.def_submodule("advanced");
    bind_advanced<1>(advanced_mod);
    bind_advanced<2>(advanced_mod);
    bind_advanced<3>(advanced_mod);

    bases(m);
    filter(m);
}
} // namespace vampyr
