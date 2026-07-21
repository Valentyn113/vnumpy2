#pragma once

#include <pybind11/pybind11.h>

#include <MRCPP/operators/CartesianConvolution.h>
#include <MRCPP/operators/HelmholtzOperator.h>
#include <MRCPP/operators/IdentityConvolution.h>
#include <MRCPP/operators/PoissonOperator.h>
#include <MRCPP/operators/TimeEvolutionOperator.h>
#include <MRCPP/operators/HeatOperator.h>
#include <MRCPP/treebuilders/apply.h>

namespace vampyr {

void cartesian_convolution(pybind11::module &);
void helmholtz_operator(pybind11::module &);
void poisson_operator(pybind11::module &);
void time_evolution_operator(pybind11::module &m);
void heat_operator(pybind11::module &m);

// ============================================================================
//  Operator __call__ helpers (local to this translation unit)
//
//  VAMPyR exposes the Poisson and Helmholtz operators through two
//  intentionally distinct conventions:
//
//   - The class-call form,  op(rho),  applies the Green's-function
//     convention and divides the result by 4π.  This is the form most
//     users want for solving  -∇²u = ρ  (Poisson) or  (-∇² + μ²)u = ρ
//     (Helmholtz):  the returned u is in the form quantum-chemistry and
//     classical PDE texts call "the Poisson potential" or "the
//     Helmholtz potential" respectively.
//
//   - The free-function form,  vp.advanced.apply(prec, out, op, rho),
//     applies the bare kernel and does NOT divide by 4π.  This is the
//     direct binding of mrcpp::apply and preserves MRCPP's bare-kernel
//     convention, which is what users typically want when composing
//     operator applications by hand or matching a specific physics
//     convention.
//
//  Both forms are correct and deliberate.  The factor of 4π between
//  them is intentional and not a bug.  These two helpers are defined
//  here to make the choice visible at the call site:
//
//      apply_bare                returns mrcpp::apply(...) unchanged
//      apply_with_greens_rescale returns mrcpp::apply(...) / (4π)
//
// ============================================================================
namespace detail {
template <int D, typename Scalar, typename Op>
std::unique_ptr<mrcpp::FunctionTree<D, Scalar>>
apply_bare(Op &op, mrcpp::FunctionTree<D, Scalar> *inp) {
    auto out = std::make_unique<mrcpp::FunctionTree<D, Scalar>>(inp->getMRA());
    mrcpp::apply<D, Scalar>(op.getBuildPrec(), *out, op, *inp);
    return out;
}

template <int D, typename Scalar, typename Op>
std::unique_ptr<mrcpp::FunctionTree<D, Scalar>>
apply_with_greens_rescale(Op &op, mrcpp::FunctionTree<D, Scalar> *inp) {
    auto out = std::make_unique<mrcpp::FunctionTree<D, Scalar>>(inp->getMRA());
    mrcpp::apply<D, Scalar>(op.getBuildPrec(), *out, op, *inp);
    out->rescale(1.0 / (4.0 * mrcpp::pi));
    return out;
}
} // namespace detail


// ----------------------------------------------------------------------------
//  Generic ConvolutionOperator and IdentityConvolution (templated on D)
// ----------------------------------------------------------------------------
template <int D> void convolutions(pybind11::module &m) {
    namespace py = pybind11;
    using namespace mrcpp;
    using namespace pybind11::literals;

    py::class_<ConvolutionOperator<D>>(m, (std::string("ConvolutionOperator") + std::to_string(D) + "D").c_str())
        .def(py::init<const MultiResolutionAnalysis<D> &, GaussExp<1> &, double>(), "mra"_a, "kernel"_a, "prec"_a)
        .def(py::init<const MultiResolutionAnalysis<D> &, GaussExp<1> &, double, int, int>())
        .def("__call__",
             [](ConvolutionOperator<D> &C, FunctionTree<D, double> *inp) {
                 return detail::apply_bare<D, double>(C, inp);
             }, "inp"_a)
        .def("__call__",
             [](ConvolutionOperator<D> &C, FunctionTree<D, ComplexDouble> *inp) {
                 return detail::apply_bare<D, ComplexDouble>(C, inp);
             }, "inp"_a);

    py::class_<IdentityConvolution<D>, ConvolutionOperator<D>>(m, (std::string("IdentityConvolution") + std::to_string(D) + "D").c_str())
        .def(py::init<const MultiResolutionAnalysis<D> &, double>(), "mra"_a, "prec"_a)
        .def(py::init<const MultiResolutionAnalysis<D> &, double, int, int>(),
             "mra"_a, "prec"_a, "root"_a = 0, "reach"_a = 1)
        .def("__call__",
             [](IdentityConvolution<D> &I, FunctionTree<D, double> *inp) {
                 return detail::apply_bare<D, double>(I, inp);
             }, "inp"_a)
        .def("__call__",
             [](IdentityConvolution<D> &I, FunctionTree<D, ComplexDouble> *inp) {
                 return detail::apply_bare<D, ComplexDouble>(I, inp);
             }, "inp"_a);

    if constexpr (D == 3) cartesian_convolution(m);
    if constexpr (D == 3) helmholtz_operator(m);
    if constexpr (D == 3) poisson_operator(m);
    if constexpr (D == 1) time_evolution_operator(m);
    if constexpr (D == 1) heat_operator(m);
}
// ----------------------------------------------------------------------------
//  CartesianConvolution (3D only)
//  Bare-kernel convention -- no rescale.
// ----------------------------------------------------------------------------
void cartesian_convolution(pybind11::module &m) {
    namespace py = pybind11;
    using namespace mrcpp;
    using namespace pybind11::literals;

    py::class_<CartesianConvolution, ConvolutionOperator<3>>(m, "CartesianConvolution")
        .def(py::init<const MultiResolutionAnalysis<3> &, GaussExp<1> &, double>(),
             "mra"_a, "kernel"_a, "prec"_a)
        .def("__call__",
             [](CartesianConvolution &O, FunctionTree<3, double> *inp) {
                 return detail::apply_bare<3, double>(O, inp);
             }, "inp"_a,
             "Apply the CartesianConvolution operator to a real source.")
        .def("__call__",
             [](CartesianConvolution &O, FunctionTree<3, ComplexDouble> *inp) {
                 return detail::apply_bare<3, ComplexDouble>(O, inp);
             }, "inp"_a,
             "Apply the CartesianConvolution operator to a complex source.")
        .def("setCartesianComponents", &CartesianConvolution::setCartesianComponents);
}

// ----------------------------------------------------------------------------
//  PoissonOperator (3D only)
//  Uses the Green's-function convention: result is divided by 4π.
//  See the convention note at the top of this file.
// ----------------------------------------------------------------------------
void poisson_operator(pybind11::module &m) {
    namespace py = pybind11;
    using namespace mrcpp;
    using namespace pybind11::literals;

    py::class_<PoissonOperator, ConvolutionOperator<3>>(m, "PoissonOperator")
        .def(py::init<const MultiResolutionAnalysis<3> &, double>(), "mra"_a, "prec"_a)
        .def(py::init<const MultiResolutionAnalysis<3> &, double, int, int>(),
             "mra"_a, "prec"_a, "root"_a = 0, "reach"_a = 1)
        .def("__call__",
             [](PoissonOperator &P, FunctionTree<3, double> *inp) {
                 return detail::apply_with_greens_rescale<3, double>(P, inp);
             }, "inp"_a,
             "Apply Poisson with the Green's-function convention "
             "(includes 1/(4π) factor).")
        .def("__call__",
             [](PoissonOperator &P, FunctionTree<3, ComplexDouble> *inp) {
                 return detail::apply_with_greens_rescale<3, ComplexDouble>(P, inp);
             }, "inp"_a,
             "Apply Poisson to a complex source.  Same Green's-function "
             "convention as the real overload.");
}


// ----------------------------------------------------------------------------
//  HelmholtzOperator (3D only)
//  Same Green's-function convention as Poisson.
// ----------------------------------------------------------------------------
void helmholtz_operator(pybind11::module &m) {
    namespace py = pybind11;
    using namespace mrcpp;
    using namespace pybind11::literals;

    py::class_<HelmholtzOperator, ConvolutionOperator<3>>(m, "HelmholtzOperator")
        .def(py::init<const MultiResolutionAnalysis<3> &, double, double>(), "mra"_a, "exp"_a, "prec"_a)
        .def(py::init<const MultiResolutionAnalysis<3> &, double, double, int, int>(),
             "mra"_a, "exp"_a, "prec"_a, "root"_a = 0, "reach"_a = 1)
        .def("__call__",
             [](HelmholtzOperator &H, FunctionTree<3, double> *inp) {
                 return detail::apply_with_greens_rescale<3, double>(H, inp);
             }, "inp"_a,
             "Apply Helmholtz with the Green's-function convention "
             "(includes 1/(4π) factor).")
        .def("__call__",
             [](HelmholtzOperator &H, FunctionTree<3, ComplexDouble> *inp) {
                 return detail::apply_with_greens_rescale<3, ComplexDouble>(H, inp);
             }, "inp"_a,
             "Apply Helmholtz to a complex source.  Same Green's-function "
             "convention as the real overload.");
}


// ----------------------------------------------------------------------------
//  TimeEvolutionOperator (1D only)
//  Bare-kernel convention -- no rescale.
// ----------------------------------------------------------------------------
void time_evolution_operator(pybind11::module &m) {
    namespace py = pybind11;
    using namespace mrcpp;
    using namespace pybind11::literals;

    py::class_<TimeEvolutionOperator<1>, ConvolutionOperator<1>>(m, "TimeEvolutionOperator")
        .def(py::init<const MultiResolutionAnalysis<1> &, double, double, int, bool, int>(),
             "mra"_a, "prec"_a, "time"_a, "finest_scale"_a, "imaginary"_a, "max_Jpower"_a = 20)
        .def(py::init<const MultiResolutionAnalysis<1> &, double, double, bool, int>(),
             "mra"_a, "prec"_a, "time"_a, "imaginary"_a, "max_Jpower"_a = 40)
        .def("__call__",
             [](TimeEvolutionOperator<1> &T, FunctionTree<1, double> *inp) {
                 return detail::apply_bare<1, double>(T, inp);
             }, "inp"_a,
             "Apply the time-evolution operator to a real source.");

}


// ----------------------------------------------------------------------------
//  HeatOperator (1D only)
//  Bare-kernel convention -- no rescale.
// ----------------------------------------------------------------------------
void heat_operator(pybind11::module &m) {
    namespace py = pybind11;
    using namespace mrcpp;
    using namespace pybind11::literals;

    py::class_<HeatOperator<1>, ConvolutionOperator<1>>(m, "HeatOperator")
        .def(py::init<const MultiResolutionAnalysis<1> &, double, double>(),
             "mra"_a, "time"_a, "prec"_a)
        .def("__call__",
             [](HeatOperator<1> &T, FunctionTree<1, double> *inp) {
                 return detail::apply_bare<1, double>(T, inp);
             }, "inp"_a,
             "Apply the heat operator to a real source.")
        .def("__call__",
             [](HeatOperator<1> &T, FunctionTree<1, ComplexDouble> *inp) {
                 return detail::apply_bare<1, ComplexDouble>(T, inp);
             }, "inp"_a,
             "Apply the heat operator to a complex source.");
}

} // namespace vampyr
