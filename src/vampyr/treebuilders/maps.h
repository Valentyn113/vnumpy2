#pragma once
 
#include <pybind11/functional.h>
 
#include "PyFunctionMap.h"
#include <MRCPP/treebuilders/map.h>
 
namespace vampyr {
 
// Friendly class-call form: vp.FunctionMap3D(fmap, prec)(tree)
template <int D> void map(pybind11::module &m) {
    using namespace mrcpp;
    namespace py = pybind11;
    using namespace pybind11::literals;
 
    // Real version
    py::class_<PyFunctionMap<D, double>>(
        m, (std::string("FunctionMap") + std::to_string(D) + "D").c_str())
        .def(py::init<std::function<double(double)>, double>(),
             "fmap"_a, "prec"_a)
        .def("__call__",
             [](PyFunctionMap<D, double> &F, FunctionTree<D, double> &inp) {
                 auto old_threads = mrcpp_get_num_threads();
                 set_max_threads(1);
                 auto out = F(inp);
                 set_max_threads(old_threads);
                 return out;
             },
             "inp"_a);
 
    // Complex version
    py::class_<PyFunctionMap<D, ComplexDouble>>(
        m, (std::string("FunctionMap") + std::to_string(D) + "D_Complex").c_str())
        .def(py::init<std::function<ComplexDouble(ComplexDouble)>, double>(),
             "fmap"_a, "prec"_a)
        .def("__call__",
             [](PyFunctionMap<D, ComplexDouble> &F,
                FunctionTree<D, ComplexDouble> &inp) {
                 auto old_threads = mrcpp_get_num_threads();
                 set_max_threads(1);
                 auto out = F(inp);
                 set_max_threads(old_threads);
                 return out;
             },
             "inp"_a);
}
 
// Advanced free-function form: vp.advanced.map(prec, out, inp, fmap)
template <int D> void advanced_map(pybind11::module &m) {
    using namespace mrcpp;
    namespace py = pybind11;
    using namespace pybind11::literals;
 
    // Real version (existing)
    m.def("map",
          [](double prec,
             FunctionTree<D, double> &out,
             FunctionTree<D, double> &inp,
             std::function<double(double)> fmap,
             int max_iter,
             bool abs_prec) {
              auto old_threads = mrcpp_get_num_threads();
              mrcpp::set_max_threads(1);
              mrcpp::treeMap<D, double>(prec, out, inp, fmap, max_iter, abs_prec);
              mrcpp::set_max_threads(old_threads);
          },
          "prec"_a = -1.0,
          "out"_a,
          "inp"_a,
          "fmap"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false);
 
    // Complex version (new)
    m.def("map",
          [](double prec,
             FunctionTree<D, ComplexDouble> &out,
             FunctionTree<D, ComplexDouble> &inp,
             std::function<ComplexDouble(ComplexDouble)> fmap,
             int max_iter,
             bool abs_prec) {
              auto old_threads = mrcpp_get_num_threads();
              mrcpp::set_max_threads(1);
              mrcpp::treeMap<D, ComplexDouble>(prec, out, inp, fmap, max_iter, abs_prec);
              mrcpp::set_max_threads(old_threads);
          },
          "prec"_a = -1.0,
          "out"_a,
          "inp"_a,
          "fmap"_a,
          "max_iter"_a = -1,
          "abs_prec"_a = false);
}
} // namespace vampyr
