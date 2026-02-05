import pytest

import vampyr as vp


def test_ScalingProjector_1d_scalar_func():
    """In 1D, scalar functions (lambda x: ...) should now work with projectors."""
    def f(x):
        return x

    mra = vp.MultiResolutionAnalysis(box=[0, 1], order=7)
    P_scaling = vp.ScalingProjector(mra, 2)
    P_wavelet = vp.WaveletProjector(mra, 2)

    tree_s = P_scaling(f)
    assert tree_s.nNodes() > 0
    assert tree_s([0.5]) == pytest.approx(0.5, abs=1e-3)

    tree_w = P_wavelet(f)
    assert tree_w.nNodes() > 0


def test_ScalingProjector_1d_vector_func():
    """In 1D, vector-style functions (lambda r: r[0]...) still work."""
    def f(r):
        return r[0]

    mra = vp.MultiResolutionAnalysis(box=[0, 1], order=7)
    P_scaling = vp.ScalingProjector(mra, 2)
    P_wavelet = vp.WaveletProjector(mra, 2)

    tree_s = P_scaling(f)
    assert tree_s.nNodes() > 0
    assert tree_s([0.5]) == pytest.approx(0.5, abs=1e-3)

    tree_w = P_wavelet(f)
    assert tree_w.nNodes() > 0
