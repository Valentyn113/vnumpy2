"""Tests for the Gentle 1D Interface.

Verifies that scalar inputs (floats/ints) work for 1D VAMPyR operations,
while maintaining full backward compatibility with the [x] vector notation.
"""
import numpy as np
import pytest

import vampyr as vp

# Common setup
k = 5
N = -2
epsilon = 1.0e-3


class TestScalarEvaluation:
    """Test tree(0.5) works for 1D real and complex trees."""

    def setup_method(self):
        self.mra = vp.MultiResolutionAnalysis(box=[0, 1], order=k)

    def test_real_tree_scalar_eval(self):
        """tree(0.5) should return the same as tree([0.5])."""
        P = vp.ScalingProjector(self.mra, epsilon)
        gauss = vp.GaussFunc(alpha=1.0, beta=10.0, position=[0.5])
        tree = P(gauss)

        val_scalar = tree(0.5)
        val_vector = tree([0.5])
        assert val_scalar == pytest.approx(val_vector)

    def test_real_tree_scalar_eval_multiple_points(self):
        """Scalar eval should work at various points."""
        P = vp.ScalingProjector(self.mra, epsilon)
        gauss = vp.GaussFunc(alpha=1.0, beta=10.0, position=[0.5])
        tree = P(gauss)

        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert tree(x) == pytest.approx(tree([x]))

    def test_complex_tree_scalar_eval(self):
        """Complex tree(0.5) should return a complex value."""
        P = vp.ScalingProjector(self.mra, epsilon, dtype=complex)
        tree = P(lambda r: complex(r[0], r[0] ** 2))

        val_scalar = tree(0.5)
        val_vector = tree([0.5])
        assert val_scalar == pytest.approx(val_vector)
        assert isinstance(val_scalar, complex)

    def test_vector_eval_still_works(self):
        """Backward compat: tree([0.5]) must still work."""
        P = vp.ScalingProjector(self.mra, epsilon)
        gauss = vp.GaussFunc(alpha=1.0, beta=10.0, position=[0.5])
        tree = P(gauss)

        val = tree([0.5])
        assert isinstance(val, float)
        assert val > 0.0


class TestScalarFactories:
    """Test that scalar args are auto-wrapped to lists for 1D factories."""

    def test_bounding_box_scalar_args(self):
        """BoundingBox(corner=0, nboxes=1, scaling=1.0) should work."""
        box = vp.BoundingBox(corner=0, nboxes=1, scaling=1.0)
        assert box.dimension == 1

    def test_bounding_box_mixed_scalar(self):
        """BoundingBox with scalar nboxes and vector corner."""
        box = vp.BoundingBox(corner=[0], nboxes=1, scaling=[1.0])
        assert box.dimension == 1

    def test_bounding_box_vector_still_works(self):
        """Backward compat: BoundingBox(corner=[0], ...) still works."""
        box = vp.BoundingBox(corner=[0], nboxes=[1], scaling=[1.0])
        assert box.dimension == 1

    def test_gaussian_scalar_position(self):
        """Gaussian with scalar positional arg should work."""
        # Gaussian uses positional args: (alpha, beta, position, poly_exponent)
        g = vp.Gaussian(1.0, 10.0, 0.5, 0)
        assert g is not None

    def test_gaussian_vector_position_still_works(self):
        """Backward compat: Gaussian with [0.5] position still works."""
        g = vp.Gaussian(1.0, 10.0, [0.5], [0])
        assert g is not None

    def test_gaussfunc_scalar_position(self):
        """GaussFunc with position=0.5 (scalar) should work."""
        g = vp.GaussFunc(alpha=1.0, beta=10.0, position=0.5)
        assert g is not None

    def test_gaussfunc_vector_position_still_works(self):
        """Backward compat: GaussFunc with position=[0.5] still works."""
        g = vp.GaussFunc(alpha=1.0, beta=10.0, position=[0.5])
        assert g is not None

    def test_node_index_scalar_translation(self):
        """NodeIndex with translation=2 (scalar int) should work."""
        idx = vp.NodeIndex(scale=0, translation=2)
        assert idx.translation() == [2]

    def test_node_index_vector_translation_still_works(self):
        """Backward compat: NodeIndex with translation=[2] still works."""
        idx = vp.NodeIndex(scale=0, translation=[2])
        assert idx.translation() == [2]

class TestProjectorScalarFunctions:
    """Test that projectors accept lambda x: ... in 1D."""

    def setup_method(self):
        self.mra = vp.MultiResolutionAnalysis(box=[0, 1], order=k)

    def test_scaling_projector_scalar_func(self):
        """ScalingProjector should accept lambda x: x**2."""
        P = vp.ScalingProjector(self.mra, epsilon)
        tree = P(lambda x: x ** 2)
        # integral of x^2 from 0 to 1 = 1/3
        assert tree.integrate() == pytest.approx(1.0 / 3.0, rel=epsilon)

    def test_wavelet_projector_scalar_func(self):
        """WaveletProjector should accept lambda x: x**2."""
        P = vp.WaveletProjector(self.mra, 2)
        tree = P(lambda x: x ** 2)
        assert tree.nNodes() > 0

    def test_scaling_projector_vector_func_still_works(self):
        """Backward compat: lambda r: r[0]**2 still works."""
        P = vp.ScalingProjector(self.mra, epsilon)
        tree = P(lambda r: r[0] ** 2)
        assert tree.integrate() == pytest.approx(1.0 / 3.0, rel=epsilon)

    def test_scaling_projector_gaussfunc_still_works(self):
        """Backward compat: Projecting GaussFunc objects still works."""
        beta = 100.0
        alpha = (beta / np.pi) ** 0.5  # Normalized 1D Gaussian
        P = vp.ScalingProjector(self.mra, epsilon)
        gauss = vp.GaussFunc(beta=beta, alpha=alpha, position=[0.5])
        tree = P(gauss)
        assert tree.integrate() == pytest.approx(1.0, rel=epsilon)

    def test_complex_scaling_projector_scalar_func(self):
        """Complex ScalingProjector should accept scalar functions."""
        P = vp.ScalingProjector(self.mra, epsilon, dtype=complex)
        tree = P(lambda x: complex(x, x ** 2))
        assert tree.is_complex
        val = tree(0.5)
        assert isinstance(val, complex)
        assert val.real == pytest.approx(0.5, abs=0.1)
        assert val.imag == pytest.approx(0.25, abs=0.1)

    def test_complex_wavelet_projector_scalar_func(self):
        """Complex WaveletProjector should accept scalar functions."""
        P = vp.WaveletProjector(self.mra, 2, dtype=complex)
        tree = P(lambda x: complex(x, 0.0))
        assert tree.is_complex
        assert tree.nNodes() > 0

    def test_complex_projector_vector_func_still_works(self):
        """Backward compat: complex projector with lambda r: ... still works."""
        P = vp.ScalingProjector(self.mra, epsilon, dtype=complex)
        tree = P(lambda r: complex(r[0], r[0] ** 2))
        assert tree.is_complex


# ---- Integration test: full 1D workflow with scalars ----

class TestGentleWorkflow:
    """End-to-end test of the gentle 1D interface."""

    def test_full_scalar_workflow(self):
        """Complete 1D workflow using only scalar inputs."""
        # 1. Create MRA with scalar-friendly BoundingBox
        box = vp.BoundingBox(corner=0, nboxes=1, scaling=1.0)
        mra = vp.MultiResolutionAnalysis(box=box, order=k)

        # 2. Project a scalar function
        P = vp.ScalingProjector(mra, epsilon)
        tree = P(lambda x: np.sin(np.pi * x))

        # 3. Evaluate at scalar point
        val = tree(0.5)
        assert val == pytest.approx(1.0, rel=epsilon)

        # 4. Create GaussFunc with scalar position
        gauss = vp.GaussFunc(beta=100.0, alpha=1.0, position=0.5)
        tree2 = P(gauss)
        assert tree2(0.5) > tree2(0.0)
