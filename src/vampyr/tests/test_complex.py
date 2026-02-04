"""Tests for complex number support in VAMPyR."""

import pytest
import vampyr as vp


# Test setup for different dimensions
k = 5
N = -1

# 1D setup
world_1d = vp.BoundingBox(dim=1, scale=N)
mra_1d = vp.MultiResolutionAnalysis(box=world_1d, order=k)

# 2D setup
world_2d = vp.BoundingBox(dim=2, scale=N)
mra_2d = vp.MultiResolutionAnalysis(box=world_2d, order=k)

# 3D setup
world_3d = vp.BoundingBox(dim=3, scale=N)
mra_3d = vp.MultiResolutionAnalysis(box=world_3d, order=k)

# For backwards compatibility with existing tests
D = 3
world = world_3d
mra = mra_3d


def test_create_complex_function_tree():
    """Test creating a complex FunctionTree via the factory with dtype parameter."""
    tree = vp.FunctionTree(mra, dtype=complex)
    assert tree is not None
    assert tree.dtype == "complex128"
    assert tree.is_complex is True


def test_create_complex_function_tree_string_dtype():
    """Test creating a complex FunctionTree with string dtype."""
    tree = vp.FunctionTree(mra, dtype='complex128')
    assert tree.dtype == "complex128"
    assert tree.is_complex is True

    tree2 = vp.FunctionTree(mra, dtype='complex')
    assert tree2.dtype == "complex128"
    assert tree2.is_complex is True


def test_real_function_tree_dtype():
    """Test that real FunctionTree has correct dtype properties."""
    tree = vp.FunctionTree(mra)
    assert tree.dtype == "float64"
    assert tree.is_complex is False


def test_complex_tree_basic_properties():
    """Test basic properties of a complex FunctionTree."""
    name = "complex_func"
    tree = vp.FunctionTree(mra, name, dtype=complex)

    assert tree.name() == name
    assert tree.MRA() == mra
    assert tree.nNodes() == 1
    assert tree.nEndNodes() == 1
    assert tree.nGenNodes() == 0
    assert tree.rootScale() == N
    assert tree.depth() == 1


def test_complex_tree_set_zero():
    """Test setting a complex FunctionTree to zero."""
    tree = vp.FunctionTree(mra, dtype=complex)
    tree.setZero()

    assert tree.squaredNorm() == 0.0
    assert tree.norm() == 0.0


def test_complex_tree_integrate_zero():
    """Test integration of a zero complex FunctionTree."""
    tree = vp.FunctionTree(mra, dtype=complex)
    tree.setZero()

    result = tree.integrate()
    assert result == 0.0 + 0.0j


def test_complex_tree_eval_zero():
    """Test evaluation of a zero complex FunctionTree."""
    tree = vp.FunctionTree(mra, dtype=complex)
    tree.setZero()

    r = [0.1, 0.1, 0.1]
    result = tree(r)
    assert result == 0.0 + 0.0j


def test_complex_tree_deep_copy():
    """Test deep copy of a complex FunctionTree."""
    tree = vp.FunctionTree(mra, dtype=complex)
    tree.setZero()

    tree_copy = tree.deepCopy()
    assert tree_copy is not tree
    assert tree_copy.dtype == "complex128"
    assert tree_copy.squaredNorm() == 0.0


def test_complex_tree_clear():
    """Test clearing a complex FunctionTree."""
    tree = vp.FunctionTree(mra, dtype=complex)
    tree.setZero()
    assert tree.squaredNorm() == 0.0

    tree.clear()
    assert tree.squaredNorm() < 0.0  # Undefined state


class TestComplex1D:
    """Tests for 1D complex FunctionTree."""

    def test_create_complex_tree_1d(self):
        """Test creating a 1D complex FunctionTree."""
        tree = vp.FunctionTree(mra_1d, dtype=complex)
        assert tree.dtype == "complex128"
        assert tree.is_complex is True

    def test_complex_tree_1d_basic_operations(self):
        """Test basic operations on 1D complex tree."""
        tree = vp.FunctionTree(mra_1d, dtype=complex)
        tree.setZero()

        assert tree.squaredNorm() == 0.0
        assert tree.integrate() == 0.0 + 0.0j

        r = [0.1]
        assert tree(r) == 0.0 + 0.0j

    def test_complex_tree_1d_nodes(self):
        """Test node access on 1D complex tree."""
        tree = vp.FunctionTree(mra_1d, dtype=complex)
        tree.setZero()

        assert tree.nRootNodes() >= 1
        root = tree.fetchRootNode(0)
        assert root.isRootNode()
        assert root.hasCoefs()


class TestComplex2D:
    """Tests for 2D complex FunctionTree."""

    def test_create_complex_tree_2d(self):
        """Test creating a 2D complex FunctionTree."""
        tree = vp.FunctionTree(mra_2d, dtype=complex)
        assert tree.dtype == "complex128"
        assert tree.is_complex is True

    def test_complex_tree_2d_basic_operations(self):
        """Test basic operations on 2D complex tree."""
        tree = vp.FunctionTree(mra_2d, dtype=complex)
        tree.setZero()

        assert tree.squaredNorm() == 0.0
        assert tree.integrate() == 0.0 + 0.0j

        r = [0.1, 0.1]
        assert tree(r) == 0.0 + 0.0j

    def test_complex_tree_2d_nodes(self):
        """Test node access on 2D complex tree."""
        tree = vp.FunctionTree(mra_2d, dtype=complex)
        tree.setZero()

        assert tree.nRootNodes() >= 1
        root = tree.fetchRootNode(0)
        assert root.isRootNode()
        assert root.hasCoefs()


class TestComplex3D:
    """Tests for 3D complex FunctionTree (extended)."""

    def test_complex_tree_3d_nodes(self):
        """Test node access on 3D complex tree."""
        tree = vp.FunctionTree(mra_3d, dtype=complex)
        tree.setZero()

        assert tree.nRootNodes() >= 1
        root = tree.fetchRootNode(0)
        assert root.isRootNode()
        assert root.hasCoefs()
        assert root.squaredNorm() == 0.0

    def test_complex_tree_3d_fetch_node(self):
        """Test fetching nodes by index on 3D complex tree."""
        tree = vp.FunctionTree(mra_3d, dtype=complex)
        tree.setZero()

        idx = vp.NodeIndex(scale=N, dim=3)
        node = tree.fetchNode(idx)
        assert node is not None


class TestComplexArithmetic:
    """Tests for complex arithmetic operations."""

    def test_complex_add(self):
        """Test addition of two complex trees."""
        tree_a = vp.FunctionTree(mra_3d, dtype=complex)
        tree_b = vp.FunctionTree(mra_3d, dtype=complex)
        tree_a.setZero()
        tree_b.setZero()

        result = tree_a + tree_b
        assert result.dtype == "complex128"
        assert result.squaredNorm() == 0.0

    def test_complex_sub(self):
        """Test subtraction of two complex trees."""
        tree_a = vp.FunctionTree(mra_3d, dtype=complex)
        tree_b = vp.FunctionTree(mra_3d, dtype=complex)
        tree_a.setZero()
        tree_b.setZero()

        result = tree_a - tree_b
        assert result.dtype == "complex128"
        assert result.squaredNorm() == 0.0

    def test_complex_mul_trees(self):
        """Test multiplication of two complex trees."""
        tree_a = vp.FunctionTree(mra_3d, dtype=complex)
        tree_b = vp.FunctionTree(mra_3d, dtype=complex)
        tree_a.setZero()
        tree_b.setZero()

        result = tree_a * tree_b
        assert result.dtype == "complex128"

    def test_complex_mul_scalar(self):
        """Test multiplication of complex tree by complex scalar."""
        tree = vp.FunctionTree(mra_3d, dtype=complex)
        tree.setZero()

        result = tree * (2.0 + 1.0j)
        assert result.dtype == "complex128"

        result2 = (2.0 + 1.0j) * tree
        assert result2.dtype == "complex128"

    def test_complex_div_scalar(self):
        """Test division of complex tree by complex scalar."""
        tree = vp.FunctionTree(mra_3d, dtype=complex)
        tree.setZero()

        result = tree / (2.0 + 1.0j)
        assert result.dtype == "complex128"

    def test_complex_neg(self):
        """Test negation of complex tree."""
        tree = vp.FunctionTree(mra_3d, dtype=complex)
        tree.setZero()

        result = -tree
        assert result.dtype == "complex128"
        assert result.squaredNorm() == 0.0

    def test_complex_pos(self):
        """Test positive copy of complex tree."""
        tree = vp.FunctionTree(mra_3d, dtype=complex)
        tree.setZero()

        result = +tree
        assert result.dtype == "complex128"
        assert result.squaredNorm() == 0.0

    def test_complex_dot(self):
        """Test dot product of two complex trees."""
        tree_a = vp.FunctionTree(mra_3d, dtype=complex)
        tree_b = vp.FunctionTree(mra_3d, dtype=complex)
        tree_a.setZero()
        tree_b.setZero()

        result = vp.dot(tree_a, tree_b)
        assert result == 0.0 + 0.0j

    def test_mixed_dot_real_complex(self):
        """Test dot product of real and complex trees."""
        tree_real = vp.FunctionTree(mra_3d)
        tree_complex = vp.FunctionTree(mra_3d, dtype=complex)
        tree_real.setZero()
        tree_complex.setZero()

        result = vp.dot(tree_real, tree_complex)
        assert result == 0.0 + 0.0j

    def test_mixed_dot_complex_real(self):
        """Test dot product of complex and real trees."""
        tree_complex = vp.FunctionTree(mra_3d, dtype=complex)
        tree_real = vp.FunctionTree(mra_3d)
        tree_complex.setZero()
        tree_real.setZero()

        result = vp.dot(tree_complex, tree_real)
        assert result == 0.0 + 0.0j


class TestComplexProjection:
    """Tests for complex projection."""

    def test_scaling_projector_complex_creation(self):
        """Test creating a complex ScalingProjector."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)
        assert P is not None

    def test_scaling_projector_complex_with_scale(self):
        """Test creating a complex ScalingProjector with fixed scale."""
        P = vp.ScalingProjector(mra_3d, scale=0, dtype=complex)
        assert P is not None

    def test_wavelet_projector_complex_creation(self):
        """Test creating a complex WaveletProjector."""
        P = vp.WaveletProjector(mra_3d, scale=0, dtype=complex)
        assert P is not None

    def test_scaling_projector_project_complex_function(self):
        """Test projecting a complex-valued function."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)

        def complex_func(r):
            return (1.0 + 1.0j) * r[0]

        tree = P(complex_func)
        assert tree.dtype == "complex128"
        assert tree.is_complex is True

    def test_scaling_projector_project_pure_imaginary(self):
        """Test projecting a pure imaginary function."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)

        def imag_func(r):
            return 1.0j * (r[0] + r[1] + r[2])

        tree = P(imag_func)
        assert tree.dtype == "complex128"
        # The integral of i*(x+y+z) over [-1,1]^3 should be purely imaginary
        integral = tree.integrate()
        assert integral.real == pytest.approx(0.0, abs=1e-10)

    def test_wavelet_projector_project_complex_function(self):
        """Test projecting a complex function with WaveletProjector."""
        P = vp.WaveletProjector(mra_3d, scale=0, dtype=complex)

        def complex_func(r):
            return (1.0 + 2.0j)

        tree = P(complex_func)
        assert tree.dtype == "complex128"

    def test_complex_projection_1d(self):
        """Test complex projection in 1D."""
        P = vp.ScalingProjector(mra_1d, prec=1e-3, dtype=complex)

        def complex_func(r):
            return (1.0 + 1.0j) * r[0]

        tree = P(complex_func)
        assert tree.dtype == "complex128"

    def test_complex_projection_2d(self):
        """Test complex projection in 2D."""
        P = vp.ScalingProjector(mra_2d, prec=1e-3, dtype=complex)

        def complex_func(r):
            return (1.0 + 1.0j) * (r[0] + r[1])

        tree = P(complex_func)
        assert tree.dtype == "complex128"


class TestComplexConversion:
    """Tests for complex type conversion utilities."""

    def test_real_to_complex(self):
        """Test converting a real tree to complex."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3)

        def real_func(r):
            return r[0] + r[1] + r[2]

        real_tree = P(real_func)
        assert real_tree.dtype == "float64"

        complex_tree = real_tree.to_complex()
        assert complex_tree.dtype == "complex128"
        assert complex_tree.is_complex is True

        # Verify the values are the same (just with zero imaginary part)
        r = [0.1, 0.2, 0.3]
        real_val = real_tree(r)
        complex_val = complex_tree(r)
        assert complex_val.real == pytest.approx(real_val, rel=1e-10)
        assert complex_val.imag == pytest.approx(0.0, abs=1e-10)

    def test_extract_real_part(self):
        """Test extracting real part from complex tree."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)

        def complex_func(r):
            return (2.0 + 3.0j) * r[0]

        complex_tree = P(complex_func)
        real_tree = complex_tree.real()

        assert real_tree.dtype == "float64"
        assert real_tree.is_complex is False

        r = [0.5, 0.1, 0.1]
        # Real part of (2+3j)*0.5 = 1.0
        assert real_tree(r) == pytest.approx(1.0, rel=1e-3)

    def test_extract_imag_part(self):
        """Test extracting imaginary part from complex tree."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)

        def complex_func(r):
            return (2.0 + 3.0j) * r[0]

        complex_tree = P(complex_func)
        imag_tree = complex_tree.imag()

        assert imag_tree.dtype == "float64"
        assert imag_tree.is_complex is False

        r = [0.5, 0.1, 0.1]
        # Imag part of (2+3j)*0.5 = 1.5
        assert imag_tree(r) == pytest.approx(1.5, rel=1e-3)

    def test_conjugate_method(self):
        """Test the conj() method on complex tree."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)

        def complex_func(r):
            return (1.0 + 2.0j) * r[0]

        tree = P(complex_func)
        conj_tree = tree.conj()

        assert conj_tree.dtype == "complex128"

        r = [0.5, 0.1, 0.1]
        original = tree(r)
        conjugated = conj_tree(r)

        assert conjugated.real == pytest.approx(original.real, rel=1e-10)
        assert conjugated.imag == pytest.approx(-original.imag, rel=1e-10)

    def test_conjugate_function(self):
        """Test the vp.conj() function."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)

        def complex_func(r):
            return (1.0 + 2.0j) * r[0]

        tree = P(complex_func)
        conj_tree = vp.conj(tree)

        assert conj_tree.dtype == "complex128"

        r = [0.5, 0.1, 0.1]
        original = tree(r)
        conjugated = conj_tree(r)

        assert conjugated.real == pytest.approx(original.real, rel=1e-10)
        assert conjugated.imag == pytest.approx(-original.imag, rel=1e-10)

    def test_real_imag_roundtrip(self):
        """Test that extracting real+imag and combining gives back original."""
        P_complex = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)

        def complex_func(r):
            return (1.0 + 2.0j) * (r[0] + r[1])

        original = P_complex(complex_func)
        real_part = original.real()
        imag_part = original.imag()

        # Convert back: real + i*imag should equal original
        real_as_complex = real_part.to_complex()
        imag_as_complex = imag_part.to_complex()

        # Multiply imag by i
        reconstructed = real_as_complex + (1.0j) * imag_as_complex

        r = [0.3, 0.4, 0.1]
        original_val = original(r)
        reconstructed_val = reconstructed(r)

        assert reconstructed_val.real == pytest.approx(original_val.real, rel=1e-3)
        assert reconstructed_val.imag == pytest.approx(original_val.imag, rel=1e-3)


class TestComplexOperators:
    """Tests for complex operator application."""

    def test_derivative_operator_on_complex_tree_1d(self):
        """Test applying derivative operator to a 1D complex tree."""
        P = vp.ScalingProjector(mra_1d, prec=1e-4, dtype=complex)
        D = vp.ABGVDerivative(mra_1d, a=0.5, b=0.5)

        # f(x) = (1 + 2j) * x^2
        # f'(x) = (1 + 2j) * 2x = (2 + 4j) * x
        def complex_func(r):
            return (1.0 + 2.0j) * r[0] ** 2

        f = P(complex_func)
        assert f.dtype == "complex128"

        # Apply derivative using __call__
        df = D(f, axis=0)
        assert df.dtype == "complex128"

        # Check value at x=0.5: should be (2 + 4j) * 0.5 = 1 + 2j
        r = [0.5]
        df_val = df(r)
        expected = (2.0 + 4.0j) * 0.5
        assert df_val.real == pytest.approx(expected.real, rel=1e-3)
        assert df_val.imag == pytest.approx(expected.imag, rel=1e-3)

    def test_derivative_operator_on_complex_tree_3d(self):
        """Test applying derivative operator to a 3D complex tree."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)
        D = vp.ABGVDerivative(mra_3d, a=0.5, b=0.5)

        # f(x,y,z) = (1 + 1j) * x * y
        # df/dx = (1 + 1j) * y
        def complex_func(r):
            return (1.0 + 1.0j) * r[0] * r[1]

        f = P(complex_func)
        df_dx = D(f, axis=0)

        assert df_dx.dtype == "complex128"

        # Check at (0.3, 0.4, 0.1): df/dx = (1+1j) * 0.4
        r = [0.3, 0.4, 0.1]
        expected = (1.0 + 1.0j) * 0.4
        df_val = df_dx(r)
        assert df_val.real == pytest.approx(expected.real, rel=1e-2)
        assert df_val.imag == pytest.approx(expected.imag, rel=1e-2)

    def test_gradient_complex_3d(self):
        """Test gradient of a 3D complex tree."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)
        D = vp.ABGVDerivative(mra_3d, a=0.0, b=0.0)

        # f(x,y,z) = (1 + 2j) * (x + 2*y + 3*z)
        # grad f = (1+2j, 2+4j, 3+6j)
        def complex_func(r):
            return (1.0 + 2.0j) * (r[0] + 2.0 * r[1] + 3.0 * r[2])

        f = P(complex_func)
        grad_f = vp.gradient(oper=D, inp=f)

        assert len(grad_f) == 3
        for component in grad_f:
            assert component.dtype == "complex128"

        r = [0.1, 0.2, 0.3]

        # df/dx = 1 + 2j
        gx_val = grad_f[0](r)
        assert gx_val.real == pytest.approx(1.0, rel=1e-2)
        assert gx_val.imag == pytest.approx(2.0, rel=1e-2)

        # df/dy = 2 + 4j
        gy_val = grad_f[1](r)
        assert gy_val.real == pytest.approx(2.0, rel=1e-2)
        assert gy_val.imag == pytest.approx(4.0, rel=1e-2)

        # df/dz = 3 + 6j
        gz_val = grad_f[2](r)
        assert gz_val.real == pytest.approx(3.0, rel=1e-2)
        assert gz_val.imag == pytest.approx(6.0, rel=1e-2)

    def test_divergence_complex_3d(self):
        """Test divergence of complex vector field."""
        P = vp.ScalingProjector(mra_3d, prec=1e-3, dtype=complex)
        D = vp.ABGVDerivative(mra_3d, a=0.0, b=0.0)

        # F = ((1+j)*x, (2+j)*y, (3+j)*z)
        # div F = (1+j) + (2+j) + (3+j) = 6 + 3j
        def f_x(r):
            return (1.0 + 1.0j) * r[0]

        def f_y(r):
            return (2.0 + 1.0j) * r[1]

        def f_z(r):
            return (3.0 + 1.0j) * r[2]

        Fx = P(f_x)
        Fy = P(f_y)
        Fz = P(f_z)

        div_F = vp.divergence(oper=D, inp=[Fx, Fy, Fz])
        assert div_F is not None
        assert div_F.dtype == "complex128"

        r = [0.1, 0.2, 0.3]
        div_val = div_F(r)
        # div F = 6 + 3j (constant)
        assert div_val.real == pytest.approx(6.0, rel=1e-2)
        assert div_val.imag == pytest.approx(3.0, rel=1e-2)

    def test_ph_derivative_complex(self):
        """Test PH derivative on complex tree."""
        P = vp.ScalingProjector(mra_1d, prec=1e-4, dtype=complex)
        D = vp.PHDerivative(mra_1d, order=1)

        def complex_func(r):
            return (1.0 + 1.0j) * r[0] ** 2

        f = P(complex_func)
        df = D(f, axis=0)

        assert df.dtype == "complex128"

        # df/dx at x=0.3 should be (2 + 2j) * 0.3 = 0.6 + 0.6j
        r = [0.3]
        df_val = df(r)
        expected = (2.0 + 2.0j) * 0.3
        assert df_val.real == pytest.approx(expected.real, rel=1e-2)
        assert df_val.imag == pytest.approx(expected.imag, rel=1e-2)

    def test_bs_derivative_complex(self):
        """Test BS derivative on complex tree."""
        P = vp.ScalingProjector(mra_1d, prec=1e-4, dtype=complex)
        D = vp.BSDerivative(mra_1d, order=1)

        def complex_func(r):
            return (1.0 + 1.0j) * r[0] ** 2

        f = P(complex_func)
        df = D(f, axis=0)

        assert df.dtype == "complex128"

        # df/dx at x=0.3 should be (2 + 2j) * 0.3 = 0.6 + 0.6j
        # Note: BS derivative has slightly lower accuracy, use looser tolerance
        r = [0.3]
        df_val = df(r)
        expected = (2.0 + 2.0j) * 0.3
        assert df_val.real == pytest.approx(expected.real, rel=0.05)
        assert df_val.imag == pytest.approx(expected.imag, rel=0.05)

    def test_advanced_apply_derivative_complex(self):
        """Test advanced.apply with derivative operator on complex tree."""
        P = vp.ScalingProjector(mra_1d, prec=1e-4, dtype=complex)
        D = vp.ABGVDerivative(mra_1d, a=0.5, b=0.5)

        def complex_func(r):
            return (1.0 + 2.0j) * r[0] ** 2

        f = P(complex_func)
        df = vp.FunctionTree(mra_1d, dtype=complex)

        vp.advanced.apply(out=df, oper=D, inp=f, dir=0)

        assert df.dtype == "complex128"

        r = [0.5]
        df_val = df(r)
        expected = (2.0 + 4.0j) * 0.5
        assert df_val.real == pytest.approx(expected.real, rel=1e-3)
        assert df_val.imag == pytest.approx(expected.imag, rel=1e-3)

    def test_convolution_operator_complex_3d(self):
        """Test applying convolution operator to complex tree."""
        import numpy as np

        # Create a Gaussian convolution kernel
        epsilon = 1e-3
        a = 10.0
        b = np.sqrt(a / np.pi) ** 3

        # Create identity convolution approximation
        ifunc = vp.GaussFunc(alpha=a, beta=b, dim=1)
        iexp = vp.GaussExp(dim=1)
        iexp.append(ifunc)
        I = vp.ConvolutionOperator(mra_3d, iexp, prec=epsilon)

        # Create complex input
        P = vp.ScalingProjector(mra_3d, prec=epsilon, dtype=complex)

        def complex_func(r):
            return (1.0 + 0.5j) * np.exp(-a * (r[0]**2 + r[1]**2 + r[2]**2))

        f = P(complex_func)
        g = vp.FunctionTree(mra_3d, dtype=complex)

        # Apply convolution
        vp.advanced.apply(prec=epsilon, out=g, oper=I, inp=f)

        # Verify the result is complex
        assert g.dtype == "complex128"

        # Verify the result has non-zero values (convolution produced something)
        g_integral = g.integrate()
        assert isinstance(g_integral, complex)
        assert g_integral != 0.0 + 0.0j

        # Verify the imaginary/real ratio is preserved approximately
        # (convolution is a linear operation)
        f_integral = f.integrate()
        if abs(f_integral.real) > 1e-10:
            ratio_f = f_integral.imag / f_integral.real
            ratio_g = g_integral.imag / g_integral.real
            assert ratio_g == pytest.approx(ratio_f, rel=0.2)
