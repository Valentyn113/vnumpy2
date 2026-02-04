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
