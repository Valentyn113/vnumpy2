"""Tests for complex number support in VAMPyR (Phase 1: FunctionTree3D_Complex)."""

import vampyr as vp


# Test setup
D = 3
k = 5
N = -1
world = vp.BoundingBox(dim=D, scale=N)
mra = vp.MultiResolutionAnalysis(box=world, order=k)


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
