# -*- coding: utf-8 -*-

from ._vampyr import *
from .environ import _set_mwfilters_path

__version__ = _vampyr.__version__
__doc__ = _vampyr.__doc__

def BoundingBox(*args, **kwargs):
    d = kwargs.pop('dim', None)
    if d is not None:
        if d == 1: return BoundingBox1D(*args, **kwargs)
        if d == 2: return BoundingBox2D(*args, **kwargs)
        if d == 3: return BoundingBox3D(*args, **kwargs)

    for arg in args:
        if isinstance(arg, (list, tuple)) and len(arg) > 0:
            if isinstance(arg[0], (int, float)):
                d = len(arg)
                if d == 1: return BoundingBox1D(*args, **kwargs)
                if d == 2: return BoundingBox2D(*args, **kwargs)
                if d == 3: return BoundingBox3D(*args, **kwargs)

    for key in ['corner', 'nboxes', 'scaling']:
        if key in kwargs:
            val = kwargs[key]
            d = len(val)
            if d == 1: return BoundingBox1D(*args, **kwargs)
            if d == 2: return BoundingBox2D(*args, **kwargs)
            if d == 3: return BoundingBox3D(*args, **kwargs)

    if len(args) > 1 and isinstance(args[1], (list, tuple)):
         d = len(args[1])
         if d == 1: return BoundingBox1D(*args, **kwargs)
         if d == 2: return BoundingBox2D(*args, **kwargs)
         if d == 3: return BoundingBox3D(*args, **kwargs)

    raise ValueError("Could not infer dimension for BoundingBox. Please ensure 'corner', 'nboxes', 'scaling' or 'dim' is provided.")

def MultiResolutionAnalysis(*args, **kwargs):
    d = kwargs.pop('dim', None)
    if d is None:
        if len(args) > 0:
            box = args[0]
            if hasattr(box, "dimension"):
                d = box.dimension
            elif isinstance(box, (list, tuple)):
                 if len(box) == 2: d = 1
                 else: raise ValueError("List argument for MRA box must be length 2 (1D interval). Use BoundingBox for higher dims.")
        elif 'box' in kwargs:
            box = kwargs['box']
            if hasattr(box, "dimension"):
                d = box.dimension
            elif isinstance(box, (list, tuple)):
                 if len(box) == 2: d = 1

    if d is None:
        raise ValueError("MultiResolutionAnalysis requires 'box' argument (BoundingBox or list) or 'dim'.")

    if d == 1: return MultiResolutionAnalysis1D(*args, **kwargs)
    if d == 2: return MultiResolutionAnalysis2D(*args, **kwargs)
    if d == 3: return MultiResolutionAnalysis3D(*args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

def FunctionTree(mra, *args, dtype=None, **kwargs):
    """Create a FunctionTree with the given MRA.

    Parameters
    ----------
    mra : MultiResolutionAnalysis
        The multi-resolution analysis defining the computational domain.
    dtype : type or str, optional
        Data type for the tree. Options:
        - None, float, 'float64' (default): Real-valued tree
        - complex, 'complex', 'complex128': Complex-valued tree

    Returns
    -------
    FunctionTree
        A FunctionTree of the appropriate dimension and dtype.
    """
    d = mra.dimension
    is_complex = dtype in (complex, 'complex', 'complex128')

    if is_complex:
        if d == 1: raise NotImplementedError("Complex FunctionTree1D not yet implemented")
        if d == 2: raise NotImplementedError("Complex FunctionTree2D not yet implemented")
        if d == 3: return FunctionTree3D_Complex(mra, *args, **kwargs)
    else:
        if d == 1: return FunctionTree1D(mra, *args, **kwargs)
        if d == 2: return FunctionTree2D(mra, *args, **kwargs)
        if d == 3: return FunctionTree3D(mra, *args, **kwargs)

    raise ValueError(f"Unsupported dimension: {d}")

def Gaussian(*args, **kwargs):
    d = kwargs.pop('dim', None)
    if d is None:
        if len(args) > 2:
            d = len(args[2])
        elif 'position' in kwargs:
            d = len(kwargs['position'])

        if d is None:
            if len(args) > 3:
                d = len(args[3])
            elif 'poly_exponent' in kwargs:
                d = len(kwargs['poly_exponent'])

    if d is None:
         raise ValueError("Could not infer dimension for Gaussian. Please provide 'position', 'poly_exponent', or 'dim'.")

    if d == 1: return Gaussian1D(*args, **kwargs)
    if d == 2: return Gaussian2D(*args, **kwargs)
    if d == 3: return Gaussian3D(*args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

def GaussFunc(*args, **kwargs):
    d = kwargs.pop('dim', None)
    if d is None:
        if len(args) > 2: d = len(args[2])
        elif 'position' in kwargs: d = len(kwargs['position'])

        if d is None:
            if len(args) > 3: d = len(args[3])
            elif 'poly_exponent' in kwargs: d = len(kwargs['poly_exponent'])

    if d is None:
        raise ValueError("Could not infer dimension for GaussFunc. Please provide 'position', 'poly_exponent', or 'dim'.")

    if d == 1: return GaussFunc1D(*args, **kwargs)
    if d == 2: return GaussFunc2D(*args, **kwargs)
    if d == 3: return GaussFunc3D(*args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

def GaussExp(*args, **kwargs):
    d = kwargs.pop('dim', None)
    if d is None:
        # Default to 3D if no args, or try to infer if positional args provided (rare for GaussExp)
        if len(args) == 0:
            return GaussExp3D(*args, **kwargs)
        raise ValueError("Could not infer dimension for GaussExp. Please provide 'dim'.")

    if d == 1: return GaussExp1D(*args, **kwargs)
    if d == 2: return GaussExp2D(*args, **kwargs)
    if d == 3: return GaussExp3D(*args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

def ConvolutionOperator(mra, *args, **kwargs):
    d = mra.dimension
    if d == 1: return ConvolutionOperator1D(mra, *args, **kwargs)
    if d == 2: return ConvolutionOperator2D(mra, *args, **kwargs)
    if d == 3: return ConvolutionOperator3D(mra, *args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

def IdentityConvolution(mra, *args, **kwargs):
    d = mra.dimension
    if d == 1: return IdentityConvolution1D(mra, *args, **kwargs)
    if d == 2: return IdentityConvolution2D(mra, *args, **kwargs)
    if d == 3: return IdentityConvolution3D(mra, *args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

def ABGVDerivative(mra, *args, **kwargs):
    d = mra.dimension
    if d == 1: return ABGVDerivative1D(mra, *args, **kwargs)
    if d == 2: return ABGVDerivative2D(mra, *args, **kwargs)
    if d == 3: return ABGVDerivative3D(mra, *args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

def PHDerivative(mra, *args, **kwargs):
    d = mra.dimension
    if d == 1: return PHDerivative1D(mra, *args, **kwargs)
    if d == 2: return PHDerivative2D(mra, *args, **kwargs)
    if d == 3: return PHDerivative3D(mra, *args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

def BSDerivative(mra, *args, **kwargs):
    d = mra.dimension
    if d == 1: return BSDerivative1D(mra, *args, **kwargs)
    if d == 2: return BSDerivative2D(mra, *args, **kwargs)
    if d == 3: return BSDerivative3D(mra, *args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

def NodeIndex(*args, **kwargs):
    d = kwargs.pop('dim', None)
    if d is None:
        if len(args) > 1: d = len(args[1])
        elif 'translation' in kwargs: d = len(kwargs['translation'])

    if d == 1: return NodeIndex1D(*args, **kwargs)
    if d == 2: return NodeIndex2D(*args, **kwargs)
    if d == 3: return NodeIndex3D(*args, **kwargs)
    raise ValueError("Could not infer dimension for NodeIndex. Please provide 'translation' or 'dim'.")

def TreeIterator(*args, **kwargs):
    d = kwargs.pop('dim', None)
    if d is None:
        if len(args) > 0 and hasattr(args[0], "MRA"):
             d = args[0].MRA().dimension
        elif 'tree' in kwargs:
             d = kwargs['tree'].MRA().dimension

    if d == 1: return TreeIterator1D(*args, **kwargs)
    if d == 2: return TreeIterator2D(*args, **kwargs)
    if d == 3: return TreeIterator3D(*args, **kwargs)
    raise ValueError("Could not infer dimension for TreeIterator. Please provide 'tree' or 'dim'.")

def ScalingProjector(mra, *args, **kwargs):
    d = mra.dimension
    if d == 1: return ScalingProjector1D(mra, *args, **kwargs)
    if d == 2: return ScalingProjector2D(mra, *args, **kwargs)
    if d == 3: return ScalingProjector3D(mra, *args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

def WaveletProjector(mra, *args, **kwargs):
    d = mra.dimension
    if d == 1: return WaveletProjector1D(mra, *args, **kwargs)
    if d == 2: return WaveletProjector2D(mra, *args, **kwargs)
    if d == 3: return WaveletProjector3D(mra, *args, **kwargs)
    raise ValueError(f"Unsupported dimension: {d}")

class FunctionMap:
    def __init__(self, fmap, prec, dim=None):
        self.fmap = fmap
        self.prec = prec
        self.dim = dim
        self._impl = None
        if dim is not None:
             if dim == 1: self._impl = FunctionMap1D(fmap, prec)
             elif dim == 2: self._impl = FunctionMap2D(fmap, prec)
             elif dim == 3: self._impl = FunctionMap3D(fmap, prec)
             else: raise ValueError(f"Unsupported dimension: {dim}")

    def __call__(self, inp):
        if self._impl is None:
             d = inp.MRA().dimension
             if d == 1: self._impl = FunctionMap1D(self.fmap, self.prec)
             elif d == 2: self._impl = FunctionMap2D(self.fmap, self.prec)
             elif d == 3: self._impl = FunctionMap3D(self.fmap, self.prec)
             else: raise ValueError(f"Unsupported dimension: {d}")

        return self._impl(inp)

_set_mwfilters_path()
