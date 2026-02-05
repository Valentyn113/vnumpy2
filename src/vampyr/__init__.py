# -*- coding: utf-8 -*-

from ._vampyr import *
from .environ import _set_mwfilters_path

__version__ = _vampyr.__version__
__doc__ = _vampyr.__doc__

def BoundingBox(*args, **kwargs):
    # Normalize scalar kwargs to 1-element lists for 1D convenience
    for key in ['corner', 'nboxes']:
        if key in kwargs and isinstance(kwargs[key], (int, float)):
            kwargs[key] = [int(kwargs[key])]
    if 'scaling' in kwargs and isinstance(kwargs['scaling'], (int, float)):
        kwargs['scaling'] = [float(kwargs['scaling'])]

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
        if d == 1: return FunctionTree1D_Complex(mra, *args, **kwargs)
        if d == 2: return FunctionTree2D_Complex(mra, *args, **kwargs)
        if d == 3: return FunctionTree3D_Complex(mra, *args, **kwargs)
    else:
        if d == 1: return FunctionTree1D(mra, *args, **kwargs)
        if d == 2: return FunctionTree2D(mra, *args, **kwargs)
        if d == 3: return FunctionTree3D(mra, *args, **kwargs)

    raise ValueError(f"Unsupported dimension: {d}")

def Gaussian(*args, **kwargs):
    # Normalize scalar kwargs to 1-element lists for 1D convenience
    for key in ['position', 'poly_exponent']:
        if key in kwargs and isinstance(kwargs[key], (int, float)):
            kwargs[key] = [kwargs[key]]
    # Normalize scalar positional args (position=args[2], poly_exponent=args[3])
    args = list(args)
    if len(args) > 2 and isinstance(args[2], (int, float)):
        args[2] = [args[2]]
    if len(args) > 3 and isinstance(args[3], (int, float)):
        args[3] = [args[3]]
    args = tuple(args)

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
    # Normalize scalar kwargs to 1-element lists for 1D convenience
    for key in ['position', 'poly_exponent']:
        if key in kwargs and isinstance(kwargs[key], (int, float)):
            kwargs[key] = [kwargs[key]]
    # Normalize scalar positional args (position=args[2], poly_exponent=args[3])
    args = list(args)
    if len(args) > 2 and isinstance(args[2], (int, float)):
        args[2] = [args[2]]
    if len(args) > 3 and isinstance(args[3], (int, float)):
        args[3] = [args[3]]
    args = tuple(args)

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
    # Normalize scalar translation to 1-element list for 1D convenience
    if 'translation' in kwargs and isinstance(kwargs['translation'], int):
        kwargs['translation'] = [kwargs['translation']]
    args = list(args)
    if len(args) > 1 and isinstance(args[1], int):
        args[1] = [args[1]]
    args = tuple(args)

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

def _wrap_1d_func(f):
    """Wrap a scalar 1D function f(x) -> f([x]) for the C++ projector interface.

    Tries calling f with a scalar first. If that works, wraps it so the C++
    side (which passes [x]) still works. If f already expects a vector [x],
    returns it unchanged. This preserves backward compatibility.
    """
    try:
        # Use the same test value as the C++ projector validation
        f(111111.111)
    except (TypeError, IndexError):
        # f expects a list/array argument like r[0] — no wrapping needed
        return f
    except Exception:
        # Other errors (e.g. domain errors) — f still accepted a scalar
        pass
    # f accepts a scalar — wrap it to unpack the 1-element list from C++
    def wrapped(r):
        return f(r[0])
    return wrapped

def ScalingProjector(mra, *args, dtype=None, **kwargs):
    """Create a ScalingProjector.

    Parameters
    ----------
    mra : MultiResolutionAnalysis
        The multi-resolution analysis.
    prec : float, optional
        Precision for adaptive projection.
    scale : int, optional
        Fixed scale for projection.
    dtype : type or str, optional
        Data type: None/'float64' (default) or complex/'complex128'.

    Returns
    -------
    ScalingProjector
        A projector of the appropriate dimension and dtype.

    Notes
    -----
    In 1D, analytic functions can be defined as ``lambda x: x**2``
    (scalar) or ``lambda r: r[0]**2`` (vector). Both forms are supported.
    """
    d = mra.dimension
    is_complex = dtype in (complex, 'complex', 'complex128')

    if is_complex:
        if d == 1:
            proj = ScalingProjector1D_Complex(mra, *args, **kwargs)
            return _Projector1DWrapper(proj)
        if d == 2: return ScalingProjector2D_Complex(mra, *args, **kwargs)
        if d == 3: return ScalingProjector3D_Complex(mra, *args, **kwargs)
    else:
        if d == 1:
            proj = ScalingProjector1D(mra, *args, **kwargs)
            return _Projector1DWrapper(proj)
        if d == 2: return ScalingProjector2D(mra, *args, **kwargs)
        if d == 3: return ScalingProjector3D(mra, *args, **kwargs)

    raise ValueError(f"Unsupported dimension: {d}")

def WaveletProjector(mra, *args, dtype=None, **kwargs):
    """Create a WaveletProjector.

    Parameters
    ----------
    mra : MultiResolutionAnalysis
        The multi-resolution analysis.
    scale : int
        Fixed scale for projection.
    dtype : type or str, optional
        Data type: None/'float64' (default) or complex/'complex128'.

    Returns
    -------
    WaveletProjector
        A projector of the appropriate dimension and dtype.

    Notes
    -----
    In 1D, analytic functions can be defined as ``lambda x: x**2``
    (scalar) or ``lambda r: r[0]**2`` (vector). Both forms are supported.
    """
    d = mra.dimension
    is_complex = dtype in (complex, 'complex', 'complex128')

    if is_complex:
        if d == 1:
            proj = WaveletProjector1D_Complex(mra, *args, **kwargs)
            return _Projector1DWrapper(proj)
        if d == 2: return WaveletProjector2D_Complex(mra, *args, **kwargs)
        if d == 3: return WaveletProjector3D_Complex(mra, *args, **kwargs)
    else:
        if d == 1:
            proj = WaveletProjector1D(mra, *args, **kwargs)
            return _Projector1DWrapper(proj)
        if d == 2: return WaveletProjector2D(mra, *args, **kwargs)
        if d == 3: return WaveletProjector3D(mra, *args, **kwargs)

    raise ValueError(f"Unsupported dimension: {d}")

class _Projector1DWrapper:
    """Wrapper for 1D projectors that auto-adapts scalar functions.

    Allows users to write ``projector(lambda x: x**2)`` instead of
    ``projector(lambda r: r[0]**2)`` while keeping full backward compatibility.
    For RepresentableFunction objects (like GaussFunc), passes them through directly.
    """
    def __init__(self, proj):
        self._proj = proj

    def __call__(self, f):
        if callable(f) and not hasattr(f, 'evalf'):
            f = _wrap_1d_func(f)
        return self._proj(f)

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
