import warnings

import numpy as np
import nengo
from nengo.connection import Connection
from nengo.exceptions import ObsoleteError, ValidationError
from nengo.network import Network
from nengo.node import Node
from nengo.networks.ensemblearray import EnsembleArray


def circconv(a, b, invert_a=False, invert_b=False, axis=-1):
    """A reference Numpy implementation of circular convolution."""
    A = np.fft.fft(a, axis=axis)
    B = np.fft.fft(b, axis=axis)
    if invert_a:
        A = A.conj()
    if invert_b:
        B = B.conj()
    return np.fft.ifft(A * B, axis=axis).real


def transform_in(dims, align, invert):
    """
    Create a transform to map the input into the Fourier domain.

    See CircularConvolution docstring for more details.

    Parameters
    ----------
    dims : int
        Input dimensions.
    align : 'A' or 'B'
        How to align the real and imaginary components; the alignment
        depends on whether we're doing transformA or transformB.
    invert : bool
        Whether to reverse the order of elements.
    """
    if align not in ("A", "B"):
        raise ValidationError("'align' must be either 'A' or 'B'", "align")

    dims2 = 4 * (dims // 2 + 1)
    tr = np.zeros((dims2, dims))
    dft = dft_half(dims)

    for i in range(dims2):
        row = dft[i // 4] if not invert else dft[i // 4].conj()
        if align == "A":
            tr[i] = row.real if i % 2 == 0 else row.imag
        else:  # align == 'B'
            tr[i] = row.real if i % 4 == 0 or i % 4 == 3 else row.imag

    remove_imag_rows(tr)
    return tr.reshape((-1, dims))


def transform_out(dims):
    dims2 = dims // 2 + 1
    tr = np.zeros((dims2, 4, dims))
    idft = dft_half(dims).conj()

    for i in range(dims2):
        row = idft[i] if i == 0 or 2 * i == dims else 2 * idft[i]
        tr[i, 0] = row.real
        tr[i, 1] = -row.real
        tr[i, 2] = -row.imag
        tr[i, 3] = -row.imag

    tr = tr.reshape(4 * dims2, dims)
    remove_imag_rows(tr)
    # IDFT has a 1/D scaling factor
    tr /= dims

    return tr.T


def remove_imag_rows(tr):
    """Throw away imaginary rows we do not need (they are zero)."""
    i = np.arange(tr.shape[0])
    if tr.shape[1] % 2 == 0:
        tr = tr[(i == 0) | (i > 3) & (i < len(i) - 3)]
    else:
        tr = tr[(i == 0) | (i > 3)]


def dft_half(n):
    x = np.arange(n)
    w = np.arange(n // 2 + 1)
    return np.exp((-2.0j * np.pi / n) * (w[:, None] * x[None, :]))


class CircularConvolution(Network):
    r"""
    Compute the circular convolution of two vectors.

    The circular convolution :math:`c` of vectors :math:`a` and :math:`b`
    is given by

    .. math:: c[i] = \sum_j a[j] b[i - j]

    where negative indices on :math:`b` wrap around to the end of the vector.

    This computation can also be done in the Fourier domain,

    .. math:: c = DFT^{-1} ( DFT(a) DFT(b) )

    where :math:`DFT` is the Discrete Fourier Transform operator, and
    :math:`DFT^{-1}` is its inverse. This network uses this method.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to use in each product computation
    dimensions : int
        The number of dimensions of the input and output vectors.

    invert_a, invert_b : bool, optional
        Whether to reverse the order of elements in either
        the first input (``invert_a``) or the second input (``invert_b``).
        Flipping the second input will make the network perform circular
        correlation instead of circular convolution.
    input_magnitude : float, optional
        The expected magnitude of the vectors to be convolved.
        This value is used to determine the radius of the ensembles
        computing the element-wise product.
    **kwargs
        Keyword arguments passed through to ``nengo.Network``
        like 'label' and 'seed'.

    Attributes
    ----------
    input_a : Node
        The first vector to be convolved.
    input_b : Node
        The second vector to be convolved.
    product : Network
        Network created with `.Product` to do the element-wise product
        of the :math:`DFT` components.
    output : Node
        The resulting convolved vector.

    Examples
    --------

    A basic example computing the circular convolution of two 10-dimensional
    vectors represented by ensemble arrays:

    .. testcode::

       from nengo.networks import CircularConvolution, EnsembleArray

       with nengo.Network():
           A = EnsembleArray(50, n_ensembles=10)
           B = EnsembleArray(50, n_ensembles=10)
           C = EnsembleArray(50, n_ensembles=10)
           cconv = CircularConvolution(50, dimensions=10)
           nengo.Connection(A.output, cconv.input_a)
           nengo.Connection(B.output, cconv.input_b)
           nengo.Connection(cconv.output, C.input)

    Notes
    -----

    The network maps the input vectors :math:`a` and :math:`b` of length N into
    the Fourier domain and aligns them for complex multiplication.
    Letting :math:`F = DFT(a)` and :math:`G = DFT(b)`, this is given by::

        [ F[i].real ]     [ G[i].real ]     [ w[i] ]
        [ F[i].imag ]  *  [ G[i].imag ]  =  [ x[i] ]
        [ F[i].real ]     [ G[i].imag ]     [ y[i] ]
        [ F[i].imag ]     [ G[i].real ]     [ z[i] ]

    where :math:`i` only ranges over the lower half of the spectrum, since
    the upper half of the spectrum is the flipped complex conjugate of
    the lower half, and therefore redundant. The input transforms are
    used to perform the DFT on the inputs and align them correctly for
    complex multiplication.

    The complex product :math:`H = F * G` is then

    .. math:: H[i] = (w[i] - x[i]) + (y[i] + z[i]) I

    where :math:`I = \sqrt{-1}`. We can perform this addition along with the
    inverse DFT :math:`c = DFT^{-1}(H)` in a single output transform, finding
    only the real part of :math:`c` since the imaginary part
    is analytically zero.
    """

    def __init__(
        self,
        n_neurons,
        dimensions,
        invert_a=False,
        invert_b=False,
        input_magnitude=1.0,
        label='circonv',solver=nengo.Default,
        **kwargs,
    ):
        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        super().__init__(label=label,**kwargs)

        tr_a = transform_in(dimensions, "A", invert_a)
        tr_b = transform_in(dimensions, "B", invert_b)
        tr_out = transform_out(dimensions)

        with self:
            self.input_a = Node(size_in=dimensions, label=label+"_input_a")
            self.input_b = Node(size_in=dimensions, label=label+"_input_b")
            self.product = Product(
                n_neurons, tr_out.shape[1], input_magnitude=input_magnitude * 2,
                label=label+'_product', solver=solver
            )
            self.output = Node(size_in=dimensions, label=label+"_output")

            Connection(self.input_a, self.product.input_a, transform=tr_a, synapse=None)
            Connection(self.input_b, self.product.input_b, transform=tr_b, synapse=None)
            Connection(self.product.output, self.output, transform=tr_out, synapse=None)

    @property
    def A(self):
        warnings.warn(DeprecationWarning("Use 'input_a' instead of 'A'"))
        return self.input_a

    @property
    def B(self):
        warnings.warn(DeprecationWarning("Use 'input_b' instead of 'B'."))
        return self.input_b
    
    


class Product(Network):
    """
    Computes the element-wise product of two equally sized vectors.

    The network used to calculate the product is described in
    `Gosmann, 2015`_. A simpler version of this network can be found in the
    :doc:`Multiplication example <examples/basic/multiplication>`.

    Note that this network is optimized under the assumption that both input
    values (or both values for each input dimensions of the input vectors) are
    uniformly and independently distributed. Visualized in a joint 2D space,
    this would give a square of equal probabilities for pairs of input values.
    This assumption is violated with non-uniform input value distributions
    (for example, if the input values follow a Gaussian or cosine similarity
    distribution). In that case, no square of equal probabilities is obtained,
    but a probability landscape with circular equi-probability lines. To obtain
    the optimal network accuracy, scale the *input_magnitude* by a factor of
    ``1 / sqrt(2)``.

    .. _Gosmann, 2015:
       https://nbviewer.org/github/ctn-archive/technical-reports/blob/master/Precise-multiplications-with-the-NEF.ipynb

    Parameters
    ----------
    n_neurons : int
        Number of neurons per dimension in the vector.

        .. note:: These neurons will be distributed evenly across two
                  ensembles. If an odd number of neurons is specified, the
                  extra neuron will not be used.
    dimensions : int
        Number of dimensions in each of the vectors to be multiplied.

    input_magnitude : float, optional
        The expected magnitude of the vectors to be multiplied.
        This value is used to determine the radius of the ensembles
        computing the element-wise product.
    **kwargs
        Keyword arguments passed through to ``nengo.Network``
        like 'label' and 'seed'.

    Attributes
    ----------
    input_a : Node
        The first vector to be multiplied.
    input_b : Node
        The second vector to be multiplied.
    output : Node
        The resulting product.
    sq1 : EnsembleArray
        Represents the first squared term. See `Gosmann, 2015`_ for details.
    sq2 : EnsembleArray
        Represents the second squared term. See `Gosmann, 2015`_ for details.
    """

    def __init__(self, n_neurons, dimensions, input_magnitude=1.0,
                 dot_product=False, label='product',solver=nengo.Default, **kwargs):
        super().__init__(label=label,**kwargs)

        with self:
            self.input_a = Node(size_in=dimensions, label=label+"_input_a")
            self.input_b = Node(size_in=dimensions, label=label+"_input_b")
            self.output = Node(size_in=dimensions, label=label+"_output")

            self.sq1 = EnsembleArray(
                max(1, n_neurons // 2),
                n_ensembles=dimensions,
                ens_dimensions=1,
                radius=input_magnitude * np.sqrt(2), label=label+"_sq1"
            )
            self.sq2 = EnsembleArray(
                max(1, n_neurons // 2),
                n_ensembles=dimensions,
                ens_dimensions=1,
                radius=input_magnitude * np.sqrt(2),label=label+"_sq2"
            )

            tr = 1.0 / np.sqrt(2.0)
            Connection(self.input_a, self.sq1.input, transform=tr, synapse=None)
            Connection(self.input_b, self.sq1.input, transform=tr, synapse=None)
            Connection(self.input_a, self.sq2.input, transform=tr, synapse=None)
            Connection(self.input_b, self.sq2.input, transform=-tr, synapse=None)

            sq1_out = self.sq1.add_output("square", np.square, solver=solver)
            sq2_out = self.sq2.add_output("square", np.square, solver=solver)

            if dot_product:
                Connection(sq1_out, self.output, transform=dot_product_transform(dimensions,0.5), synapse=None)
                Connection(sq2_out, self.output, transform=dot_product_transform(dimensions,-0.5), synapse=None)
            else:
                Connection(sq1_out, self.output, transform=0.5, synapse=None)
                Connection(sq2_out, self.output, transform=-0.5, synapse=None)

    @property
    def A(self):  # pragma: no cover
        warnings.warn(DeprecationWarning("Use 'input_a' instead of 'A'."))
        return self.input_a

    @property
    def B(self):  # pragma: no cover
        warnings.warn(DeprecationWarning("Use 'input_b' instead of 'B'."))
        return self.input_b


def dot_product_transform(dimensions, scale=1.0):
    """Returns a transform for output to compute the scaled dot product."""
    return scale * np.ones((1, dimensions))