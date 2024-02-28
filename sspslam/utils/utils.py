import numpy as np
import math
import scipy 

def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))


def get_mean_and_ci(raw_data, n=3000, p=0.95):
    """
    Gets the mean and 95% confidence intervals of data *see Note
    NOTE: data has to be grouped along rows, for example: having 5 sets of
    100 data points would be a list of shape (5,100)
    """
    sample = []
    upper_bound = []
    lower_bound = []
    sets = np.array(raw_data).shape[0]  # pylint: disable=E1136
    data_pts = np.array(raw_data).shape[1]  # pylint: disable=E1136
    print("Mean and CI calculation found %i sets of %i data points" % (sets, data_pts))
    raw_data = np.array(raw_data)
    for i in range(data_pts):
        data = raw_data[:, i]
        index = int(n * (1 - p) / 2)
        samples = np.random.choice(data, size=(n, len(data)))
        r = [np.mean(s) for s in samples]
        r.sort()
        ci = r[index], r[-index]
        sample.append(np.mean(data))
        lower_bound.append(ci[0])
        upper_bound.append(ci[1])

    data = {"mean": sample, "lower_bound": lower_bound, "upper_bound": upper_bound}
    return data


def Rd_sampling(n,d,seed=0.5):
    def phi(d): 
        x=2.0000 
        for i in range(10): 
          x = pow(1+x,1/(d+1)) 
        return x
    g = phi(d) 
    alpha = np.zeros(d) 
    for j in range(d): 
      alpha[j] = pow(1/g,j+1) %1 
    z = np.zeros((n, d)) 
    for i in range(n):
        z[i] = seed + alpha*(i+1)
    z = z %1
    return z




# All from nengolib. copied here to avoid conflicts with nengo v3
import warnings

from scipy.special import beta, betainc, betaincinv
from nengo.dists import Distribution, UniformHypersphere


def spherical_transform(samples):
    """Map samples from the ``[0, 1]``--cube onto the hypersphere.
    Applies the `inverse transform method` to the distribution
    :class:`.SphericalCoords` to map uniform samples from the ``[0, 1]``--cube
    onto the surface of the hypersphere. [#]_
    Parameters
    ----------
    samples : ``(n, d) array_like``
        ``n`` uniform samples from the d-dimensional ``[0, 1]``--cube.
    Returns
    -------
    mapped_samples : ``(n, d+1) np.array``
        ``n`` uniform samples from the ``d``--dimensional sphere
        (Euclidean dimension of ``d+1``).
    See Also
    --------
    :class:`.Rd`
    :class:`.Sobol`
    :class:`.ScatteredHypersphere`
    :class:`.SphericalCoords`
    References
    ----------
    .. [#] K.-T. Fang and Y. Wang, Number-Theoretic Methods in Statistics.
       Chapman & Hall, 1994.
    Examples
    --------
    >>> from nengolib.stats import spherical_transform
    In the simplest case, we can map a one-dimensional uniform distribution
    onto a circle:
    >>> line = np.linspace(0, 1, 20)
    >>> mapped = spherical_transform(line)
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(6, 3))
    >>> plt.subplot(121)
    >>> plt.title("Original")
    >>> plt.scatter(line, np.zeros_like(line), s=30)
    >>> plt.subplot(122)
    >>> plt.title("Mapped")
    >>> plt.scatter(*mapped.T, s=25)
    >>> plt.show()
    This technique also generalizes to less trivial situations, for instance
    mapping a square onto a sphere:
    >>> square = np.asarray([[x, y] for x in np.linspace(0, 1, 50)
    >>>                             for y in np.linspace(0, 1, 10)])
    >>> mapped = spherical_transform(square)
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> plt.figure(figsize=(6, 3))
    >>> plt.subplot(121)
    >>> plt.title("Original")
    >>> plt.scatter(*square.T, s=15)
    >>> ax = plt.subplot(122, projection='3d')
    >>> ax.set_title("Mapped").set_y(1.)
    >>> ax.patch.set_facecolor('white')
    >>> ax.set_xlim3d(-1, 1)
    >>> ax.set_ylim3d(-1, 1)
    >>> ax.set_zlim3d(-1, 1)
    >>> ax.scatter(*mapped.T, s=15)
    >>> plt.show()
    """

    samples = np.asarray(samples)
    samples = samples[:, None] if samples.ndim == 1 else samples
    coords = np.empty_like(samples)
    n, d = coords.shape

    # inverse transform method (section 1.5.2)
    for j in range(d):
        coords[:, j] = SphericalCoords(d-j).ppf(samples[:, j])

    # spherical coordinate transform
    mapped = np.ones((n, d+1))
    i = np.ones(d)
    i[-1] = 2.0
    s = np.sin(i[None, :] * np.pi * coords)
    c = np.cos(i[None, :] * np.pi * coords)
    mapped[:, 1:] = np.cumprod(s, axis=1)
    mapped[:, :-1] *= c
    return mapped


class SphericalCoords(Distribution):
    """Spherical coordinates for inverse transform method.
    This is used to map the hypercube onto the hypersphere and hyperball. [#]_
    Parameters
    ----------
    m : ``integer``
        Positive index for spherical coordinate.
    See Also
    --------
    :func:`.spherical_transform`
    :class:`nengo.dists.SqrtBeta`
    References
    ----------
    .. [#] K.-T. Fang and Y. Wang, Number-Theoretic Methods in Statistics.
       Chapman & Hall, 1994.
    Examples
    --------
    >>> from nengolib.stats import SphericalCoords
    >>> coords = SphericalCoords(3)
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 1, 1000)
    >>> plt.figure(figsize=(8, 8))
    >>> plt.subplot(411)
    >>> plt.title(str(coords))
    >>> plt.ylabel("Samples")
    >>> plt.hist(coords.sample(1000), bins=50, normed=True)
    >>> plt.subplot(412)
    >>> plt.ylabel("PDF")
    >>> plt.plot(x, coords.pdf(x))
    >>> plt.subplot(413)
    >>> plt.ylabel("CDF")
    >>> plt.plot(x, coords.cdf(x))
    >>> plt.subplot(414)
    >>> plt.ylabel("PPF")
    >>> plt.plot(x, coords.ppf(x))
    >>> plt.xlabel("x")
    >>> plt.show()
    """

    def __init__(self, m):
        super(SphericalCoords, self).__init__()
        self.m = m

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.m)

    def sample(self, n, d=None, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        shape = self._sample_shape(n, d)
        y = rng.uniform(size=shape)
        return self.ppf(y)

    def pdf(self, x):
        """Evaluates the PDF along the values ``x``."""
        return (np.pi * np.sin(np.pi * x) ** (self.m-1) /
                beta(self.m / 2., .5))

    def cdf(self, x):
        """Evaluates the CDF along the values ``x``."""
        y = .5 * betainc(self.m / 2., .5, np.sin(np.pi * x) ** 2)
        return np.where(x < .5, y, 1 - y)

    def ppf(self, y):
        """Evaluates the inverse CDF along the values ``x``."""
        y_reflect = np.where(y < .5, y, 1 - y)
        z_sq = betaincinv(self.m / 2., .5, 2 * y_reflect)
        x = np.arcsin(np.sqrt(z_sq)) / np.pi
        return np.where(y < .5, x, 1 - x)


def i4_sobol_generate(dim_num, n, skip=1):
    """
    i4_sobol_generate generates a Sobol dataset.
    Parameters:
      Input, integer dim_num, the spatial dimension.
      Input, integer N, the number of points to generate.
      Input, integer SKIP, the number of initial points to skip.
      Output, real R(M,N), the points.
    """
    r = np.full((n, dim_num), np.nan)
    for j in range(n):
        seed = j + 1
        r[j, 0:dim_num], next_seed = i4_sobol(dim_num, seed)

    return r

class Sobol(Distribution):
    """Sobol sequence for quasi Monte Carlo sampling the ``[0, 1]``--cube.
    This is similar to ``np.random.uniform(0, 1, size=(num, d))``, but with
    the additional property that each ``d``--dimensional point is `uniformly
    scattered`.
    This is a wrapper around a library by the authors Corrado Chisari and
    John Burkardt (see `License <license.html>`__). [#]_
    See Also
    --------
    :class:`.Rd`
    :class:`.ScatteredCube`
    :func:`.spherical_transform`
    :class:`.ScatteredHypersphere`
    Notes
    -----
    This is **deterministic** for dimensions up to ``40``, although
    it should in theory work up to ``1111``. For higher dimensions, this
    approach will fall back to ``rng.uniform(size=(n, d))``.
    References
    ----------
    .. [#] http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
    Examples
    --------
    >>> from nengolib.stats import Sobol
    >>> sobol = Sobol().sample(10000, 2)
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(6, 6))
    >>> plt.scatter(*sobol.T, c=np.arange(len(sobol)), cmap='Blues', s=7)
    >>> plt.show()
    """

    def __repr__(self):
        return "%s()" % (type(self).__name__)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            # Tile the points optimally. TODO: refactor
            return np.linspace(1./n, 1, n)[:, None]
        if d is None or not np.issubdtype(type(d), np.integer) or d < 1:
            # TODO: this should be raised when the ensemble is created
            raise ValueError("d (%d) must be positive integer" % d)
        if d > 40:
            warnings.warn("i4_sobol_generate does not support d > 40; "
                          "falling back to Monte Carlo method", UserWarning)
            return rng.uniform(size=(n, d))
        return i4_sobol_generate(d, n, skip=0)


def _rd_generate(n, d, seed=0.5):
    """Generates the first ``n`` points in the ``R_d`` sequence."""

    # http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    def gamma(d, n_iter=20):
        """Newton-Raphson-Method to calculate g = phi_d."""
        x = 1.0
        for _ in range(n_iter):
            x -= (x**(d + 1) - x - 1) / ((d + 1) * x**d - 1)
        return x

    g = gamma(d)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = (1/g) ** (j + 1) % 1

    z = np.zeros((n, d))
    z[0] = (seed + alpha) % 1
    for i in range(1, n):
        z[i] = (z[i-1] + alpha) % 1

    return z


class Rd(Distribution):
    """Rd sequence for quasi Monte Carlo sampling the ``[0, 1]``--cube.
    This is similar to ``np.random.uniform(0, 1, size=(num, d))``, but with
    the additional property that each ``d``--dimensional point is `uniformly
    scattered`.
    This is based on the tutorial and code from [#]_. For `d=2` this is often
    called the Padovan sequence. [#]_
    See Also
    --------
    :class:`.Sobol`
    :class:`.ScatteredCube`
    :func:`.spherical_transform`
    :class:`.ScatteredHypersphere`
    References
    ----------
    .. [#] http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    .. [#] http://oeis.org/A000931
    Examples
    --------
    >>> from nengolib.stats import Rd
    >>> rd = Rd().sample(10000, 2)
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(6, 6))
    >>> plt.scatter(*rd.T, c=np.arange(len(rd)), cmap='Blues', s=7)
    >>> plt.show()
    """  # noqa: E501

    def __repr__(self):
        return "%s()" % (type(self).__name__)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            # Tile the points optimally. TODO: refactor
            return np.linspace(1./n, 1, n)[:, None]
        if d is None or not np.issubdtype(type(d), np.integer) or d < 1:
            # TODO: this should be raised when the ensemble is created
            raise ValueError("d (%d) must be positive integer" % d)
        return _rd_generate(n, d)


class ScatteredHypersphere(UniformHypersphere):
    """Number--theoretic distribution over the hypersphere and hyperball.
    Applies the :func:`.spherical_transform` to the number-theoretic
    sequence :class:`.Rd` to obtain uniformly scattered samples.
    This distribution has the nice mathematical property that the
    `discrepancy` between the `empirical distribution` and :math:`n` samples
    is :math:`\\widetilde{\\mathcal{O}}\\left(\\frac{1}{n}\\right)` as opposed
    to :math:`\\mathcal{O}\\left(\\frac{1}{\\sqrt{n}}\\right)` for the `Monte
    Carlo` method. [#]_ This means that the number of samples are effectively
    squared, making this useful as a means for sampling ``eval_points`` and
    ``encoders`` in Nengo.
    See :doc:`notebooks/research/sampling_high_dimensional_vectors` for
    mathematical details.
    Parameters
    ----------
    surface : ``boolean``
        Set to ``True`` to restrict the points to the surface of the ball
        (i.e., the sphere, with one lower dimension). Set to ``False`` to
        sample from the ball. See also :attr:`.sphere` and :attr:`.ball` for
        pre-instantiated objects with these two options respectively.
    Other Parameters
    ----------------
    base : :class:`nengo.dists.Distribution`, optional
        The base distribution from which to draw `quasi Monte Carlo` samples.
        Defaults to :class:`.Rd` and should not be changed unless
        you have some alternative `number-theoretic sequence` over ``[0, 1]``.
    See Also
    --------
    :attr:`.sphere`
    :attr:`.ball`
    :class:`nengo.dists.UniformHypersphere`
    :class:`.Rd`
    :class:`.Sobol`
    :func:`.spherical_transform`
    :class:`.ScatteredCube`
    Notes
    -----
    The :class:`.Rd` and :class:`.Sobol` distributions are deterministic.
    Nondeterminism comes from a random ``d``--dimensional rotation
    (see :func:`.random_orthogonal`).
    The nengolib logo was created using this class with the Sobol sequence.
    References
    ----------
    .. [#] K.-T. Fang and Y. Wang, Number-Theoretic Methods in Statistics.
       Chapman & Hall, 1994.
    Examples
    --------
    >>> from nengolib.stats import ball, sphere
    >>> b = ball.sample(1000, 2)
    >>> s = sphere.sample(1000, 3)
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> plt.figure(figsize=(6, 3))
    >>> plt.subplot(121)
    >>> plt.title("Ball")
    >>> plt.scatter(*b.T, s=10, alpha=.5)
    >>> ax = plt.subplot(122, projection='3d')
    >>> ax.set_title("Sphere").set_y(1.)
    >>> ax.patch.set_facecolor('white')
    >>> ax.set_xlim3d(-1, 1)
    >>> ax.set_ylim3d(-1, 1)
    >>> ax.set_zlim3d(-1, 1)
    >>> ax.scatter(*s.T, s=10, alpha=.5)
    >>> plt.show()
    """

    def __init__(self, surface, base=Rd()):
        super(ScatteredHypersphere, self).__init__(surface)
        self.base = base

    def __repr__(self):
        return "%s(surface=%r, base=%r)" % (
            type(self).__name__, self.surface, self.base)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            return super(ScatteredHypersphere, self).sample(n, d, rng)

        if self.surface:
            samples = self.base.sample(n, d-1, rng)
            radius = 1.
        else:
            samples = self.base.sample(n, d, rng)
            samples, radius = samples[:, :-1], samples[:, -1:] ** (1. / d)

        mapped = spherical_transform(samples)

        # radius adjustment for ball versus sphere, and a random rotation
        #rotation = random_orthogonal(d, rng=rng)
        return mapped * radius
