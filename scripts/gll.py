
import numpy as np
from numpy.polynomial.legendre import legder, legval

def gll(n):
    """
    Compute Gauss-Legendre-Lobatto points for n points.
    Includes endpoints -1 and 1.
    """
    if n < 2:
        raise ValueError("Number of points must be at least 2.")

    # Legendre polynomial of degree n-1
    coeffs = np.zeros(n)
    coeffs[-1] = 1  # P_{n-1}(x)

    # Derivative of Legendre polynomial
    dcoeffs = legder(coeffs)

    # Roots of derivative (internal points)
    internal_roots = np.polynomial.legendre.legroots(dcoeffs)

    # Combine endpoints and sort
    points = np.concatenate(([-1.0], internal_roots, [1.0]))
    x = np.sort(points)

    return x
