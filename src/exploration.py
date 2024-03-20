import numpy as np

def test_func(x):
    """This function will try to calculate:

    .. math::
        \sum_{i=1}^{\infty} x_{i}

    good luck!
    """
    pass

E0 = 1

def rect(t: np.ndarray):
    return (t > 0) & (t < 1)

a = np.linspace(0, 2, 100)

print(rect(a))