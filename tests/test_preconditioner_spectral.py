"""The spectral transform, and the ridge failure it exists to fix.

These pin the mechanism with concrete numbers so a future change back to a
scalar ridge cannot pass silently.
"""

import numpy as np
import pytest
import scipy.linalg

from rabbit import preconditioner as precond


def _cond_true(m):
    sv = np.linalg.svd(np.asarray(m, dtype=np.float64), compute_uv=False)
    return float(sv[0] / sv[-1])


def _whiten(chol, block):
    """L^-1 B L^-T, the true block in the new coordinates."""
    t = scipy.linalg.solve_triangular(chol, block, lower=True, trans="N")
    return scipy.linalg.solve_triangular(chol, t.T, lower=True, trans="N").T


# The case that motivated this: curvatures spanning nine orders of magnitude,
# one of them negative. Physically alphaS + NP lambda (huge, and one negative
# direction) sharing a block with a unit-normalised nuisance.
HARD = np.array(
    [
        [3.3e9, -1.2e8, 30.0],
        [-1.2e8, -8.7e6, 5.0],
        [30.0, 5.0, 1.0],
    ]
)


def _factorise(block, ridge=1e-8, transform="spectral"):
    return precond.Preconditioner._factorise(
        block, np.arange(block.shape[0]), ridge, 4, "", transform=transform
    )


def test_spectral_whitens_a_block_a_ridge_cannot():
    blk = _factorise(HARD)
    assert blk is not None
    t = _whiten(blk.chol, HARD)
    # spectral reaches the identity up to sign
    assert _cond_true(t) == pytest.approx(1.0, rel=1e-6)
    assert np.allclose(np.abs(np.diag(t)), 1.0, atol=1e-5)

    # and the sign of the negative direction SURVIVES -- trust-krylov needs it
    assert np.count_nonzero(np.diag(t) < 0) == 1

    # what a scalar ridge would have done instead, for the record
    lam = np.linalg.eigvalsh(HARD)
    ridge = abs(lam[0]) * 1.1
    lr = scipy.linalg.cholesky(HARD + ridge * np.eye(3), lower=True)
    tr = _whiten(lr, HARD)
    assert _cond_true(tr) > 1e7  # measured 1.44e+08
    # the soft direction collapses to near-null: that is the whole failure
    assert np.min(np.abs(np.diag(tr))) < 1e-6  # measured 7e-08


def test_identical_to_cholesky_when_positive_definite():
    """Nothing is lost on the easy blocks: both give exactly the identity."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((40, 40))
    pd = a @ a.T + 40.0 * np.eye(40)

    spectral = _whiten(_factorise(pd).chol, pd)
    plain = _whiten(scipy.linalg.cholesky(pd, lower=True), pd)
    assert _cond_true(spectral) == pytest.approx(1.0, rel=1e-8)
    assert _cond_true(plain) == pytest.approx(1.0, rel=1e-8)
    assert np.allclose(spectral, plain, atol=1e-8)


def test_all_negative_block_is_usable():
    """A block with NO positive diagonal used to be skipped outright.

    Nothing can be whitened by a ridge scaled to max(diag) when that maximum is
    negative, so 21 of 34 blocks were dropped in a real fit -- exactly the ones
    needing help. |Lambda| has no such problem.
    """
    neg = np.diag([-7.5e3, -6.3e4, -2.1e3]).astype(float)
    blk = _factorise(neg)
    assert blk is not None
    t = _whiten(blk.chol, neg)
    assert _cond_true(t) == pytest.approx(1.0, rel=1e-6)
    assert np.all(np.diag(t) < 0)  # still a maximum in every direction


def test_near_null_direction_is_floored_not_amplified():
    """A genuinely flat direction has no scale to whiten to."""
    m = np.diag([1.0, 1e-18]).astype(float)
    blk = _factorise(m, ridge=1e-8)
    assert blk is not None
    # the floor keeps L^-1 bounded; without it 1/sqrt(1e-18) = 1e9
    assert np.max(np.abs(blk.chol)) < 1e3


def test_default_transform_is_ridge_so_existing_behaviour_is_unchanged():
    """The spectral transform is opt-in. Shipping it as the default would change
    the numerics of every --precondition user on a feature that is not ours."""
    import inspect

    sig = inspect.signature(precond.Preconditioner._factorise)
    assert sig.parameters["transform"].default == "ridge"

    # and the ridge path still reproduces the failure mode it is known for,
    # which is the reason spectral exists -- if this ever passes, the ridge
    # implementation changed underneath us
    blk = _factorise(HARD, transform="ridge")
    assert blk is not None
    t = _whiten(blk.chol, HARD)
    assert _cond_true(t) > 1e7
