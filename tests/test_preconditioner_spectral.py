"""The spectral transform, and the ridge failure it exists to fix.

These pin the mechanism with concrete numbers so a future change back to a
scalar ridge cannot pass silently.
"""

import numpy as np
import pytest
import scipy.linalg

from rabbit import preconditioner as precond

# the module's own, so these tests measure exactly what the log reports
_cond_true = precond._cond_true


# L^-1 B L^-T, the true block in the new coordinates. The module's own, so
# these tests measure what the log line reports.
_whiten = precond._whiten


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
    # the reported pair is the TRUE condition number at both ends, and the
    # scale-free degeneracy is carried separately -- on this block the latter
    # is 1.2, which is exactly why it must not be the "after" number
    assert blk.cond_before == pytest.approx(_cond_true(HARD), rel=1e-6)
    assert blk.cond_after == pytest.approx(1.0, rel=1e-6)
    assert blk.corr_before == pytest.approx(precond._cond_corr(HARD), rel=1e-6)
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


@pytest.mark.parametrize("n", [5, 12, 30, 60])
def test_spectral_leaves_true_condition_number_exactly_one(n):
    """The spec of the spectral transform, at any block size.

    B and |B| share eigenvectors, so with |B| = L L^T the congruence
    L^-1 B L^-T sends Lambda to its own signature: `tb` is symmetric AND
    orthogonal, hence tb^2 = I and true kappa == 1 identically -- not
    approximately, and not only on a hand-built 3x3.

    This also pins why the *correlation* number is not reported for the
    whitened block: L is a Cholesky factor rather than the eigenbasis, so
    diag(tb) picks up the arbitrary orientation between the two and the
    1/sqrt|diag| normalisation turns it into an artefact that GROWS with block
    size, while the truth stays 1.
    """
    rng = np.random.default_rng(n)
    a = rng.standard_normal((n, n))
    block = a + a.T  # symmetric and indefinite, the case that matters

    # nothing floored, or the congruence identity below does not hold
    w = np.linalg.eigvalsh(block)
    floor = np.finfo(np.float64).eps * n * np.max(np.abs(w))
    assert not np.any(np.abs(w) < floor)

    blk = _factorise(block)
    tb = _whiten(blk.chol, block)

    assert np.allclose(tb @ tb, np.eye(n), atol=1e-8 * n)
    assert _cond_true(tb) == pytest.approx(1.0, rel=1e-6)
    assert blk.cond_after == pytest.approx(1.0, rel=1e-6)

    # and the correlation number would have claimed otherwise, increasingly so
    if n >= 12:
        assert precond._cond_corr(tb) > 5.0


def test_flooring_is_why_true_kappa_is_measured_not_assumed():
    """A floored direction is one the congruence no longer preserves."""
    m = np.diag([1.0, -2.0, 1e-20])
    blk = _factorise(m)
    tb = _whiten(blk.chol, m)
    assert not np.allclose(tb @ tb, np.eye(3), atol=1e-6)
    assert blk.cond_after > 1e3  # legitimately far from 1


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
    """A genuinely flat direction has no scale to whiten to.

    The quantity to pin is L^-1, not L. L holds sqrt(|lam|), so a near-null
    direction makes its entries SMALL -- asserting on them passes whether or
    not the flooring happens.
    """
    m = np.diag([1.0, 1e-18]).astype(float)
    blk = _factorise(m, ridge=1e-8)
    assert blk is not None

    # what the floor bounds: 1/sqrt(eps*m*wmax) rather than 1/sqrt(1e-18) = 1e9
    floor = np.finfo(np.float64).eps * 2 * 1.0
    assert np.max(np.abs(np.linalg.inv(blk.chol))) == pytest.approx(
        1.0 / np.sqrt(floor), rel=0.1
    )

    # so the protection is a factor of ~21 here (1e9 -> 4.7e7), not unbounded
    # -> bounded. Enough that the trust region can reject the step; the comment
    # in _factorise_spectral says so, and this is the number it refers to.
    unfloored = scipy.linalg.cholesky(np.diag([1.0, 1e-18]), lower=True)
    gain = np.max(np.abs(np.linalg.inv(unfloored))) / np.max(
        np.abs(np.linalg.inv(blk.chol))
    )
    assert 10.0 < gain < 100.0

    # and the resolvable direction is untouched
    assert blk.chol[0, 0] == pytest.approx(1.0)


def test_ridge_can_start_on_an_all_negative_block():
    """The DEFAULT path must not drop a block for having no positive diagonal.

    The ridge is expressed in units of max|diag|; scaling it by max(diag)
    instead skipped every all-negative block outright (21 of 34 on one real
    fit). The first Cholesky still fails, then the spectrum branch sizes the
    ridge from |lam_min| and it succeeds. Not as good as spectral -- 320
    against 1 on this block -- but a usable transform beats none.
    """
    neg = np.diag([-7.5e3, -6.3e4, -2.1e3]).astype(float)
    blk = _factorise(neg, transform="ridge")
    assert blk is not None
    t = _whiten(blk.chol, neg)
    assert _cond_true(t) < 1e3
    assert np.all(np.diag(t) < 0)  # signs survive here too

    spectral = _whiten(_factorise(neg, transform="spectral").chol, neg)
    assert _cond_true(spectral) < _cond_true(t)


def test_default_transform_is_ridge_so_existing_behaviour_is_unchanged():
    """The spectral transform is opt-in. Shipping it as the default would change
    the numerics of every --precondition user on a feature that is not ours."""
    import inspect

    # all three places the default lives, so flipping one cannot pass silently
    sig = inspect.signature(precond.Preconditioner._factorise)
    assert sig.parameters["transform"].default == "ridge"
    assert (
        inspect.signature(precond.Preconditioner.from_hessian)
        .parameters["transform"]
        .default
        == "ridge"
    )
    # the --preconditionTransform and Fitter defaults, the other two places it
    # lives, are pinned in test_preconditioner.py (they need the fitter stack)

    # and the ridge path still reproduces the failure mode it is known for,
    # which is the reason spectral exists -- if this ever passes, the ridge
    # implementation changed underneath us
    blk = _factorise(HARD, transform="ridge")
    assert blk is not None
    t = _whiten(blk.chol, HARD)
    assert _cond_true(t) > 1e7
