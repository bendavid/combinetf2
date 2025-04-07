import numpy as np
import scipy

# all possible methods for scipy.optimize.minimize
minimize_methods = [
    "nelder-mead",
    "powell",
    "cg",
    "bfgs",
    "newton-cg",
    "l-bfgs-b",
    "tnc",
    "cobyla",
    "cobyqa",
    "slsqp",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
]

# only those that support hessian-vector products
minimize_methods_hessp = ["newton-cg", "trust-ncg", "trust-krylov", "trust-constr"]

# only those that support hessians
minimize_methods_hess = [
    "newton-cg",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
    "trust-constr",
]


# as scipy.linalg.cho_factor but with the other half of the matrix zeroed for downstream use
def cho_factor_clean(a, lower=False, overwrite_a=False, check_finite=True):
    c = scipy.linalg.cholesky(
        a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite
    )
    return c, lower


# inverse of a triangular matrix
def tri_inv(a, lower=False, overwrite_a=False, check_finite=True):
    a1 = np.asarray_chkfinite(a) if check_finite else np.asarray(a)

    # trtri is the lower level lapack function which computes the inverse of a triangular matrix
    (trtri,) = scipy.linalg.lapack.get_lapack_funcs(("trtri",), (a1,))
    ainv, _ = trtri(a1, lower=lower, overwrite_c=overwrite_a)

    return ainv


# inverse of a matrix given its Cholesky factorization (output of cho_factor_clean)
# WARNING the provided Cholesky decomposition must have zeros in the other half of the matrix
# which is NOT the case for scipy.linalg.cho_factor
# this avoid the need for the intermediate identity matrix when using scipy.linalg.cho_solve
def cho_inv(chol, overwrite_a=False, check_finite=True):
    c, lower = chol
    cinv = tri_inv(c, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)

    if lower:
        minv = cinv.T @ cinv
    else:
        minv = cinv @ cinv.T

    return minv


def scipy_edmval_cov(grad, hess):
    hess = hess.__array__()
    # FIXME catch this exception to mark failed toys and continue
    try:
        chol = cho_factor_clean(hess.__array__(), lower=False)
    except scipy.linalg.LinAlgError:
        raise ValueError(
            "Cholesky decomposition failed, Hessian is not positive-definite"
        )

    grad = grad.__array__()
    gradv = grad[..., None]
    edmval = 0.5 * gradv.T @ scipy.linalg.cho_solve(chol, gradv)
    edmval = edmval[0, 0]

    cov = cho_inv(chol, overwrite_a=True)

    return edmval, cov


def scipy_compute_preconditioner(hess, overwrite_a=False):
    lower = False
    U = scipy.linalg.cholesky(hess, overwrite_a=overwrite_a, lower=lower)
    Uinv = tri_inv(U, lower=lower)

    m_precond = Uinv
    m_precond_inv = U

    return m_precond, m_precond_inv
