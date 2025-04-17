import tensorflow as tf

from .scipyhelpers import scipy_compute_preconditioner, scipy_edmval_cov


def simple_sparse_slice0end(in_sparse, end):
    """
    Slice a tf.sparse.SparseTensor along axis 0 starting 0 to the 'end'.
    """

    # Convert dense_shape, indices, and values to tensors if they aren't already
    dense_shape = in_sparse.dense_shape
    indices = in_sparse.indices
    values = in_sparse.values

    # Compute output dense shape after slicing
    out_shape = tf.concat([[end], dense_shape[1:]], axis=0)

    # Filter rows: select entries where indices[:, 0] < end
    mask = indices[:, 0] < end
    selected_indices = tf.boolean_mask(indices, mask)
    selected_values = tf.boolean_mask(values, mask)

    # Return the sliced sparse tensor
    return tf.sparse.SparseTensor(
        indices=selected_indices, values=selected_values, dense_shape=out_shape
    )


def is_diag(x):
    return tf.math.equal(
        tf.math.count_nonzero(x), tf.math.count_nonzero(tf.linalg.diag_part(x))
    )


def is_on_gpu(tensor):
    """
    Check if tensor is on a GPU device
    """

    device = tensor.device
    device_type = device.split(":")[-2]
    return device_type == "GPU"


def tf_edmval_cov(grad, hess):
    # use a Cholesky decomposition to easily detect the non-positive-definite case
    chol = tf.linalg.cholesky(hess)

    # FIXME catch this exception to mark failed toys and continue
    if tf.reduce_any(tf.math.is_nan(chol)).numpy():
        raise ValueError(
            "Cholesky decomposition failed, Hessian is not positive-definite"
        )

    gradv = grad[..., None]
    edmval = 0.5 * tf.linalg.matmul(
        gradv, tf.linalg.cholesky_solve(chol, gradv), transpose_a=True
    )
    edmval = edmval[0, 0].numpy()

    cov = tf.linalg.cholesky_solve(chol, tf.eye(chol.shape[0], dtype=chol.dtype))

    return edmval, cov


def tf_compute_preconditioner(hess):
    L = tf.linalg.cholesky(hess)
    U = tf.transpose(L)
    del L
    Uinv = tf.linalg.triangular_solve(
        U, tf.eye(hess.shape[0], dtype=hess.dtype), lower=False
    )

    m_precond = Uinv
    m_precond_inv = U

    return m_precond, m_precond_inv


def edmval_cov(grad, hess):
    # scipy is faster than tensorflow on CPU so use it as appropriate
    if is_on_gpu(hess):
        return tf_edmval_cov(grad, hess)
    else:
        return scipy_edmval_cov(grad, hess)


def compute_preconditioner(hess, overwrite_a=False):
    # scipy is faster than tensorflow on CPU so use it as appropriate
    if is_on_gpu(hess):
        return tf_compute_preconditioner(hess)
    else:
        return scipy_compute_preconditioner(hess, overwrite_a=overwrite_a)


def cholesky_adaptive(a):
    if is_diag(a):
        return tf.math.sqrt(a)
    else:
        return tf.linalg.cholesky(a)
