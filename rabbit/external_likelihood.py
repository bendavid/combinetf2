"""Helpers for external likelihood terms (linear + quadratic parameter priors).

An "external likelihood term" is an additive contribution to the NLL of
the form

    -log L_ext = g^T x_sub + 0.5 * x_sub^T H x_sub

where ``x_sub`` is the subset of the fit parameters the term constrains.
Both the linear (``grad``) and quadratic (``hess_dense`` / ``hess_sparse``)
parts are optional; the sparse Hessian is stored as a
``tf.sparse.SparseTensor`` whose indices are in canonical row-major order.

This module centralizes three things that were previously inlined in
``Fitter.__init__``, ``Fitter._compute_external_nll``, and
``FitInputData.__init__``:

* :func:`read_external_terms_from_h5` — load the raw numpy-level
  per-term dicts from an HDF5 group (used by FitInputData)
* :func:`build_tf_external_terms` — turn that list into tf-side per-term
  dicts (resolved parameter indices, tf.constant grads, CSRSparseMatrix
  Hessians). Used by the Fitter when it takes ownership of the input
  data.
* :func:`compute_external_nll` — evaluate the scalar NLL contribution
  of a list of tf-side terms at the current ``x``.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops as tf_sparse_csr

from rabbit.h5pyutils_read import makesparsetensor, maketensor


def read_external_term_from_h5(ext_group):
    """Decode the HDF5 ``external_term`` group into a raw dict.

    The TensorWriter combines all user-supplied external-likelihood-term
    contributions into a single (params, grad, hess) tuple at write
    time, so the reader sees one flat group rather than per-term
    subgroups. Returns a dict with keys ``params``, ``grad_values``,
    ``hess_dense``, ``hess_sparse`` (at most one of dense/sparse is
    populated), or ``None`` if the group is absent.
    """
    if ext_group is None:
        return None

    raw_params = ext_group["params"][...]
    params = np.array([s.decode() if isinstance(s, bytes) else s for s in raw_params])
    grad_values = (
        np.asarray(maketensor(ext_group["grad_values"]))
        if "grad_values" in ext_group.keys()
        else None
    )
    hess_dense = (
        np.asarray(maketensor(ext_group["hess_dense"]))
        if "hess_dense" in ext_group.keys()
        else None
    )
    hess_sparse = (
        makesparsetensor(ext_group["hess_sparse"])
        if "hess_sparse" in ext_group.keys()
        else None
    )
    return {
        "params": params,
        "grad_values": grad_values,
        "hess_dense": hess_dense,
        "hess_sparse": hess_sparse,
    }


def build_tf_external_term(term, parms, dtype):
    """Turn the raw external-term dict into a tf-side dict ready for the fitter.

    Parameters
    ----------
    term : dict or None
        The raw combined term dict returned by
        :func:`read_external_term_from_h5`.
    parms : array-like of str
        Full ordered list of fit parameter names (POIs + systematics).
    dtype : tf.DType
        Fitter dtype for gradient / Hessian tensors.

    Returns
    -------
    dict or None
        A dict with keys ``indices``, ``grad``, ``hess_dense``,
        ``hess_csr``, or ``None`` if ``term`` is ``None``.
    """
    if term is None:
        return None

    parms_str = np.asarray(parms).astype(str)
    parms_idx = {name: i for i, name in enumerate(parms_str)}
    if len(parms_idx) != len(parms_str):
        raise RuntimeError(
            "Duplicate parameter names in fitter parameter list; "
            "external term resolution requires unique names."
        )

    params = np.asarray(term["params"]).astype(str)
    indices = np.empty(len(params), dtype=np.int64)
    for i, p in enumerate(params):
        j = parms_idx.get(p, -1)
        if j < 0:
            raise RuntimeError(
                f"External likelihood term parameter '{p}' not found "
                "in fit parameters"
            )
        indices[i] = j
    tf_indices = tf.constant(indices, dtype=tf.int64)

    tf_grad = (
        tf.constant(term["grad_values"], dtype=dtype)
        if term["grad_values"] is not None
        else None
    )

    tf_hess_dense = None
    tf_hess_csr = None
    if term["hess_dense"] is not None:
        tf_hess_dense = tf.constant(term["hess_dense"], dtype=dtype)
    elif term["hess_sparse"] is not None:
        # Build a CSRSparseMatrix view of the stored sparse Hessian for
        # use in the closed-form external gradient/HVP path via
        # sm.matmul. The Hessian is assumed symmetric, so the loss
        # L = 0.5 x_sub^T H x_sub has gradient H @ x_sub and HVP
        # H @ p_sub, each a single sm.matmul call. NOTE:
        # SparseMatrixMatMul has no XLA kernel, so any tf.function that
        # calls sm.matmul must be built with jit_compile=False. The
        # TensorWriter sorts the indices into canonical row-major order
        # at write time, so we can feed the SparseTensor straight to
        # the CSR builder without an additional reorder step.
        tf_hess_csr = tf_sparse_csr.CSRSparseMatrix(term["hess_sparse"])

    return {
        "indices": tf_indices,
        "grad": tf_grad,
        "hess_dense": tf_hess_dense,
        "hess_csr": tf_hess_csr,
    }


def compute_external_nll(term, x, dtype):
    """Evaluate the scalar NLL contribution of the external term.

    Adds ``g^T x_sub + 0.5 * x_sub^T H x_sub`` where ``x_sub`` is the
    slice of the full parameter vector selected by ``term["indices"]``.
    Sparse Hessians use ``sm.matmul`` for the ``H @ x_sub`` product,
    which dispatches to a multi-threaded CSR kernel.

    Parameters
    ----------
    term : dict or None
        tf-side external term dict as returned by
        :func:`build_tf_external_term`.
    x : tf.Tensor
        Current full parameter vector.
    dtype : tf.DType
        Dtype for the accumulator.

    Returns
    -------
    tf.Tensor or None
        Scalar contribution to the NLL, or ``None`` if ``term`` is
        ``None``.
    """
    if term is None:
        return None
    x_sub = tf.gather(x, term["indices"])
    total = tf.zeros([], dtype=dtype)
    if term["grad"] is not None:
        total = total + tf.reduce_sum(term["grad"] * x_sub)
    if term["hess_dense"] is not None:
        # 0.5 * x_sub^T H x_sub
        total = total + 0.5 * tf.reduce_sum(
            x_sub * tf.linalg.matvec(term["hess_dense"], x_sub)
        )
    elif term["hess_csr"] is not None:
        Hx = tf.squeeze(
            tf_sparse_csr.matmul(term["hess_csr"], x_sub[:, None]),
            axis=-1,
        )
        total = total + 0.5 * tf.reduce_sum(x_sub * Hx)
    return total
