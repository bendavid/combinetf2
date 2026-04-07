import math
import os
from collections import defaultdict

import h5py
import numpy as np
from wums.sparse_hist import SparseHist  # noqa: F401  re-exported for convenience

from rabbit import common, h5pyutils_write

from wums import ioutils, logging  # isort: skip

logger = logging.child_logger(__name__)


class TensorWriter:
    def __init__(
        self,
        sparse=False,
        systematic_type="log_normal",
        allow_negative_expectation=False,
        add_bin_by_bin_stat_to_data_cov=False,
    ):
        self.allow_negative_expectation = allow_negative_expectation

        self.systematic_type = systematic_type

        self.symmetric_tensor = True  # If all shape systematics are symmetrized the systematic tensor is symmetric leading to reduced memory and improved efficiency
        self.add_bin_by_bin_stat_to_data_cov = add_bin_by_bin_stat_to_data_cov  # add bin by bin statistical uncertainty to data covariance matrix

        self.signals = set()
        self.bkgs = set()

        self.channels = {}
        self.nbinschan = {}
        self.pseudodata_names = set()

        self.dict_systgroups = defaultdict(lambda: set())

        self.systsstandard = set()
        self.systsnoi = set()
        self.systsnoconstraint = set()
        self.systscovariance = set()

        self.sparse = sparse
        self.idxdtype = "int64"

        # temporary data
        self.dict_data_obs = {}  # [channel]
        self.dict_data_var = {}  # [channel]
        self.data_covariance = None
        self.dict_pseudodata = {}  # [channel][pseudodata]
        self.dict_norm = {}  # [channel][process]
        self.dict_sumw2 = {}  # [channel][process]
        self.dict_logkavg = {}  # [channel][proc][syst]
        self.dict_logkhalfdiff = {}  # [channel][proc][syst]
        self.dict_logkavg_indices = {}
        self.dict_logkhalfdiff_indices = {}
        self.dict_beta_variations = {}  # [channel][syst][process]

        self.has_beta_variations = False

        # External likelihood terms. Each term is a dict with keys:
        #   name: identifier
        #   params: 1D ndarray of parameter name strings; both grad and hess
        #     refer to this same parameter list in the same order
        #   grad_values: 1D float ndarray (length == len(params)) or None
        #   hess_dense: 2D float ndarray of shape (len(params), len(params)) or None
        #   hess_sparse: tuple (rows, cols, values) for sparse hessian or None
        # Exactly one of hess_dense / hess_sparse may be set, or neither
        # (gradient-only term). Parameter names are resolved against the full
        # fit parameter list (POIs + systs) at fit time. See
        # add_external_likelihood_term for details.
        self.external_terms = []

        self.clipSystVariations = False
        if self.clipSystVariations > 0.0:
            self.clip = np.abs(np.log(self.clipSystVariations))

        self.logkepsilon = math.log(
            1e-3
        )  # numerical cutoff in case of zeros in systematic variations

        # settings for writing out hdf5 files
        self.dtype = "float64"
        self.chunkSize = 4 * 1024**2

    @staticmethod
    def _issparse(h):
        """Check if h is a scipy sparse array/matrix."""
        return hasattr(h, "toarray") and hasattr(h, "tocoo")

    @staticmethod
    def _sparse_to_flat_csr(h, dtype, flow=False):
        """Flatten a scipy sparse array/matrix to CSR with shape (1, prod(shape)).

        For SparseHist inputs, forwards ``flow`` to ``h.to_flat_csr`` so the
        wrapper can convert from its internal with-flow layout to the requested
        layout. For raw scipy sparse inputs, the row-major flatten of ``h.shape``
        is used directly (the user is responsible for matching the channel layout).

        The returned CSR array has sorted indices suitable for searchsorted lookups.
        """
        if hasattr(h, "to_flat_csr"):
            return h.to_flat_csr(dtype, flow=flow)

        import scipy.sparse

        size = int(np.prod(h.shape))
        coo = scipy.sparse.coo_array(h)
        if coo.ndim == 2:
            flat_indices = np.ravel_multi_index((coo.row, coo.col), h.shape)
        elif coo.ndim == 1:
            flat_indices = coo.coords[0]
        else:
            raise ValueError(
                f"Unsupported dimensionality {coo.ndim} for scipy sparse input"
            )
        sort_order = np.argsort(flat_indices)
        sorted_indices = flat_indices[sort_order].astype(np.int32)
        sorted_data = coo.data[sort_order].astype(dtype)
        indptr = np.array([0, len(sorted_data)], dtype=np.int32)
        return scipy.sparse.csr_array(
            (sorted_data, sorted_indices, indptr), shape=(1, size)
        )

    def _to_flat_dense(self, h, flow=False):
        """Convert any array-like (including scipy sparse) to a flat dense numpy array.

        For SparseHist inputs, ``flow`` selects the with-flow or no-flow layout.
        """
        if isinstance(h, SparseHist):
            return np.asarray(h.toarray(flow=flow)).flatten().astype(self.dtype)
        if self._issparse(h):
            return np.asarray(h.toarray()).flatten().astype(self.dtype)
        return np.asarray(h).flatten().astype(self.dtype)

    def get_flat_values(self, h, flow=False):
        if hasattr(h, "values"):
            values = h.values(flow=flow)
        elif isinstance(h, SparseHist):
            values = h.toarray(flow=flow)
        elif self._issparse(h):
            values = np.asarray(h.toarray())
        else:
            values = h
        return np.asarray(values).flatten().astype(self.dtype)

    def get_flat_variances(self, h, flow=False):
        if hasattr(h, "variances"):
            variances = h.variances(flow=flow)
        elif isinstance(h, SparseHist):
            variances = h.toarray(flow=flow)
        elif self._issparse(h):
            variances = np.asarray(h.toarray())
        else:
            variances = h

        variances = np.asarray(variances).flatten().astype(self.dtype)
        if (variances < 0.0).any():
            raise ValueError("Negative variances encountered")

        return variances

    def add_data(self, h, channel="ch0", variances=None):
        self._check_hist_and_channel(h, channel)
        if channel in self.dict_data_obs.keys():
            raise RuntimeError(f"Data histogram for channel '{channel}' already set.")
        self.dict_data_obs[channel] = self.get_flat_values(h)
        self.dict_data_var[channel] = self.get_flat_variances(
            h if variances is None else variances
        )

    def add_data_covariance(self, cov):
        self.data_covariance = cov if isinstance(cov, np.ndarray) else cov.values()

    def add_pseudodata(self, h, name=None, channel="ch0"):
        self._check_hist_and_channel(h, channel)
        if name is None:
            name = f"pseudodata_{len(self.pseudodata_names)}"
        self.pseudodata_names.add(name)
        if channel not in self.dict_pseudodata.keys():
            self.dict_pseudodata[channel] = {}
        if name in self.dict_pseudodata[channel].keys():
            raise RuntimeError(
                f"Pseudodata histogram '{name}' for channel '{channel}' already set."
            )
        self.dict_pseudodata[channel][name] = self.get_flat_values(h)

    def add_process(self, h, name, channel="ch0", signal=False, variances=None):
        self._check_hist_and_channel(h, channel)

        if name in self.dict_norm[channel].keys():
            raise RuntimeError(
                f"Nominal histogram for process '{name}' for channel '{channel}' already set."
            )

        if signal:
            self.signals.add(name)
        else:
            self.bkgs.add(name)

        self.dict_logkavg[channel][name] = {}
        self.dict_logkhalfdiff[channel][name] = {}
        if self.sparse:
            self.dict_logkavg_indices[channel][name] = {}
            self.dict_logkhalfdiff_indices[channel][name] = {}

        flow = self.channels[channel]["flow"]

        if self.sparse and self._issparse(h):
            # Store as flat CSR, avoiding full dense conversion
            norm = self._sparse_to_flat_csr(h, self.dtype, flow=flow)
            if not np.all(np.isfinite(norm.data)):
                raise RuntimeError(
                    f"NaN or Inf values encountered in nominal histogram for {name}!"
                )
            if not self.allow_negative_expectation:
                has_negative = np.any(norm.data < 0.0)
                if has_negative:
                    norm = norm.copy()
                    norm.data[:] = np.maximum(norm.data, 0.0)
                    norm.eliminate_zeros()
        else:
            norm = self.get_flat_values(h, flow)
            if not self.allow_negative_expectation:
                norm = np.maximum(norm, 0.0)
            if not np.all(np.isfinite(norm)):
                raise RuntimeError(
                    f"{len(norm)-sum(np.isfinite(norm))} NaN or Inf values encountered in nominal histogram for {name}!"
                )

        # variances are always stored dense (needed for sumw2 output assembly)
        if variances is not None:
            sumw2 = self.get_flat_variances(variances, flow)
        elif self._issparse(h):
            sumw2 = self._to_flat_dense(h, flow=flow)
        else:
            sumw2 = self.get_flat_variances(h, flow)

        if not np.all(np.isfinite(sumw2)):
            raise RuntimeError(
                f"{len(sumw2)-sum(np.isfinite(sumw2))} NaN or Inf values encountered in variances for {name}!"
            )

        self.dict_norm[channel][name] = norm
        self.dict_sumw2[channel][name] = sumw2

    def add_channel(self, axes, name=None, masked=False, flow=False):
        if flow and masked is False:
            raise NotImplementedError(
                "Keeping underflow/overflow is currently only supported for masked channels"
            )
        if name is None:
            name = f"ch{len(self.channels)}"
        logger.debug(f"Add new channel {name}")
        ibins = np.prod([a.extent if flow else a.size for a in axes])
        self.nbinschan[name] = ibins
        self.dict_norm[name] = {}
        self.dict_sumw2[name] = {}
        self.dict_beta_variations[name] = {}

        # add masked channels last and not masked channels first
        this_channel = {"axes": [a for a in axes], "masked": masked, "flow": flow}
        if masked:
            self.channels[name] = this_channel
        else:
            self.channels = {name: this_channel, **self.channels}

        self.dict_logkavg[name] = {}
        self.dict_logkhalfdiff[name] = {}
        if self.sparse:
            self.dict_logkavg_indices[name] = {}
            self.dict_logkhalfdiff_indices[name] = {}

    def _check_hist_and_channel(self, h, channel):

        if channel not in self.channels.keys():
            raise RuntimeError(f"Channel {channel} not known!")

        if hasattr(h, "axes"):
            axes = [a for a in h.axes]
            channel_axes = self.channels[channel]["axes"]

            if not all(np.allclose(a, axes[i]) for i, a in enumerate(channel_axes)):
                raise RuntimeError(f"""
                    Histogram axes different have different edges from channel axes of channel {channel}
                    \nHistogram axes: {[a.edges for a in axes]}
                    \nChannel axes: {[a.edges for a in channel_axes]}
                    """)
        elif self._issparse(h):
            size_in = int(np.prod(h.shape))
            size_this = int(np.prod([len(a) for a in self.channels[channel]["axes"]]))
            if size_in != size_this:
                raise RuntimeError(
                    f"Total number of elements in sparse input different from channel size '{size_in}' != '{size_this}'"
                )
        else:
            shape_in = h.shape
            shape_this = tuple([len(a) for a in self.channels[channel]["axes"]])
            if shape_in != shape_this:
                raise RuntimeError(
                    f"Shape of input object different from channel axes '{shape_in}' != '{shape_this}'"
                )

    def _compute_asym_syst(
        self,
        logkup,
        logkdown,
        name,
        process,
        channel,
        symmetrize="average",
        add_to_data_covariance=False,
        _sparse_info=None,
        **kargs,
    ):
        """Compute symmetrized logk from asymmetric up/down variations.

        When _sparse_info is set to (nnz_indices, size), logkup/logkdown are value
        arrays at those indices and internal book_logk calls use sparse tuples.
        """
        var_name_out = name

        def _wrap(vals):
            """Wrap values as sparse tuple if in sparse mode."""
            if _sparse_info is not None:
                return (_sparse_info[0], vals, _sparse_info[1])
            return vals

        if symmetrize == "conservative":
            # symmetrize by largest magnitude of up and down variations
            logkavg_proc = np.where(
                np.abs(logkup) > np.abs(logkdown),
                logkup,
                logkdown,
            )
        elif symmetrize == "average":
            # symmetrize by average of up and down variations
            logkavg_proc = 0.5 * (logkup + logkdown)
        elif symmetrize in ["linear", "quadratic"]:
            # "linear" corresponds to a piecewise linear dependence of logk on theta
            # while "quadratic" corresponds to a quadratic dependence and leads
            # to a large variance
            diff_fact = np.sqrt(3.0) if symmetrize == "quadratic" else 1.0

            # split asymmetric variation into two symmetric variations
            logkavg_proc = 0.5 * (logkup + logkdown)
            logkdiffavg_proc = 0.5 * diff_fact * (logkup - logkdown)

            var_name_out = name + "SymAvg"
            var_name_out_diff = name + "SymDiff"

            # special case, book the extra systematic
            self.book_logk_avg(
                _wrap(logkdiffavg_proc), channel, process, var_name_out_diff
            )
            self.book_systematic(
                var_name_out_diff,
                add_to_data_covariance=add_to_data_covariance,
                **kargs,
            )
        else:
            if add_to_data_covariance:
                raise RuntimeError(
                    "add_to_data_covariance requires symmetric uncertainties"
                )

            self.symmetric_tensor = False

            logkavg_proc = 0.5 * (logkup + logkdown)
            logkhalfdiff_proc = 0.5 * (logkup - logkdown)

            self.book_logk_halfdiff(_wrap(logkhalfdiff_proc), channel, process, name)
        logkup = None
        logkdown = None

        return logkavg_proc, var_name_out

    def add_norm_systematic(
        self,
        name,
        process,
        channel,
        uncertainty,
        add_to_data_covariance=False,
        groups=None,
        symmetrize="average",
        **kargs,
    ):
        if not isinstance(process, (list, tuple, np.ndarray)):
            process = [process]

        if not isinstance(uncertainty, (list, tuple, np.ndarray)):
            uncertainty = [uncertainty]

        if len(uncertainty) != 1 and len(process) != len(uncertainty):
            raise RuntimeError(
                f"uncertainty must be either a scalar or list with the same length as the list of processes but len(process) = {len(process)} and len(uncertainty) = {len(uncertainty)}"
            )

        var_name_out = name

        systematic_type = "normal" if add_to_data_covariance else self.systematic_type

        for p, u in zip(process, uncertainty):
            norm = self.dict_norm[channel][p]

            if self._issparse(norm):
                # Sparse norm path: compute logk at nonzero positions only
                norm_vals = norm.data
                nnz_idx = norm.indices
                size = norm.shape[1]

                if isinstance(u, (list, tuple, np.ndarray)):
                    if len(u) != 2:
                        raise RuntimeError(
                            f"lnN uncertainty can only be a scalar for a symmetric or a list of 2 elements for asymmetric lnN uncertainties, but got a list of {len(u)} elements"
                        )
                    logkup_proc = self._get_logk_sparse(
                        norm_vals * u[0], norm_vals, 1.0, systematic_type
                    )
                    logkdown_proc = -self._get_logk_sparse(
                        norm_vals * u[1], norm_vals, 1.0, systematic_type
                    )
                    logkavg_proc, var_name_out = self._compute_asym_syst(
                        logkup_proc,
                        logkdown_proc,
                        name,
                        process,
                        channel,
                        symmetrize=symmetrize,
                        add_to_data_covariance=add_to_data_covariance,
                        _sparse_info=(nnz_idx, size),
                        **kargs,
                    )
                else:
                    logkavg_proc = self._get_logk_sparse(
                        norm_vals * u, norm_vals, 1.0, systematic_type
                    )

                self.book_logk_avg(
                    (nnz_idx, logkavg_proc, size), channel, p, var_name_out
                )
            else:
                if isinstance(u, (list, tuple, np.ndarray)):
                    if len(u) != 2:
                        raise RuntimeError(
                            f"lnN uncertainty can only be a scalar for a symmetric or a list of 2 elements for asymmetric lnN uncertainties, but got a list of {len(u)} elements"
                        )
                    # asymmetric lnN uncertainty
                    syst_up = norm * u[0]
                    syst_down = norm * u[1]

                    logkup_proc = self.get_logk(
                        syst_up, norm, systematic_type=systematic_type
                    )
                    logkdown_proc = -self.get_logk(
                        syst_down, norm, systematic_type=systematic_type
                    )

                    logkavg_proc, var_name_out = self._compute_asym_syst(
                        logkup_proc,
                        logkdown_proc,
                        name,
                        process,
                        channel,
                        symmetrize=symmetrize,
                        add_to_data_covariance=add_to_data_covariance,
                        **kargs,
                    )
                else:
                    syst = norm * u
                    logkavg_proc = self.get_logk(
                        syst, norm, systematic_type=systematic_type
                    )

                self.book_logk_avg(logkavg_proc, channel, p, var_name_out)

        self.book_systematic(
            var_name_out,
            groups=groups,
            add_to_data_covariance=add_to_data_covariance,
            **kargs,
        )

    def _add_systematic_sparse(
        self,
        h,
        name,
        process,
        channel,
        norm,
        kfactor,
        mirror,
        symmetrize,
        add_to_data_covariance,
        as_difference,
        **kargs,
    ):
        """Sparse path for add_systematic when norm is stored as scipy sparse CSR.

        Computes logk only at norm's nonzero positions, avoiding full-size dense
        intermediate arrays.  The logk result is a tuple (indices, values, size).
        """
        systematic_type = "normal" if add_to_data_covariance else self.systematic_type
        flow = self.channels[channel]["flow"]
        nnz_idx = norm.indices
        norm_vals = norm.data
        size = norm.shape[1]

        var_name_out = name

        if isinstance(h, (list, tuple)):
            self._check_hist_and_channel(h[0], channel)
            self._check_hist_and_channel(h[1], channel)

            syst_up_vals = self._get_syst_at_norm_nnz(h[0], norm, flow)
            syst_down_vals = self._get_syst_at_norm_nnz(h[1], norm, flow)

            if as_difference:
                syst_up_vals = norm_vals + syst_up_vals
                syst_down_vals = norm_vals + syst_down_vals

            logkup_vals = self._get_logk_sparse(
                syst_up_vals, norm_vals, kfactor, systematic_type
            )
            logkdown_vals = -self._get_logk_sparse(
                syst_down_vals, norm_vals, kfactor, systematic_type
            )

            logkavg_vals, var_name_out = self._compute_asym_syst(
                logkup_vals,
                logkdown_vals,
                name,
                process,
                channel,
                symmetrize,
                add_to_data_covariance,
                _sparse_info=(nnz_idx, size),
                **kargs,
            )
        elif mirror:
            self._check_hist_and_channel(h, channel)

            syst_vals = self._get_syst_at_norm_nnz(h, norm, flow)

            if as_difference:
                syst_vals = norm_vals + syst_vals

            logkavg_vals = self._get_logk_sparse(
                syst_vals, norm_vals, kfactor, systematic_type
            )
        else:
            raise RuntimeError(
                "Only one histogram given but mirror=False, can not construct a variation"
            )

        logkavg_proc = (nnz_idx, logkavg_vals, size)
        self.book_logk_avg(logkavg_proc, channel, process, var_name_out)
        self.book_systematic(
            var_name_out, add_to_data_covariance=add_to_data_covariance, **kargs
        )

    @staticmethod
    def _bin_label(ax, idx):
        """Return a string label for a hist axis bin, preferring string values."""
        try:
            v = ax.value(idx)
            if isinstance(v, (str, bytes)):
                return v.decode() if isinstance(v, bytes) else v
        except Exception:
            pass
        return str(idx)

    def _get_systematic_slices(self, h, name, channel, syst_axes=None):
        """Detect extra axes in h beyond the channel and return list of (sub_name, sub_h) slices.

        Returns None if there are no extra axes (i.e. single-systematic case).

        h may be a single histogram or a list/tuple of two (up/down) histograms.
        Both elements of a pair must share the same extra-axis structure.

        syst_axes:
          - None (default): auto-detect any axes in h not present in the channel
          - list of axis names: use exactly these axes as systematic axes
          - empty list: disable detection entirely
        """
        if syst_axes is not None and len(syst_axes) == 0:
            return None

        if isinstance(h, (list, tuple)):
            h_ref = h[0]
            is_pair = True
        else:
            h_ref = h
            is_pair = False

        # only hist-like objects (with .axes) support multi-systematic
        if not hasattr(h_ref, "axes"):
            return None

        h_axis_names = [a.name for a in h_ref.axes]
        channel_axis_names = [a.name for a in self.channels[channel]["axes"]]

        if syst_axes is None:
            extra_axis_names = [n for n in h_axis_names if n not in channel_axis_names]
        else:
            for n in syst_axes:
                if n not in h_axis_names:
                    raise RuntimeError(
                        f"Requested systematic axis '{n}' not found in histogram axes {h_axis_names}"
                    )
                if n in channel_axis_names:
                    raise RuntimeError(
                        f"Systematic axis '{n}' overlaps with channel axes {channel_axis_names}"
                    )
            extra_axis_names = list(syst_axes)

        if not extra_axis_names:
            return None

        extra_axes = [h_ref.axes[n] for n in extra_axis_names]
        extra_sizes = [len(a) for a in extra_axes]

        import itertools

        slices = []
        for idx_tuple in itertools.product(*[range(s) for s in extra_sizes]):
            labels = [self._bin_label(ax, i) for ax, i in zip(extra_axes, idx_tuple)]
            sub_name = "_".join([name, *labels])

            slice_dict = {n: i for n, i in zip(extra_axis_names, idx_tuple)}

            if is_pair:
                sub_h = [h[0][slice_dict], h[1][slice_dict]]
            else:
                sub_h = h[slice_dict]

            slices.append((sub_name, sub_h))

        return slices

    def add_systematic(
        self,
        h,
        name,
        process,
        channel,
        kfactor=1,
        mirror=True,
        symmetrize="average",
        add_to_data_covariance=False,
        as_difference=False,
        syst_axes=None,
        **kargs,
    ):
        """
        h: either a single histogram with the systematic variation if mirror=True or a list of two histograms with the up and down variation
        as_difference: if True, interpret the histogram values as the difference with respect to the nominal (i.e. the absolute variation is norm + h)
        syst_axes: optional list of axis names in h that represent independent systematics.
                   If None (default) and h is a hist-like object with axes beyond the channel,
                   the extra axes are auto-detected and each bin combination becomes a separate
                   systematic with name "{name}_{label_0}_{label_1}_...". Pass an empty list
                   to disable auto-detection.
        """

        # multi-systematic dispatch: if h has extra axes beyond the channel,
        # iterate over those and book each combination as an independent systematic
        slices = self._get_systematic_slices(h, name, channel, syst_axes)
        if slices is not None:
            for sub_name, sub_h in slices:
                self.add_systematic(
                    sub_h,
                    sub_name,
                    process,
                    channel,
                    kfactor=kfactor,
                    mirror=mirror,
                    symmetrize=symmetrize,
                    add_to_data_covariance=add_to_data_covariance,
                    as_difference=as_difference,
                    syst_axes=[],
                    **kargs,
                )
            return

        norm = self.dict_norm[channel][process]

        # Use sparse path when norm is stored as scipy sparse CSR
        if self._issparse(norm):
            return self._add_systematic_sparse(
                h,
                name,
                process,
                channel,
                norm,
                kfactor,
                mirror,
                symmetrize,
                add_to_data_covariance,
                as_difference,
                **kargs,
            )

        var_name_out = name

        systematic_type = "normal" if add_to_data_covariance else self.systematic_type

        flow = self.channels[channel]["flow"]

        if isinstance(h, (list, tuple)):
            self._check_hist_and_channel(h[0], channel)
            self._check_hist_and_channel(h[1], channel)

            syst_up = self.get_flat_values(h[0], flow=flow)
            syst_down = self.get_flat_values(h[1], flow=flow)

            if as_difference:
                syst_up = norm + syst_up
                syst_down = norm + syst_down

            logkup_proc = self.get_logk(
                syst_up, norm, kfactor, systematic_type=systematic_type
            )
            logkdown_proc = -self.get_logk(
                syst_down, norm, kfactor, systematic_type=systematic_type
            )

            logkavg_proc, var_name_out = self._compute_asym_syst(
                logkup_proc,
                logkdown_proc,
                name,
                process,
                channel,
                symmetrize,
                add_to_data_covariance,
                **kargs,
            )
        elif mirror:
            self._check_hist_and_channel(h, channel)
            syst = self.get_flat_values(h, flow=flow)

            if as_difference:
                syst = norm + syst

            logkavg_proc = self.get_logk(
                syst, norm, kfactor, systematic_type=systematic_type
            )
        else:
            raise RuntimeError(
                "Only one histogram given but mirror=False, can not construct a variation"
            )

        self.book_logk_avg(logkavg_proc, channel, process, var_name_out)
        self.book_systematic(
            var_name_out, add_to_data_covariance=add_to_data_covariance, **kargs
        )

    def add_beta_variations(
        self,
        h,
        process,
        source_channel,
        dest_channel,
    ):
        """
        Adds a template variation in the destination channel that is correlated with the beta variation in the source channel for a given process
        h: must be a histogram with the axes of the source channel and destiation channel.
        """
        if self.sparse:
            raise NotImplementedError("Sparse implementation not yet implemented")

        if source_channel not in self.channels.keys():
            raise RuntimeError(f"Channel {source_channel} not known!")
        if dest_channel not in self.channels.keys():
            raise RuntimeError(f"Channel {dest_channel} not known!")
        if not self.channels[dest_channel]["masked"]:
            raise RuntimeError(
                f"Beta variations can only be applied to masked channels"
            )

        norm = self.dict_norm[dest_channel][process]

        source_axes = self.channels[source_channel]["axes"]
        dest_axes = self.channels[dest_channel]["axes"]

        source_axes_names = [a.name for a in source_axes]
        dest_axes_names = [a.name for a in dest_axes]

        for a in source_axes:
            if a.name not in h.axes.name:
                raise RuntimeError(
                    f"Axis {a.name} not found in histogram h with {h.axes.name}"
                )
        for a in dest_axes:
            if a.name not in h.axes.name:
                raise RuntimeError(
                    f"Axis {a.name} not found in histogram h with {h.axes.name}"
                )

        flow = self.channels[dest_channel]["flow"]
        variation = h.project(*dest_axes_names, *source_axes_names).values(flow=flow)
        variation = variation.reshape((*norm.shape, -1))

        if source_channel not in self.dict_beta_variations[dest_channel].keys():
            self.dict_beta_variations[dest_channel][source_channel] = {}

        self.dict_beta_variations[dest_channel][source_channel][process] = variation

        self.has_beta_variations = True

    @staticmethod
    def _strcategory_labels(ax):
        """Return the bin labels of a hist StrCategory axis as a numpy string array.

        Raises if ``ax`` is not a StrCategory axis.
        """
        import hist as _hist

        if not isinstance(ax, _hist.axis.StrCategory):
            raise TypeError(
                f"External term axes must be hist.axis.StrCategory; got {type(ax).__name__}"
            )
        return np.array([ax.value(i) for i in range(len(ax))], dtype=object)

    def add_external_likelihood_term(self, grad=None, hess=None, name=None):
        """Add an additive quadratic term to the negative log-likelihood.

        The term has the form

            L_ext(x) = g^T x_sub + 0.5 * x_sub^T H x_sub

        where ``x_sub`` is the slice of the full fit parameter vector
        corresponding to the parameters identified by the StrCategory axes
        of ``grad`` / ``hess``. Both ``grad`` and ``hess`` must use the same
        parameter list in the same order. The parameter names are stored as
        strings and resolved against the full parameter list (POIs + systs)
        at fit time.

        Parameters
        ----------
        grad : hist.Hist, optional
            1D histogram with one ``hist.axis.StrCategory`` axis whose bin
            labels are parameter names. Values are the gradient ``g``.
        hess : hist.Hist or wums.SparseHist, optional
            2D histogram with two ``hist.axis.StrCategory`` axes; both must
            have identical bin labels equal to the gradient parameter list
            (if ``grad`` is also given). Values are the hessian ``H``. May
            be a dense ``hist.Hist`` or a ``wums.SparseHist`` for sparse
            storage. ``H`` should be symmetric (the formula is
            ``0.5 x^T H x``); the user is responsible for symmetrizing.
        name : str, optional
            Identifier for this term. Auto-generated if not provided.
            Multiple terms can be added by calling this method repeatedly.
        """
        if grad is None and hess is None:
            raise ValueError(
                "add_external_likelihood_term requires at least one of grad or hess"
            )

        if name is None:
            name = f"ext{len(self.external_terms)}"
        if any(t["name"] == name for t in self.external_terms):
            raise RuntimeError(f"External likelihood term '{name}' already added")

        params = None

        # Process gradient
        grad_values = None
        if grad is not None:
            if not hasattr(grad, "axes") or len(grad.axes) != 1:
                raise ValueError(
                    f"grad must be a 1D histogram, got {type(grad).__name__} with "
                    f"{len(grad.axes) if hasattr(grad, 'axes') else 0} axes"
                )
            grad_params = self._strcategory_labels(grad.axes[0])
            grad_values = np.asarray(grad.values()).flatten().astype(self.dtype)
            if len(grad_values) != len(grad_params):
                raise RuntimeError(
                    f"grad values length {len(grad_values)} does not match params length {len(grad_params)}"
                )
            params = grad_params

        # Process hessian
        hess_dense = None
        hess_sparse = None
        if hess is not None:
            if len(hess.axes) != 2:
                raise ValueError(
                    f"hess must be a 2D histogram, got {len(hess.axes)} axes"
                )
            hess_params0 = self._strcategory_labels(hess.axes[0])
            hess_params1 = self._strcategory_labels(hess.axes[1])
            if not np.array_equal(hess_params0, hess_params1):
                raise ValueError(
                    "hess must have identical labels on both axes (since it is "
                    "indexed by the same parameter list)"
                )
            if params is not None:
                if not np.array_equal(params, hess_params0):
                    raise ValueError(
                        "grad and hess must use the same parameter list in the "
                        f"same order; got grad params {params.tolist()} vs "
                        f"hess params {hess_params0.tolist()}"
                    )
            else:
                params = hess_params0

            if isinstance(hess, SparseHist):
                # extract sparse coordinates and values from with-flow layout
                # (StrCategory axes never have flow, so this matches the no-flow layout)
                csr = hess.to_flat_csr(self.dtype, flow=False)
                # csr has shape (1, n*n); convert flat positions to (row, col)
                n = len(params)
                flat = np.asarray(csr.indices, dtype=np.int64)
                rows = (flat // n).astype(np.int64)
                cols = (flat % n).astype(np.int64)
                vals = np.asarray(csr.data, dtype=self.dtype)
                hess_sparse = (rows, cols, vals)
            elif self._issparse(hess):
                raise ValueError(
                    "raw scipy sparse hess inputs are not supported; "
                    "wrap in wums.SparseHist with the parameter axes attached"
                )
            else:
                hess_dense = np.asarray(hess.values()).astype(self.dtype)
                if hess_dense.shape != (len(params), len(params)):
                    raise RuntimeError(
                        f"hess shape {hess_dense.shape} does not match "
                        f"params length {len(params)}"
                    )

        self.external_terms.append(
            {
                "name": name,
                "params": np.asarray(params),
                "grad_values": grad_values,
                "hess_dense": hess_dense,
                "hess_sparse": hess_sparse,
            }
        )

    @staticmethod
    def _sparse_values_at(sparse_csr, indices):
        """Extract values from a flat CSR array at the given flat indices.

        Uses searchsorted on the sorted CSR indices to avoid any dense conversion.
        Returns a dense 1D array of values at the requested positions.
        """
        result = np.zeros(len(indices), dtype=sparse_csr.dtype)
        positions = np.searchsorted(sparse_csr.indices, indices)
        valid = (positions < len(sparse_csr.indices)) & (
            sparse_csr.indices[positions] == indices
        )
        result[valid] = sparse_csr.data[positions[valid]]
        return result

    def _get_syst_at_norm_nnz(self, h, norm_csr, flow):
        """Extract flat systematic values only at norm's nonzero positions.

        h can be a histogram, scipy sparse, SparseHist, or dense array.
        ``flow`` controls the flat layout (must match the channel/norm layout).
        Returns a 1D dense array of length norm_csr.nnz.
        """
        nnz_idx = norm_csr.indices
        if hasattr(h, "values"):
            values = h.values(flow=flow)
            return values.flatten().astype(self.dtype)[nnz_idx]
        elif self._issparse(h):
            syst_csr = self._sparse_to_flat_csr(h, self.dtype, flow=flow)
            return self._sparse_values_at(syst_csr, nnz_idx)
        else:
            return np.asarray(h).flatten().astype(self.dtype)[nnz_idx]

    def get_logk(self, syst, norm, kfac=1.0, systematic_type=None):
        if not np.all(np.isfinite(syst)):
            raise RuntimeError(
                f"{len(syst)-sum(np.isfinite(syst))} NaN or Inf values encountered in systematic!"
            )

        # TODO clean this up and avoid duplication
        if systematic_type == "log_normal":
            # check if there is a sign flip between systematic and nominal
            _logk = kfac * np.log(syst / norm)
            _logk_view = np.where(
                np.equal(np.sign(norm * syst), 1),
                _logk,
                self.logkepsilon * np.ones_like(_logk),
            )

            # FIXME does this actually take effect since _logk_view is normally returned?
            if self.clipSystVariations > 0.0:
                _logk = np.clip(_logk, -self.clip, self.clip)

            return _logk_view
        elif systematic_type == "normal":
            _logk = kfac * (syst - norm)
            return _logk
        else:
            raise RuntimeError(
                f"Invalid systematic_type {systematic_type}, valid choices are 'log_normal' or 'normal'"
            )

    def _get_logk_sparse(self, syst_vals, norm_vals, kfac, systematic_type):
        """Compute logk values at norm's nonzero positions only.

        syst_vals and norm_vals are dense 1D arrays of equal length (nnz of norm).
        Returns a 1D dense array of logk values at those positions.
        """
        if not np.all(np.isfinite(syst_vals)):
            raise RuntimeError(
                f"{len(syst_vals)-sum(np.isfinite(syst_vals))} NaN or Inf values encountered in systematic!"
            )

        if systematic_type == "log_normal":
            _logk = kfac * np.log(syst_vals / norm_vals)
            _logk = np.where(
                np.equal(np.sign(norm_vals * syst_vals), 1),
                _logk,
                self.logkepsilon,
            )
            if self.clipSystVariations > 0.0:
                _logk = np.clip(_logk, -self.clip, self.clip)
            return _logk
        elif systematic_type == "normal":
            return kfac * (syst_vals - norm_vals)
        else:
            raise RuntimeError(
                f"Invalid systematic_type {systematic_type}, valid choices are 'log_normal' or 'normal'"
            )

    def book_logk_avg(self, *args):
        self.book_logk(
            self.dict_logkavg,
            self.dict_logkavg_indices,
            *args,
        )

    def book_logk_halfdiff(self, *args):
        self.book_logk(
            self.dict_logkhalfdiff,
            self.dict_logkhalfdiff_indices,
            *args,
        )

    def book_logk(
        self,
        dict_logk,
        dict_logk_indices,
        logk,
        channel,
        process,
        syst_name,
    ):
        if isinstance(logk, tuple):
            # Sparse logk from _add_systematic_sparse: (indices, values, size)
            nnz_idx, logk_vals, size = logk
            nonzero_mask = logk_vals != 0.0
            indices = nnz_idx[nonzero_mask].reshape(-1, 1)
            values = logk_vals[nonzero_mask]
            dict_logk_indices[channel][process][syst_name] = indices
            dict_logk[channel][process][syst_name] = values
            return

        norm = self.dict_norm[channel][process]
        # ensure that systematic tensor is sparse where normalization matrix is sparse
        logk = np.where(np.equal(norm, 0.0), 0.0, logk)
        if self.sparse:
            indices = np.transpose(np.nonzero(logk))
            dict_logk_indices[channel][process][syst_name] = indices
            dict_logk[channel][process][syst_name] = np.reshape(logk[indices], [-1])
        else:
            dict_logk[channel][process][syst_name] = logk

    def book_systematic(
        self,
        name,
        noi=False,
        constrained=True,
        add_to_data_covariance=False,
        groups=None,
    ):

        if add_to_data_covariance:
            if noi:
                raise ValueError(
                    f"{name} is maked as 'noi' but an 'noi' can't be added to the data covariance matrix."
                )
            self.systscovariance.add(name)
        elif not constrained:
            self.systsnoconstraint.add(name)
        else:
            self.systsstandard.add(name)

        if noi:
            self.systsnoi.add(name)

        # below only makes sense if this is an explicit nuisance parameter
        if not add_to_data_covariance:
            if groups is None:
                groups = [name]

            for group in groups:
                self.dict_systgroups[group].add(name)

    def write(self, outfolder="./", outfilename="rabbit_input.hdf5", meta_data_dict={}):

        if self.signals.intersection(self.bkgs):
            raise RuntimeError(
                f"Processes '{self.signals.intersection(self.bkgs)}' found as signal and background"
            )

        procs = sorted(list(self.signals)) + sorted(list(self.bkgs))
        nproc = len(procs)

        nbins = sum(
            [v for c, v in self.nbinschan.items() if not self.channels[c]["masked"]]
        )
        # nbinsfull including masked channels
        nbinsfull = sum([v for v in self.nbinschan.values()])

        logger.info(f"Write out nominal arrays")
        sumw = np.zeros([nbinsfull, nproc], self.dtype)
        sumw2 = np.zeros([nbinsfull, nproc], self.dtype)
        data_obs = np.zeros([nbins], self.dtype)
        data_var = np.zeros([nbins], self.dtype)
        pseudodata = np.zeros([nbins, len(self.pseudodata_names)], self.dtype)
        ibin = 0
        for chan, chan_info in self.channels.items():
            nbinschan = self.nbinschan[chan]

            for iproc, proc in enumerate(procs):
                if proc not in self.dict_norm[chan]:
                    continue

                norm_proc = self.dict_norm[chan][proc]
                if self._issparse(norm_proc):
                    sumw[ibin + norm_proc.indices, iproc] = norm_proc.data
                else:
                    sumw[ibin : ibin + nbinschan, iproc] = norm_proc
                sumw2[ibin : ibin + nbinschan, iproc] = self.dict_sumw2[chan][proc]

            if not chan_info["masked"]:
                data_obs[ibin : ibin + nbinschan] = self.dict_data_obs[chan]
                data_var[ibin : ibin + nbinschan] = self.dict_data_var[chan]

                for idx, name in enumerate(self.pseudodata_names):
                    pseudodata[ibin : ibin + nbinschan, idx] = self.dict_pseudodata[
                        chan
                    ][name]

            ibin += nbinschan

        systs = self.get_systs()
        nsyst = len(systs)

        if self.symmetric_tensor:
            logger.info("No asymmetric systematics - write fully symmetric tensor")

        ibin = 0
        if self.sparse:
            logger.info(f"Write out sparse array")
            norm_sparse_size = 0
            norm_sparse_indices = np.zeros([norm_sparse_size, 2], self.idxdtype)
            norm_sparse_values = np.zeros([norm_sparse_size], self.dtype)

            logk_sparse_size = 0
            logk_sparse_normindices = np.zeros([logk_sparse_size, 1], self.idxdtype)
            logk_sparse_systindices = np.zeros([logk_sparse_size, 1], self.idxdtype)
            logk_sparse_values = np.zeros([logk_sparse_size], self.dtype)

            for chan in self.channels.keys():
                nbinschan = self.nbinschan[chan]
                dict_norm_chan = self.dict_norm[chan]
                dict_logkavg_chan_indices = self.dict_logkavg_indices[chan]
                dict_logkavg_chan_values = self.dict_logkavg[chan]

                for iproc, proc in enumerate(procs):
                    if proc not in dict_norm_chan:
                        continue
                    norm_proc = dict_norm_chan[proc]

                    if self._issparse(norm_proc):
                        # Use scipy sparse structure directly
                        norm_indices = norm_proc.indices.reshape(-1, 1)
                        norm_values = norm_proc.data.copy()

                        nvals = len(norm_values)
                        oldlength = norm_sparse_size
                        norm_sparse_size = oldlength + nvals
                        norm_sparse_indices.resize([norm_sparse_size, 2])
                        norm_sparse_values.resize([norm_sparse_size])

                        out_indices = np.array([[ibin, iproc]]) + np.pad(
                            norm_indices, ((0, 0), (0, 1)), "constant"
                        )
                        norm_indices = None

                        norm_sparse_indices[oldlength:norm_sparse_size] = out_indices
                        out_indices = None

                        norm_sparse_values[oldlength:norm_sparse_size] = norm_values
                        norm_values = None

                        # sorted CSR indices allow searchsorted in logk mapping below
                        norm_nnz_idx = norm_proc.indices
                        oldlength_norm = oldlength
                    else:
                        norm_indices = np.transpose(np.nonzero(norm_proc))
                        norm_values = np.reshape(norm_proc[norm_indices], [-1])

                        nvals = len(norm_values)
                        oldlength = norm_sparse_size
                        norm_sparse_size = oldlength + nvals
                        norm_sparse_indices.resize([norm_sparse_size, 2])
                        norm_sparse_values.resize([norm_sparse_size])

                        out_indices = np.array([[ibin, iproc]]) + np.pad(
                            norm_indices, ((0, 0), (0, 1)), "constant"
                        )
                        norm_indices = None

                        norm_sparse_indices[oldlength:norm_sparse_size] = out_indices
                        out_indices = None

                        norm_sparse_values[oldlength:norm_sparse_size] = norm_values
                        norm_values = None

                        norm_idx_map = (
                            np.cumsum(np.not_equal(norm_proc, 0.0)) - 1 + oldlength
                        )
                        norm_nnz_idx = None

                    dict_logkavg_proc_indices = dict_logkavg_chan_indices[proc]
                    dict_logkavg_proc_values = dict_logkavg_chan_values[proc]

                    for isyst, syst in enumerate(systs):
                        if syst not in dict_logkavg_proc_indices.keys():
                            continue

                        logkavg_proc_indices = dict_logkavg_proc_indices[syst]
                        logkavg_proc_values = dict_logkavg_proc_values[syst]

                        nvals_proc = len(logkavg_proc_values)
                        oldlength = logk_sparse_size
                        logk_sparse_size = oldlength + nvals_proc
                        logk_sparse_normindices.resize([logk_sparse_size, 1])
                        logk_sparse_systindices.resize([logk_sparse_size, 1])
                        logk_sparse_values.resize([logk_sparse_size])

                        # first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
                        # second dimension is flattened in the [2,nsyst] space, where logkavg corresponds to [0,isyst] flattened to isyst
                        # two dimensions are kept in separate arrays for now to reduce the number of copies needed later
                        if norm_nnz_idx is not None:
                            # scipy sparse norm: use searchsorted on sorted CSR indices
                            flat_positions = logkavg_proc_indices.flatten()
                            out_normindices = (
                                np.searchsorted(norm_nnz_idx, flat_positions)
                                + oldlength_norm
                            ).reshape(-1, 1)
                        else:
                            out_normindices = norm_idx_map[logkavg_proc_indices]
                        logkavg_proc_indices = None

                        logk_sparse_normindices[oldlength:logk_sparse_size] = (
                            out_normindices
                        )
                        logk_sparse_systindices[oldlength:logk_sparse_size] = isyst
                        out_normindices = None

                        logk_sparse_values[oldlength:logk_sparse_size] = (
                            logkavg_proc_values
                        )
                        logkavg_proc_values = None

                        if syst in self.dict_logkhalfdiff_indices[chan][proc].keys():
                            logkhalfdiff_proc_indices = self.dict_logkhalfdiff_indices[
                                chan
                            ][proc][syst]
                            logkhalfdiff_proc_values = self.dict_logkhalfdiff[chan][
                                proc
                            ][syst]

                            nvals_proc = len(logkhalfdiff_proc_values)
                            oldlength = logk_sparse_size
                            logk_sparse_size = oldlength + nvals_proc
                            logk_sparse_normindices.resize([logk_sparse_size, 1])
                            logk_sparse_systindices.resize([logk_sparse_size, 1])
                            logk_sparse_values.resize([logk_sparse_size])

                            # out_indices = np.array([[ibin,iproc,isyst,1]]) + np.pad(logkhalfdiff_proc_indices,((0,0),(0,3)),'constant')
                            # first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
                            # second dimension is flattened in the [2,nsyst] space, where logkhalfdiff corresponds to [1,isyst] flattened to nsyst + isyst
                            # two dimensions are kept in separate arrays for now to reduce the number of copies needed later
                            if norm_nnz_idx is not None:
                                flat_positions = logkhalfdiff_proc_indices.flatten()
                                out_normindices = (
                                    np.searchsorted(norm_nnz_idx, flat_positions)
                                    + oldlength_norm
                                ).reshape(-1, 1)
                            else:
                                out_normindices = norm_idx_map[
                                    logkhalfdiff_proc_indices
                                ]
                            logkhalfdiff_proc_indices = None

                            logk_sparse_normindices[oldlength:logk_sparse_size] = (
                                out_normindices
                            )
                            logk_sparse_systindices[oldlength:logk_sparse_size] = (
                                nsyst + isyst
                            )
                            out_normindices = None

                            logk_sparse_values[oldlength:logk_sparse_size] = (
                                logkhalfdiff_proc_values
                            )
                            logkhalfdiff_proc_values = None

                    # free memory
                    dict_logkavg_proc_indices = None
                    dict_logkavg_proc_values = None

                # free memory
                norm_proc = None
                norm_idx_map = None

                ibin += nbinschan

            logger.info(f"Resize and sort sparse arrays into canonical order")
            # resize sparse arrays to actual length
            norm_sparse_indices.resize([norm_sparse_size, 2])
            norm_sparse_values.resize([norm_sparse_size])
            logk_sparse_normindices.resize([logk_sparse_size, 1])
            logk_sparse_systindices.resize([logk_sparse_size, 1])
            logk_sparse_values.resize([logk_sparse_size])

            # straightforward sorting of norm_sparse into canonical order
            norm_sparse_dense_shape = (nbinsfull, nproc)
            norm_sort_indices = np.argsort(
                np.ravel_multi_index(
                    np.transpose(norm_sparse_indices), norm_sparse_dense_shape
                )
            )
            norm_sparse_indices = norm_sparse_indices[norm_sort_indices]
            norm_sparse_values = norm_sparse_values[norm_sort_indices]

            # now permute the indices of the first dimension of logk_sparse corresponding to the resorting of norm_sparse

            # compute the inverse permutation from the sorting of norm_sparse
            # since the final indices are filled from here, need to ensure it has the correct data type
            logk_permute_indices = np.argsort(norm_sort_indices).astype(self.idxdtype)
            norm_sort_indices = None
            logk_sparse_normindices = logk_permute_indices[logk_sparse_normindices]
            logk_permute_indices = None
            logk_sparse_indices = np.concatenate(
                [logk_sparse_normindices, logk_sparse_systindices], axis=-1
            )

            # now straightforward sorting of logk_sparse into canonical order
            if self.symmetric_tensor:
                logk_sparse_dense_shape = (norm_sparse_indices.shape[0], nsyst)
            else:
                logk_sparse_dense_shape = (norm_sparse_indices.shape[0], 2 * nsyst)
            logk_sort_indices = np.argsort(
                np.ravel_multi_index(
                    np.transpose(logk_sparse_indices), logk_sparse_dense_shape
                )
            )
            logk_sparse_indices = logk_sparse_indices[logk_sort_indices]
            logk_sparse_values = logk_sparse_values[logk_sort_indices]
            logk_sort_indices = None

        else:
            logger.info(f"Write out dense array")
            # initialize with zeros, i.e. no variation
            norm = np.zeros([nbinsfull, nproc], self.dtype)
            if self.symmetric_tensor:
                logk = np.zeros([nbinsfull, nproc, nsyst], self.dtype)
            else:
                logk = np.zeros([nbinsfull, nproc, 2, nsyst], self.dtype)

            if self.has_beta_variations:
                beta_variations = np.zeros([nbinsfull, nbins, nproc], self.dtype)

            for chan in self.channels.keys():
                nbinschan = self.nbinschan[chan]
                dict_norm_chan = self.dict_norm[chan]

                for iproc, proc in enumerate(procs):
                    if proc not in dict_norm_chan:
                        continue

                    norm_proc = dict_norm_chan[proc]

                    norm[ibin : ibin + nbinschan, iproc] = norm_proc

                    dict_logkavg_proc = self.dict_logkavg[chan][proc]
                    dict_logkhalfdiff_proc = self.dict_logkhalfdiff[chan][proc]
                    for isyst, syst in enumerate(systs):
                        if syst not in dict_logkavg_proc.keys():
                            continue

                        if self.symmetric_tensor:
                            logk[ibin : ibin + nbinschan, iproc, isyst] = (
                                dict_logkavg_proc[syst]
                            )
                        else:
                            logk[ibin : ibin + nbinschan, iproc, 0, isyst] = (
                                dict_logkavg_proc[syst]
                            )
                            if syst in dict_logkhalfdiff_proc.keys():
                                logk[ibin : ibin + nbinschan, iproc, 1, isyst] = (
                                    dict_logkhalfdiff_proc[syst]
                                )

                    if not self.has_beta_variations:
                        continue

                    for (
                        source_channel,
                        source_channel_dict,
                    ) in self.dict_beta_variations[chan].items():
                        if proc not in source_channel_dict:
                            continue
                        # find the bins of the source channel
                        ibin_start = 0
                        for c, nb in self.nbinschan.items():
                            if self.channels[c]["masked"]:
                                continue  # masked channels can not be source channels
                            if c == source_channel:
                                ibin_end = ibin_start + nb
                                break
                            else:
                                ibin_start += nb
                        else:
                            raise RuntimeError(
                                f"Did not find source channel {source_channel} in list of channels {[k for k in self.nbinschan.keys()]}"
                            )

                        beta_vars = source_channel_dict[proc]
                        beta_variations[
                            ibin : ibin + nbinschan, ibin_start:ibin_end, iproc
                        ] = beta_vars

                ibin += nbinschan

        if self.data_covariance is None and (
            self.systscovariance or self.add_bin_by_bin_stat_to_data_cov
        ):
            # create data covariance
            self.data_covariance = np.diag(data_var)

        # write results to hdf5 file
        procSize = nproc * np.dtype(self.dtype).itemsize
        systSize = 2 * nsyst * np.dtype(self.dtype).itemsize
        amax = np.max([procSize, systSize])
        if amax > self.chunkSize:
            logger.info(
                f"Maximum chunk size in bytes was increased from {self.chunkSize} to {amax} to align with tensor sizes and allow more efficient reading/writing."
            )
            self.chunkSize = amax

        # create HDF5 file (chunk cache set to the chunk size since we can guarantee fully aligned writes
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        outpath = f"{outfolder}/{outfilename}"
        if len(outfilename.split(".")) < 2:
            outpath += ".hdf5"
        logger.info(f"Write output file {outpath}")
        f = h5py.File(outpath, rdcc_nbytes=self.chunkSize, mode="w")

        if meta_data_dict is not None:
            meta_data_dict.update(
                {
                    "channel_info": self.channels,
                    "symmetric_tensor": self.symmetric_tensor,
                    "systematic_type": self.systematic_type,
                }
            )
            if "meta_info" not in meta_data_dict:
                meta_data_dict["meta_info"] = {}

        ioutils.pickle_dump_h5py("meta", meta_data_dict, f)

        noiidxs = self.get_noiidxs()
        systsnoconstraint = self.get_systsnoconstraint()
        systgroups, systgroupidxs = self.get_systgroups()

        # save some lists of strings to the file for later use
        def create_dataset(
            name,
            content,
            length=None,
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",
        ):
            dimension = [len(content), length] if length else [len(content)]
            ds = f.create_dataset(
                f"h{name}", dimension, dtype=dtype, compression=compression
            )
            ds[...] = content

        create_dataset("procs", procs)
        create_dataset("signals", sorted(list(self.signals)))
        create_dataset("systs", systs)
        create_dataset("systsnoconstraint", systsnoconstraint)
        create_dataset("systgroups", systgroups)
        create_dataset(
            "systgroupidxs",
            systgroupidxs,
            dtype=h5py.special_dtype(vlen=np.dtype("int32")),
        )
        create_dataset("noiidxs", noiidxs, dtype="int32")
        create_dataset("pseudodatanames", [n for n in self.pseudodata_names])

        # create h5py datasets with optimized chunk shapes
        nbytes = 0

        constraintweights = self.get_constraintweights(self.dtype)
        nbytes += h5pyutils_write.writeFlatInChunks(
            constraintweights, f, "hconstraintweights", maxChunkBytes=self.chunkSize
        )
        constraintweights = None

        nbytes += h5pyutils_write.writeFlatInChunks(
            data_obs, f, "hdata_obs", maxChunkBytes=self.chunkSize
        )
        if np.any(data_var != data_obs):
            nbytes += h5pyutils_write.writeFlatInChunks(
                data_var, f, "hdata_var", maxChunkBytes=self.chunkSize
            )
            data_var = None
        data_obs = None

        nbytes += h5pyutils_write.writeFlatInChunks(
            pseudodata, f, "hpseudodata", maxChunkBytes=self.chunkSize
        )
        pseudodata = None

        if self.data_covariance is not None:
            for syst in self.systscovariance:
                systv = np.zeros(shape=(nbinsfull, 1), dtype=self.dtype)

                ibin = 0
                for chan in self.channels.keys():
                    nbinschan = self.nbinschan[chan]
                    dict_norm_chan = self.dict_norm[chan]

                    for proc in procs:
                        if proc not in dict_norm_chan:
                            continue

                        dict_logkavg_proc = self.dict_logkavg[chan][proc]

                        if syst not in dict_logkavg_proc.keys():
                            continue

                        systv[ibin : ibin + nbinschan, 0] += dict_logkavg_proc[syst]

                    ibin += nbinschan

                self.data_covariance[...] += systv @ systv.T

            full_cov = (
                np.add(self.data_covariance, np.diag(sumw2))
                if self.add_bin_by_bin_stat_to_data_cov
                else self.data_covariance
            )
            full_cov_inv = np.linalg.inv(full_cov)

            nbytes += h5pyutils_write.writeFlatInChunks(
                full_cov_inv,
                f,
                "hdata_cov_inv",
                maxChunkBytes=self.chunkSize,
            )

        nbytes += h5pyutils_write.writeFlatInChunks(
            sumw, f, "hsumw", maxChunkBytes=self.chunkSize
        )

        nbytes += h5pyutils_write.writeFlatInChunks(
            sumw2, f, "hsumw2", maxChunkBytes=self.chunkSize
        )

        if self.sparse:
            nbytes += h5pyutils_write.writeSparse(
                norm_sparse_indices,
                norm_sparse_values,
                norm_sparse_dense_shape,
                f,
                "hnorm_sparse",
                maxChunkBytes=self.chunkSize,
            )
            norm_sparse_indices = None
            norm_sparse_values = None
            nbytes += h5pyutils_write.writeSparse(
                logk_sparse_indices,
                logk_sparse_values,
                logk_sparse_dense_shape,
                f,
                "hlogk_sparse",
                maxChunkBytes=self.chunkSize,
            )
            logk_sparse_indices = None
            logk_sparse_values = None
        else:
            nbytes += h5pyutils_write.writeFlatInChunks(
                norm, f, "hnorm", maxChunkBytes=self.chunkSize
            )
            norm = None
            nbytes += h5pyutils_write.writeFlatInChunks(
                logk, f, "hlogk", maxChunkBytes=self.chunkSize
            )
            logk = None

            if self.has_beta_variations:
                nbytes += h5pyutils_write.writeFlatInChunks(
                    beta_variations, f, "hbetavariations", maxChunkBytes=self.chunkSize
                )
                beta_variations = None

        # Write external likelihood terms
        if self.external_terms:
            ext_group = f.create_group("external_terms")
            create_dataset(
                "external_term_names",
                [t["name"] for t in self.external_terms],
            )
            for term in self.external_terms:
                term_group = ext_group.create_group(term["name"])
                params_ds = term_group.create_dataset(
                    "params",
                    [len(term["params"])],
                    dtype=h5py.special_dtype(vlen=str),
                    compression="gzip",
                )
                params_ds[...] = [str(p) for p in term["params"]]

                if term["grad_values"] is not None:
                    nbytes += h5pyutils_write.writeFlatInChunks(
                        term["grad_values"],
                        term_group,
                        "grad_values",
                        maxChunkBytes=self.chunkSize,
                    )

                if term["hess_dense"] is not None:
                    nbytes += h5pyutils_write.writeFlatInChunks(
                        term["hess_dense"],
                        term_group,
                        "hess_dense",
                        maxChunkBytes=self.chunkSize,
                    )
                elif term["hess_sparse"] is not None:
                    rows, cols, vals = term["hess_sparse"]
                    n = len(term["params"])
                    indices = np.stack([rows, cols], axis=-1).astype(self.idxdtype)
                    nbytes += h5pyutils_write.writeSparse(
                        indices,
                        vals.astype(self.dtype),
                        (n, n),
                        term_group,
                        "hess_sparse",
                        maxChunkBytes=self.chunkSize,
                    )

        logger.info(f"Total raw bytes in arrays = {nbytes}")

    def get_systsstandard(self):
        return list(common.natural_sort(self.systsstandard))

    def get_systsnoi(self):
        return list(common.natural_sort(self.systsnoi))

    def get_systsnoconstraint(self):
        return list(common.natural_sort(self.systsnoconstraint))

    def get_systs(self):
        return self.get_systsnoconstraint() + self.get_systsstandard()

    def get_constraintweights(self, dtype):
        systs = self.get_systs()
        constraintweights = np.ones([len(systs)], dtype=dtype)
        for syst in self.get_systsnoconstraint():
            constraintweights[systs.index(syst)] = 0.0
        return constraintweights

    def get_groups(self, group_dict):
        systs = self.get_systs()
        groups = []
        idxs = []
        for group, members in common.natural_sort_dict(group_dict).items():
            groups.append(group)
            idx = []
            for syst in members:
                idx.append(systs.index(syst))
            idxs.append(idx)
        return groups, idxs

    def get_noiidxs(self):
        # list of indeces of nois w.r.t. systs
        systs = self.get_systs()
        idxs = []
        for noi in self.get_systsnoi():
            idxs.append(systs.index(noi))
        return idxs

    def get_systgroups(self):
        # list of groups of systematics (nuisances) and lists of indexes
        return self.get_groups(self.dict_systgroups)
