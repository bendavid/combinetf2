import h5py
import hist
import numpy as np
import tensorflow as tf

from rabbit.h5pyutils_read import makesparsetensor, maketensor


class FitInputData:
    def __init__(self, filename, pseudodata=None):
        with h5py.File(filename, mode="r") as f:

            # load text arrays from file
            self.procs = f["hprocs"][...]
            self.signals = f["hsignals"][...]
            self.systs = f["hsysts"][...]
            self.systsnoconstraint = f["hsystsnoconstraint"][...]
            self.systgroups = f["hsystgroups"][...]
            self.systgroupidxs = f["hsystgroupidxs"][...]

            self.noiidxs = (
                f["hnoiidxs"][...]
                if "hnoiidxs" in f.keys()
                else f["hnoigroupidxs"][...]
            )

            if "hpseudodatanames" in f.keys():
                self.pseudodatanames = f["hpseudodatanames"][...].astype(str)
            else:
                self.pseudodatanames = []

            # load arrays from file

            if "hdata_cov_inv" in f.keys():
                hdata_cov_inv = f["hdata_cov_inv"]
                self.data_cov_inv = maketensor(hdata_cov_inv)
            else:
                self.data_cov_inv = None

            # load data/pseudodata
            if pseudodata is not None:
                print("Initialize pseudodata")
                hpseudodata_obs = f["hpseudodata"]

                self.pseudodata_obs = maketensor(hpseudodata_obs)
                if "hpseudodatavar" in f.keys():
                    hpseudodata_var = f["hpseudodatavar"]
                    self.pseudodata_var = maketensor(hpseudodata_var)
                else:
                    self.pseudodata_var = None

                # if explicit pseudodata sets are requested, select them (keep all for empty list)
                if len(pseudodata) > 0:
                    # Check if all pseudodata are in self.pseudodatanames
                    if not np.all(np.isin(pseudodata, self.pseudodatanames)):
                        missing = [
                            pd for pd in pseudodata if pd not in self.pseudodatanames
                        ]
                        raise ValueError(f"Missing pseudodata: {missing}")

                    mask = np.isin(self.pseudodatanames, pseudodata)
                    indices = np.where(mask)[0]
                    if not indices.size:
                        raise ValueError(
                            f"Invalid pseudodata choice {pseudodata}. Valid choices are {self.pseudodatanames}"
                        )

                    self.pseudodata_obs = tf.gather(
                        self.pseudodata_obs, indices, axis=1
                    )
                    self.pseudodatanames = self.pseudodatanames[indices]

            self.data_obs = maketensor(f["hdata_obs"])
            if "hdata_var" in f.keys():
                self.data_var = maketensor(f["hdata_var"])
            else:
                self.data_var = None

            # start by creating tensors which read in the hdf5 arrays (optimized for memory consumption)
            self.constraintweights = maketensor(f["hconstraintweights"])

            self.sparse = not "hnorm" in f

            if self.sparse:
                self.norm = makesparsetensor(f["hnorm_sparse"])
                self.logk = makesparsetensor(f["hlogk_sparse"])
                # Canonicalize index ordering once at load time. The fitter's
                # sparse fast path reduces nonzero entries via row-keyed
                # reductions; sorted row-major indices give coalesced memory
                # access. tf.sparse.reorder sorts into row-major order.
                self.norm = tf.sparse.reorder(self.norm)
                self.logk = tf.sparse.reorder(self.logk)
                # Pre-build a CSRSparseMatrix view of logk for use in the
                # fitter's sparse matvec path via sm.matmul, which dispatches
                # to a multi-threaded CSR kernel and is much faster per call
                # than the equivalent gather + unsorted_segment_sum. NOTE:
                # SparseMatrixMatMul has no XLA kernel, so any tf.function
                # that calls sm.matmul must be built with jit_compile=False.
                from tensorflow.python.ops.linalg.sparse import (
                    sparse_csr_matrix_ops as _tf_sparse_csr,
                )

                self.logk_csr = _tf_sparse_csr.CSRSparseMatrix(self.logk)
            else:
                self.norm = maketensor(f["hnorm"])
                self.logk = maketensor(f["hlogk"])
            if "hbetavariations" in f.keys():
                self.betavar = maketensor(f["hbetavariations"])
            else:
                self.betavar = None

            # infer some metadata from loaded information
            self.dtype = self.data_obs.dtype
            self.nbins = self.data_obs.shape[-1]
            self.nbinsfull = self.norm.shape[0]
            self.nbinsmasked = self.nbinsfull - self.nbins
            self.nproc = len(self.procs)
            self.nsyst = len(self.systs)
            self.nsystnoconstraint = len(self.systsnoconstraint)
            self.nsignals = len(self.signals)
            self.nsystgroups = len(self.systgroups)

            # reference meta data if available
            self.metadata = {}
            if "meta" in f.keys():
                from wums.ioutils import pickle_load_h5py

                self.metadata = pickle_load_h5py(f["meta"])
                self.channel_info = self.metadata["channel_info"]
            else:
                self.channel_info = {
                    "ch0": {
                        "axes": [
                            hist.axis.Integer(
                                0,
                                self.nbins,
                                underflow=False,
                                overflow=False,
                                name="obs",
                            )
                        ]
                    }
                }
                if self.nbinsmasked:
                    self.channel_info["ch1_masked"] = {
                        "axes": [
                            hist.axis.Integer(
                                0,
                                self.nbinsmasked,
                                underflow=False,
                                overflow=False,
                                name="masked",
                            )
                        ]
                    }

            self.symmetric_tensor = self.metadata.get("symmetric_tensor", False)

            if self.metadata.get("exponential_transform", False):
                raise NotImplementedError(
                    "exponential_transform functionality has been removed.   Please use systematic_type normal instead"
                )

            self.systematic_type = self.metadata.get("systematic_type", "log_normal")

            if "hsumw2" in f.keys():
                self.sumw = maketensor(f["hsumw"])
                self.sumw2 = maketensor(f["hsumw2"])
            else:
                # fallback for older datacards
                kstat = maketensor(f["hkstat"])

                self.sumw = self.expected_events_nominal()
                self.sumw2 = self.sumw**2 / kstat

                self.sumw2 = tf.where(kstat == 0.0, self.sumw, self.sumw2)

            # compute indices for channels
            ibin = 0
            for channel, info in self.channel_info.items():
                axes = info["axes"]
                flow = info.get("flow", False)
                shape = tuple([a.extent if flow else a.size for a in axes])
                size = int(np.prod(shape))

                start = ibin
                stop = start + size

                info["start"] = start
                info["stop"] = stop

                ibin = stop

            for channel, info in self.channel_info.items():
                print(channel, info)

            self.axis_procs = hist.axis.StrCategory(self.procs, name="processes")

            # Load external likelihood terms (optional).
            # Each entry is a dict with keys:
            #   name: str
            #   params: 1D ndarray of parameter name strings
            #   grad_values: 1D float ndarray or None
            #   hess_dense: 2D float ndarray or None
            #   hess_sparse: tuple (rows, cols, values) or None
            self.external_terms = []
            if "external_terms" in f.keys():
                names = [
                    s.decode() if isinstance(s, bytes) else s
                    for s in f["hexternal_term_names"][...]
                ]
                ext_group = f["external_terms"]
                for tname in names:
                    tg = ext_group[tname]
                    raw_params = tg["params"][...]
                    params = np.array(
                        [s.decode() if isinstance(s, bytes) else s for s in raw_params]
                    )
                    grad_values = (
                        np.asarray(maketensor(tg["grad_values"]))
                        if "grad_values" in tg.keys()
                        else None
                    )
                    hess_dense = (
                        np.asarray(maketensor(tg["hess_dense"]))
                        if "hess_dense" in tg.keys()
                        else None
                    )
                    hess_sparse = None
                    if "hess_sparse" in tg.keys():
                        hg = tg["hess_sparse"]
                        idx_dset = hg["indices"]
                        if "original_shape" in idx_dset.attrs:
                            idx_shape = tuple(idx_dset.attrs["original_shape"])
                            indices = np.asarray(idx_dset).reshape(idx_shape)
                        else:
                            indices = np.asarray(idx_dset)
                        rows = indices[:, 0]
                        cols = indices[:, 1]
                        vals = np.asarray(hg["values"])
                        hess_sparse = (rows, cols, vals)
                    self.external_terms.append(
                        {
                            "name": tname,
                            "params": params,
                            "grad_values": grad_values,
                            "hess_dense": hess_dense,
                            "hess_sparse": hess_sparse,
                        }
                    )

    @tf.function
    def expected_events_nominal(self):
        rnorm = tf.ones(self.nproc, dtype=self.dtype)
        mrnorm = tf.expand_dims(rnorm, -1)

        if self.sparse:
            nexpfullcentral = tf.sparse.sparse_dense_matmul(self.norm, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral, -1)
        else:
            nexpfullcentral = tf.matmul(self.norm, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral, -1)

        return nexpfullcentral
