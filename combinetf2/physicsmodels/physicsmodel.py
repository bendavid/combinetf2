import hist
import tensorflow as tf


class PhysicsModel:
    """
    Processing the flat input vector. can be used to inherit custom physics models from.
    """

    need_observables = True  # if observables should be provided to the compute function
    has_data = True  # if data histograms are stored or not, and if chi2 is calculated
    ndf_reduction = 0  # how much will be subtracted from the ndf / number of bins, e.g. for chi2 calculation

    def __init__(self, indata, key):
        # The result of a model in the output dictionary is stored under 'result = fitresult[cls.name]'
        #   if the model can have different instances 'self.instance' must be set to a unique string and the result will be stored in 'result = result[self.instance]'
        #   each model can have different channels that are the same or different from channels from the input data. All channel specific results are stored under 'result["channels"]'
        self.key = (
            key  # where to store the results of this model in the results dictionary
        )

    # class function to parse strings as given by the argparse input e.g. -m PhysicsModel <arg[0]> <args[1]> ...
    @classmethod
    def parse_args(cls, indata, *args):
        key = " ".join([cls.__name__, *args])
        return cls(indata, key, *args)

    # function to compute the transformation of the physics model, has to be differentiable.
    #    For custom physics models, this function should be overridden.
    #    observables are the provided histograms inclusive in processes: nbins
    #    params are the fit parameters
    def compute_flat(self, poi, theta, observables=None):
        return observables

    # function to compute the transformation of the physics model, has to be differentiable.
    #    For custom physics models, this function can be overridden.
    #    observables are the provided histograms per process: nbins x nprocesses
    #    params are the fit parameters
    def compute_flat_per_process(self, poi, theta, observables=None):
        return self.compute_flat(poi, theta, observables)

    # # generic version which should not need to be overridden
    # def make_fun(self, fun_flat, params, inclusive=True):
    #     compute = self.compute_flat if inclusive else self.compute_flat_per_process
    #
    #     def fun():
    #         if self.need_observables:
    #             exp = compute(params, fun_flat())
    #         else:
    #             exp = compute(params)
    #         return exp
    #
    #     return fun

    # generic version which should not need to be overridden
    @tf.function
    def get_data(self, poi, theta, data, data_cov_inv=None):
        with tf.GradientTape() as t:
            t.watch(data)
            output = self.compute_flat(poi, theta, data)

        jacobian = t.jacobian(
            output, data, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )

        # Ensure the Jacobian has at least 2 dimensions (expand in case output is a scalar)
        if len(jacobian.shape) == 1:
            jacobian = tf.expand_dims(jacobian, axis=0)

        if data_cov_inv is None:
            # Assume poisson uncertainties on data
            cov_output = (jacobian * data) @ tf.transpose(jacobian)
        else:
            # General case with full covariance matrix
            data_cov = tf.linalg.inv(data_cov_inv)
            cov_output = jacobian @ data_cov @ tf.transpose(jacobian)

        variances_output = tf.linalg.diag_part(cov_output)

        return output, variances_output, cov_output


class PhysicsModelChannel(PhysicsModel):
    """
    Abstract physics model to process a specific channel
    """

    def __init__(self, indata, key, channel):
        super().__init__(indata, key)
        channel_info = indata.channel_info[channel]
        self.channel_info = {channel: channel_info}

        self.start = channel_info["start"]
        self.stop = channel_info["stop"]
        self.channel_shape = [len(a) for a in channel_info["axes"]]

    def compute(self, params, observables):
        return observables

    def compute_per_process(self, params, observables):
        return self.compute(params, observables)

    def compute_flat(self, poi, theta, observables):
        exp = tf.reshape(observables[self.start : self.stop], tuple(self.channel_shape))
        exp = self.compute(poi, theta, exp)
        exp = tf.reshape(exp, [-1])  # flatten again
        return exp

    def compute_flat_per_process(self, poi, theta, observables):
        exp = tf.reshape(
            observables[self.start : self.stop],
            (*self.channel_shape, observables.shape[1]),
        )
        exp = self.compute_per_process(poi, theta, exp)
        # flatten again
        flat_shape = (-1, exp.shape[-1])
        exp = tf.reshape(exp, flat_shape)
        return exp


class Basemodel(PhysicsModel):
    """
    A class to output histograms without any transformation, can be used as base class to inherit custom physics models from.
    """

    def __init__(self, indata, key):
        super().__init__(indata, key)
        self.channel_info = indata.channel_info


class BasemodelParameters(PhysicsModel):
    """
    A class to output pois and nuisances without any transformation, can be used as base class to inherit custom physics models from.
    """

    need_observables = False
    has_data = False

    # FIXME get the pois too

    def __init__(self, indata, key):
        super().__init__(indata, key)

        self.channel_info = {
            "parms": {
                "axes": [
                    hist.axis.StrCategory(
                        indata.systs,
                        name="parms",
                    )
                ]
            }
        }

    def compute_flat(self, poi, theta, observables):
        return theta
