from collections.abc import Mapping, Sequence

import numpy as np

import keras

from keras.optimizers.schedules import LearningRateSchedule  # type: ignore

from bayesflow.types import Tensor
from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.networks.inference_network import InferenceNetwork
from bayesflow.networks.summary_network import SummaryNetwork
from bayesflow.distributions.distribution import Distribution
from bayesflow.utils import (
    filter_kwargs,
    logging,
    concatenate_valid_shapes,
    concatenate_valid,
    repeat_valid,
    split_arrays,
)
from bayesflow.networks.standardization import Standardization
from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("bayesflow.approximators")
class SelfConsistentApproximator(Approximator):
    """
    Defines a workflow for performing fast posterior inference.
    The posterior is approximated with a posterior network and
    an optional summary network trained with a self-consistency loss.

    To calculate the self-consistency loss, one needs to supply the prior and likelihood.
    Both can be either analytic distributions or approximated via inference networks.

    Parameters
    ----------
    adapter: bayesflow.adapters.Adapter
        Adapter for data processing.
    prior_network: bayesflow.networks.InferenceNetwork | bayesflow.distributions.Distribution
        Prior distribution of the parameters.
        Needs to be an instance of a class with `.log_prob` method.
    likelihood_network: bayesflow.networks.InferenceNetwork | bayesflow.distributions.Distribution
        Likelihood of the data given parameters.
        Needs to be an instance of a class with `.log_prob` method.
    posterior_network: bayesflow.networks.InferenceNetwork | bayesflow.distributions.Distribution
        Posterior of the parameters given data.
        Needs to be an instance of a class with `.log_prob` and `.sample` methods.
    summary_network: bayesflow.networks.SummaryNetwork, optional
        The summary network used for data summarization.
    standardize: str | Sequence[str] | None
        The variables to standardize before passing to the networks. Can be either
        "all" or any subset of ["inference_variables", "summary_variables", "inference_conditions"].
        (default is "all").
    num_sc_samples: int
        Number of posterior samples over which to calculate
        the variance of the marginal likelihood approximation during training (for SC loss).
    sc_gradient: str | Sequence[str] | None
        Which network(s) should be trained by the SC loss. Can be either "all"
        or any subset of ["posterior", "likelihood", "prior"]. Defaults to "posterior".
    likelihood_summary: bool
        Whether the likelihood is computed on the output
        of the summary network (`True`) or on the raw data (`False`).
        Note that when `True`, the `.log_marginal_likelihood` returns
        biased estimates that are not to be used for model comparison.
    loss_schedules: dict[str, float | LearningRateSchedule]
        Scaling of the training losses. Can depend on the training step based on a custom schedule.
        The keys of the 4 loss components are:
        'prior_network', 'likelihood_network', 'posterior_network', 'summary_network', and 'self-consistency'
        By default, will be set to 1.0. To turn off computing a loss entirely (e.g., for pure SC training),
         set to False.
    **kwargs : dict, optional
        Additional arguments passed to the :py:class:`bayesflow.approximators.Approximator` class.
    """

    def __init__(
        self,
        adapter: Adapter,
        prior_network: InferenceNetwork | Distribution,
        likelihood_network: InferenceNetwork | Distribution,
        posterior_network: InferenceNetwork | Distribution,
        summary_network: SummaryNetwork = None,
        standardize: str | Sequence[str] | None = None,
        num_sc_samples: int = 16,
        sc_gradient: str | Sequence[str] | None = "posterior",
        likelihood_summary: bool = False,
        loss_schedules: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.adapter = adapter
        self.prior_network = prior_network
        self.likelihood_network = likelihood_network
        self.posterior_network = posterior_network
        self.summary_network = summary_network
        self.num_sc_samples = num_sc_samples
        if isinstance(sc_gradient, str):
            if sc_gradient == "all":
                self.sc_gradient = ["prior", "likelihood", "posterior"]
            else:
                self.sc_gradient = [sc_gradient]
        else:
            self.sc_gradient = sc_gradient or []
        if likelihood_summary and summary_network is None:
            raise ValueError("Summary network needs to be defined for computing the summary likelihood.")
        self.likelihood_summary = likelihood_summary

        if isinstance(standardize, str) and standardize != "all":
            self.standardize = [standardize]
        else:
            self.standardize = standardize or []

        if self.standardize == "all":
            self.standardize_layers = None
        else:
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

        if loss_schedules is None:
            loss_schedules = dict()
        self.loss_schedules = loss_schedules

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        data_summary_shape = data_shapes["data"]
        if self.summary_network is not None:
            if not self.summary_network.built:
                self.summary_network.build(data_shapes["data"])
            data_summary_shape = self.summary_network.compute_output_shape(data_shapes["data"])

            if self.loss_schedules.get("summary_network") is None:
                self.loss_schedules["summary_network"] = 1.0

        if not self.prior_network.built:
            self.prior_network.build(data_shapes["parameters"], data_shapes.get("conditions"))

        if not self.likelihood_network.built:
            likelihood_conditions_shape = concatenate_valid_shapes(
                (data_shapes["parameters"], data_shapes.get("conditions")), axis=-1
            )
            if self.likelihood_summary:
                self.likelihood_network.build(data_summary_shape, likelihood_conditions_shape)
            else:
                self.likelihood_network.build(data_shapes["data"], likelihood_conditions_shape)

        if not self.posterior_network.built:
            posterior_conditions_shape = concatenate_valid_shapes(
                (data_summary_shape, data_shapes.get("conditions")), axis=-1
            )
            self.posterior_network.build(data_shapes["parameters"], posterior_conditions_shape)

        # add fixed schedules if not defined
        if self.loss_schedules.get("prior_network") is None:
            self.loss_schedules["prior_network"] = 1.0
        if self.loss_schedules.get("likelihood_network") is None:
            self.loss_schedules["likelihood_network"] = 1.0
        if self.loss_schedules.get("posterior_network") is None:
            self.loss_schedules["posterior_network"] = 1.0
        if self.loss_schedules.get("self-consistency") is None:
            self.loss_schedules["self-consistency"] = 1.0

        if self.standardize == "all":
            self.standardize = [var for var in ["parameters", "data", "conditions"] if var in data_shapes]
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

        for var, layer in self.standardize_layers.items():
            layer.build(data_shapes[var])

        self.step = keras.Variable(0, trainable=False)

        self.built = True

    def build_from_data(self, adapted_data: dict[str, any]):
        self.build(keras.tree.map_structure(keras.ops.shape, adapted_data))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "adapter": self.adapter,
            "prior_network": self.prior_network,
            "likelihood_network": self.likelihood_network,
            "posterior_network": self.posterior_network,
            "summary_network": self.summary_network,
            "standardize": self.standardize,
            "num_sc_samples": self.num_sc_samples,
            "sc_gradient": self.sc_gradient,
            "likelihood_summary": self.likelihood_summary,
            "loss_schedules": self.loss_schedules,
        }

        return base_config | serialize(config)

    def _batch_size_from_data(self, data: any):
        """
        Fetches the current batch size from an input dictionary. Can only be used during training when
        inference variables as present. By default, assumes SBI loss is computed on 'parameters'. If
        that is not found then SC loss only is assumed, in which case the batch size is computed from 'sc_data'.
        """
        if data.get("parameters") is not None:
            return keras.ops.shape(data["parameters"])[0]
        elif data.get("sc_data") is not None:
            return keras.ops.shape(data["sc_data"])[0]
        else:
            raise ValueError("data does not contain parameters or sc_data, cannot infer batch size.")

    def compute_metrics(
        self,
        parameters: Tensor = None,
        data: Tensor = None,
        conditions: Tensor = None,
        sc_data: Tensor = None,
        sc_conditions: Tensor = None,
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        """
        Computes loss and tracks metrics for the prior, likelihood, posterior, and summary networks.

        This method orchestrates the end-to-end computation of metrics and loss for the model.
        It handles standardization of input variables, combines summary outputs with conditions when necessary,
        and aggregates loss and all tracked metrics into a unified dictionary. The returned dictionary
        includes both the total loss and all individual metrics, with keys indicating their source.

        Parameters
        ----------
        parameters: Tensor
            Input tensors for the posterior network, and conditioning tensor for the likelihood network.
        data: Tensor
            Input tensors for the summary network and likelihood network.
        conditions: Tensor
            Conditioning tensors for all inference networks (prior, likelihood, and posterior).
        sc_data: Tensor
            Input tensor for the posterior network for calculating the SC loss.
        sc_condition: Tensor
            Conditioning tensors for all inference networks (prior, likelihood, and posterior)
              for calculating the SC loss.

        Returns
        -------
        metrics: dict[str, Tensor]
            Dictionary containing the total loss under the key "loss",
            as well as all tracked metrics for the prior, likelihood, posterior, and summary networks.
            Each metric key is prefixed with
            "prior_", "likelihood_", "posterior_", or "summary_" to indicate its source.
        """
        # standardize inputs before doing any computations
        if "parameters" in self.standardize:
            if parameters is not None:
                parameters = self.standardize_layers["parameters"](parameters, stage=stage)
        if "data" in self.standardize:
            if data is not None:
                data = self.standardize_layers["data"](data, stage=stage)
            if sc_data is not None:
                sc_data = self.standardize_layers["data"](sc_data, stage="inference")
        if "conditions" in self.standardize:
            if conditions is not None:
                conditions = self.standardize_layers["conditions"](conditions, stage=stage)
            if sc_conditions is not None:
                sc_conditions = self.standardize_layers["conditions"](sc_conditions, stage="inference")

        metrics, total_loss, data_summary = self._summary_metrics(data, stage)

        # prior
        metric, loss = self._prior_metrics(parameters, conditions, stage)
        metrics = metrics | metric
        total_loss += loss

        # posterior
        metric, loss = self._posterior_metrics(parameters, data_summary, conditions, stage)
        metrics = metrics | metric
        total_loss += loss

        # likelihood
        # gradient of the summary network should not be propagated here
        # because the summaries are used as 'fixed' input
        metric, loss = self._likelihood_metrics(
            keras.ops.stop_gradient(data_summary) if self.likelihood_summary else data, parameters, conditions, stage
        )
        metrics = metrics | metric
        total_loss += loss

        # self-consistency
        metric, loss = self._self_consistency_metrics(sc_data, sc_conditions)
        metrics = metrics | metric
        total_loss += loss

        metrics = {"loss": total_loss} | metrics

        # tick step counter (for sc_lambda schedule)
        if stage == "training":
            self.step.assign_add(1)

        return metrics

    def _prior_metrics(self, parameters: Tensor, conditions: Tensor, stage: str) -> tuple[dict, float]:
        if not isinstance(self.prior_network, InferenceNetwork):
            return {}, keras.ops.zeros(())

        if self.loss_schedules["prior_network"] is False:
            return {}, keras.ops.zeros(())

        metrics = self.prior_network.compute_metrics(parameters, conditions=conditions, stage=stage)
        loss = metrics.get("loss", keras.ops.zeros(()))
        metrics = {f"{key}/prior_{key}": value for key, value in metrics.items()}

        # weight the loss by schedule
        if isinstance(self.loss_schedules["prior_network"], LearningRateSchedule):
            lam = self.loss_schedules["prior_network"](self.step)
            metrics = metrics | {"lambda/prior": lam}
        else:
            lam = self.loss_schedules["prior_network"]

        loss = lam * loss

        return metrics, loss

    def _likelihood_metrics(
        self, data: Tensor, parameters: Tensor, conditions: Tensor, stage: str
    ) -> tuple[dict, float]:
        if not isinstance(self.likelihood_network, InferenceNetwork):
            return {}, keras.ops.zeros(())

        if self.loss_schedules["likelihood_network"] is False:
            return {}, keras.ops.zeros(())

        metrics = self.likelihood_network.compute_metrics(
            data, conditions=concatenate_valid((parameters, conditions), axis=-1), stage=stage
        )
        loss = metrics.get("loss", keras.ops.zeros(()))
        metrics = {f"{key}/likelihood_{key}": value for key, value in metrics.items()}

        # weight the loss by schedule
        if isinstance(self.loss_schedules["likelihood_network"], LearningRateSchedule):
            lam = self.loss_schedules["likelihood_network"](self.step)
            metrics = metrics | {"lambda/likelihood": lam}
        else:
            lam = self.loss_schedules["likelihood_network"]

        loss = lam * loss

        return metrics, loss

    def _posterior_metrics(
        self, parameters: Tensor, data: Tensor, conditions: Tensor, stage: str
    ) -> tuple[dict, float]:
        if not isinstance(self.posterior_network, InferenceNetwork):
            return {}, keras.ops.zeros(())

        if self.loss_schedules["posterior_network"] is False:
            return {}, keras.ops.zeros(())

        metrics = self.posterior_network.compute_metrics(
            parameters, conditions=concatenate_valid((data, conditions), axis=-1), stage=stage
        )

        loss = metrics.get("loss", keras.ops.zeros(()))
        metrics = {f"{key}/posterior_{key}": value for key, value in metrics.items()}

        # weight the loss by schedule
        if isinstance(self.loss_schedules["posterior_network"], LearningRateSchedule):
            lam = self.loss_schedules["posterior_network"](self.step)
            metrics = metrics | {"lambda/posterior": lam}
        else:
            lam = self.loss_schedules["posterior_network"]

        loss = lam * loss

        return metrics, loss

    def _summary_metrics(self, data: Tensor, stage: str) -> tuple[dict, float, Tensor | None]:
        if self.summary_network is None:
            return {}, keras.ops.zeros(()), data

        if data is None:
            return {}, keras.ops.zeros(()), data

        metrics = self.summary_network.compute_metrics(data, stage=stage)
        data = metrics.pop("outputs")

        if self.loss_schedules["summary_network"] is False:
            return {}, keras.ops.zeros(()), data

        loss = metrics.get("loss", keras.ops.zeros(()))
        metrics = {f"{key}/summary_{key}": value for key, value in metrics.items()}

        # weight the loss by schedule
        if isinstance(self.loss_schedules["summary_network"], LearningRateSchedule):
            lam = self.loss_schedules["summary_network"](self.step)
            metrics = metrics | {"lambda/summary": lam}
        else:
            lam = self.loss_schedules["summary_network"]

        loss = lam * loss

        return metrics, loss, data

    def _self_consistency_metrics(self, data: Tensor, conditions: Tensor) -> tuple[dict, float]:
        if self.num_sc_samples == 0:
            return {}, keras.ops.zeros(())

        if self.loss_schedules["self-consistency"] is False:
            return {}, keras.ops.zeros(())

        log_ml = self._log_marginal_likelihood(num_samples=self.num_sc_samples, data=data, conditions=conditions)
        loss = keras.ops.var(log_ml, axis=-1)
        loss = keras.ops.mean(loss)

        metrics = {"loss/self-consistency_loss": loss}

        # weight the loss by schedule
        if isinstance(self.loss_schedules["self-consistency"], LearningRateSchedule):
            lam = self.loss_schedules["self-consistency"](self.step)
            metrics = metrics | {"lambda/self-consistency": lam}
        else:
            lam = self.loss_schedules["self-consistency"]

        loss = lam * loss

        return metrics, loss

    def log_marginal_likelihood(self, num_samples: int, conditions: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Estimates the log marginal likelihood log p(y) = log ∫ p(θ) p(y | θ) dθ by inverting the Bayes' theorem:
        p(y) = p(y | θ) p(θ) / p(θ | y), where the posterior, likelihood, or prior densities may be approximated.

        The RHS is evaluated on samples from the posterior θ_1, ..., θ_k ~ p(θ | y)

        Note that when using approximate likelihood on a summary space h(y),
        the estimate of the likelihood is biased by an unknown factor of f(y): p(y | θ) = f(y) p(h(y) | θ)
        (see Fisher-Neyman factorization theorem).
        Therefore, the estimated marginal likelihood cannot be used for model comparison because p(y) != p(h(y)).

        Parameters
        ----------
        num_samples: int
            Number of posterior samples to evaluate the marginal likelihood on.
        conditions : dict[str, np.ndarray]
            Dictionary of conditioning variables as NumPy arrays.
        **kwargs : dict
            Additional keyword arguments for the adapter.

        Returns
        -------
        np.ndarray
            Estimates of the log marginal likelihood, averaged over `num_samples`.
        """
        if self.likelihood_summary:
            logging.warning(
                "Estimates of the marginal likelihood are biased "
                "when likelihood density is computed on the summary space!"
            )

        conditions = self.adapter(conditions, strict=False, stage="inference", **kwargs)

        for key in ["data", "conditions"]:
            if key in self.standardize and key in conditions:
                conditions[key] = self.standardize_layers[key](conditions[key])

        conditions = keras.tree.map_structure(keras.ops.convert_to_tensor, conditions)
        conditions = {k: v for k, v in conditions.items() if k in ["data", "conditions"]}

        log_ml = self._log_marginal_likelihood(num_samples=num_samples, **conditions)
        log_ml = keras.ops.convert_to_numpy(log_ml)

        return np.mean(log_ml, axis=-1)

    def _log_marginal_likelihood(self, num_samples: int, data: Tensor = None, conditions: Tensor = None):
        _, _, data_summary = self._summary_metrics(data, stage="inference")

        batch_size = keras.ops.shape(data_summary)[0]

        # reshape everything into (batch_size * num_samples, ...)
        data = repeat_valid(data, num_samples)
        data_summary = repeat_valid(data_summary, num_samples)
        conditions = repeat_valid(conditions, num_samples)

        # sample parameters from posterior
        posterior_conditions = concatenate_valid((data_summary, conditions), axis=-1)
        parameters = self.posterior_network.sample(
            batch_shape=batch_size * num_samples, conditions=posterior_conditions
        )
        parameters = keras.ops.stop_gradient(parameters)

        # evaluate prior, likelihood, and posterior
        log_prior = self.prior_network.log_prob(samples=parameters, conditions=conditions)
        # gradient of the summary network should not be propagated here
        # because the summaries are used as 'fixed' input
        log_likelihood = self.likelihood_network.log_prob(
            samples=keras.ops.stop_gradient(data_summary) if self.likelihood_summary else data,
            conditions=concatenate_valid((parameters, conditions), axis=-1),
        )
        log_posterior = self.posterior_network.log_prob(samples=parameters, conditions=posterior_conditions)

        # stop gradients for networks that we do not want to train by the SC loss
        if "prior" not in self.sc_gradient:
            log_prior = keras.ops.stop_gradient(log_prior)
        if "likelihood" not in self.sc_gradient:
            log_likelihood = keras.ops.stop_gradient(log_likelihood)
        if "posterior" not in self.sc_gradient:
            log_posterior = keras.ops.stop_gradient(log_posterior)

        # compute log marginal likelihood using the inverse bayes theorem
        # also then reshape back so that for every batch we have num_samples estimates of the log ml
        log_ml = log_prior + log_likelihood - log_posterior
        log_ml = keras.ops.reshape(log_ml, newshape=(batch_size, num_samples))

        return log_ml

    def sample(
        self, *, num_samples: int, conditions: Mapping[str, np.ndarray], split: bool = False, **kwargs
    ) -> dict[str, np.ndarray]:
        """
        Generates samples from the posterior network given data and conditions.
        The `conditions` dictionary is preprocessed using the `adapter`.
        Samples are converted into NumPy arrays after inference.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        conditions : dict[str, np.ndarray]
            Dictionary of conditioning variables as NumPy arrays.
        split : bool, default=False
            Whether to split the output arrays along the last axis and return one column vector per target variable
            samples.
        **kwargs : dict
            Additional keyword arguments for the adapter and sampling process.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing generated samples with the same keys as `conditions`.
        """
        conditions = self.adapter(conditions, strict=False, stage="inference", **kwargs)

        for key in ["data", "conditions"]:
            if key in self.standardize and key in conditions:
                conditions[key] = self.standardize_layers[key](conditions[key])

        conditions = keras.tree.map_structure(keras.ops.convert_to_tensor, conditions)

        conditions = {k: v for k, v in conditions.items() if k in ["data", "conditions"]}

        samples = self._sample(num_samples=num_samples, **conditions, **kwargs)

        if "parameters" in self.standardize:
            samples = self.standardize_layers["parameters"](samples, forward=False)

        samples = {"parameters": samples}
        samples = keras.tree.map_structure(keras.ops.convert_to_numpy, samples)
        samples = self.adapter(samples, inverse=True, strict=False)

        if split:
            samples = split_arrays(samples)

        return samples

    def _sample(self, num_samples: int, data: Tensor = None, conditions: Tensor = None, **kwargs) -> Tensor:
        if self.summary_network is not None:
            if data is None:
                raise ValueError("Data are required when summary network is present")

            data = self.summary_network(data, **filter_kwargs(kwargs, self.summary_network.call))

        inference_conditions = concatenate_valid((data, conditions), axis=-1)

        if inference_conditions is not None:
            # conditions must always have shape (batch_size, ..., dims)
            batch_size = keras.ops.shape(inference_conditions)[0]
            inference_conditions = keras.ops.expand_dims(inference_conditions, axis=1)
            inference_conditions = keras.ops.broadcast_to(
                inference_conditions, (batch_size, num_samples, *keras.ops.shape(inference_conditions)[2:])
            )
            batch_shape = keras.ops.shape(inference_conditions)[:-1]
        else:
            batch_shape = keras.ops.shape(inference_conditions)[1:-1]

        return self.posterior_network.sample(
            batch_shape, conditions=inference_conditions, **filter_kwargs(kwargs, self.posterior_network.sample)
        )
