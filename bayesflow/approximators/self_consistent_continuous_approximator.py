from collections.abc import Sequence

import keras
from bayesflow.types import Tensor
from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.networks.inference_network import InferenceNetwork
from bayesflow.networks.summary_network import SummaryNetwork
from bayesflow.distributions.distribution import Distribution
from bayesflow.utils.logging import concatenate_valid_shapes, concatenate_valid, repeat_valid
from bayesflow.networks.standardization import Standardization
from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("bayesflow.approximators")
class SelfConsistentContinuousApproximator(Approximator):
    def __init__(
        self,
        adapter: Adapter,
        prior_network: InferenceNetwork | Distribution,
        likelihood_network: InferenceNetwork | Distribution,
        posterior_network: InferenceNetwork | Distribution,
        summary_network: SummaryNetwork = None,
        standardize: str | Sequence[str] | None = None,
        num_sc_samples: int = 64,
        likelihood_summary: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.adapter = adapter
        self.prior_network = prior_network
        self.likelihood_network = likelihood_network
        self.posterior_network = posterior_network
        self.summary_network = summary_network
        self.num_sc_samples = num_sc_samples
        if likelihood_summary and summary_network is None:
            raise ValueError("Summary network needs to be defined for computing the summary likelihood.")
        self.likehood_summary = likelihood_summary

        if isinstance(standardize, str) and standardize != "all":
            self.standardize = [standardize]
        else:
            self.standardize = standardize or []

        if self.standardize == "all":
            self.standardize_layers = None
        else:
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        data_summary_shape = None
        if self.summary_network is not None:
            if not self.summary_network.built:
                self.summary_network.build(data_shapes["data"])
            data_summary_shape = self.summary_network.compute_output_shape(data_shapes["data"])

        if isinstance(self.prior_network, InferenceNetwork) and not self.prior_network.built:
            self.prior_network.build(data_shapes["parameters"], data_shapes["conditions"])

        if isinstance(self.likelihood_network, InferenceNetwork) and not self.likelihood_network.built:
            if self.likehood_summary:
                self.likelihood_network.build(data_summary_shape, data_shapes["conditions"])
            else:
                self.likelihood_network.build(data_shapes["data"], data_shapes["conditions"])

        if isinstance(self.posterior_network, InferenceNetwork) and not self.posterior_network.built:
            posterior_conditions_shape = concatenate_valid_shapes(data_summary_shape, data_shapes["conditions"])
            self.posterior_network(data_shapes["parameters"], posterior_conditions_shape)

        if self.standardize == "all":
            self.standardize = [var for var in ["parameters", "data", "conditions"] if var in data_shapes]
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

        for var, layer in self.standardize_layers.items():
            layer.build(data_shapes[var])

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
            "standardize": self.standardize,
            "num_sc_samples": self.num_sc_samples,
            "likelihood_summary": self.likehood_summary,
        }

        return base_config | serialize(config)

    def compute_metrics(
        self,
        parameters: Tensor,
        data: Tensor = None,
        conditions: Tensor = None,
        sc_data: Tensor = None,
        sc_conditions: Tensor = None,
        stage: str = "training",
        **kwargs,
    ):
        # standardize inputs before doing any computations
        if "parameters" in self.standardize:
            if parameters is not None:
                parameters = self.standardize_layers["parameters"](data, stage=stage)
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
        metric, loss = self._likelihood_metrics(
            data_summary if self.likelihood_summary else data, parameters, conditions, stage
        )
        metrics = metrics | metric
        total_loss += loss

        # self-consistency
        metric, loss = self._self_consistency_metrics(sc_data, sc_conditions)
        metrics = metrics | metric
        total_loss += loss

        metrics = {"loss": total_loss} | metrics

        return metrics

    def _prior_metrics(self, parameters: Tensor, conditions: Tensor, stage: str) -> tuple[dict, float]:
        if not isinstance(self.prior_network, InferenceNetwork):
            return {}, keras.ops.zeros(())

        metrics = self.prior_network.compute_metrics(parameters, conditions=conditions, stage=stage)
        loss = metrics.get("loss", keras.ops.zeros(()))
        metrics = {f"{key}/prior_{key}": value for key, value in metrics.items()}

        return metrics, loss

    def _posterior_metrics(
        self, parameters: Tensor, data: Tensor, conditions: Tensor, stage: str
    ) -> tuple[dict, float]:
        if not isinstance(self.posterior_network, InferenceNetwork):
            return {}, keras.ops.zeros(())

        metrics = self.posterior_network.compute_metrics(
            parameters, conditions=concatenate_valid((data, conditions), axis=-1), stage=stage
        )
        loss = metrics.get("loss", keras.ops.zeros(()))
        metrics = {f"{key}/posterior_{key}": value for key, value in metrics.items()}

        return metrics, loss

    def _likelihood_metrics(
        self, data: Tensor, parameters: Tensor, conditions: Tensor, stage: str
    ) -> tuple[dict, float]:
        if not isinstance(self.likelihood_network, InferenceNetwork):
            return {}, keras.ops.zeros(())

        metrics = self.likelihood_network.compute_metrics(
            data, conditions=concatenate_valid((parameters, conditions), axis=-1), stage=stage
        )
        loss = metrics.get("loss", keras.ops.zeros(()))
        metrics = {f"{key}/likelihood_{key}": value for key, value in metrics.items()}

        return metrics, loss

    def _summary_metrics(self, data: Tensor, stage: str) -> tuple[dict, float, Tensor]:
        if self.summary_network is None:
            metrics = {}
        else:
            if data is None:
                raise ValueError("Sumary variables are required when summary network is present.")

            metrics = self.summary_network.compute_metrics(data, stage=stage)
            data = metrics.pop("outputs")

        loss = metrics.get("loss", keras.ops.zeros(()))
        metrics = {f"{key}/summary_{key}": value for key, value in metrics.items()}

        return metrics, loss, data

    def _self_consistency_metrics(self, data: Tensor, conditions: Tensor) -> tuple[dict, float]:
        if self.num_sc_samples == 0:
            return {}, keras.ops.zeros(())
        log_ml = self._log_marginal_likelihood(num_samples=self.num_sc_samples, data=data, conditions=conditions)
        loss = keras.ops.var(log_ml, axis=-1)
        loss = keras.ops.mean(loss)

        metrics = {"loss/self-consistency_loss": loss}

        return metrics, loss

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
        log_likelihood = self.likelihood_network.log_prob(
            samples=data_summary if self.likehood_summary else data,
            conditions=concatenate_valid((parameters, conditions), axis=-1),
        )
        log_posterior = self.posterior_network.log_prob(samples=parameters, conditions=posterior_conditions)

        # compute log marginal likelihood using the inverse bayes theorem
        # also then reshape back so that for every batch we have num_samples estimates of the log ml
        log_ml = log_prior + log_likelihood - log_posterior
        log_ml = keras.ops.reshape(log_ml, newshape=(batch_size, num_samples))

        return log_ml
