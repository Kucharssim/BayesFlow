from collections.abc import Sequence

import keras
import bayesflow as bf
from bayesflow.networks.inference_network import InferenceNetwork
from bayesflow.networks.summary_network import SummaryNetwork
from bayesflow.distributions.distribution import Distribution
from bayesflow.types import Tensor
from bayesflow.utils import concatenate_valid_shapes, concatenate_valid, repeat_valid
from bayesflow.utils.serialization import serializable, serialize, deserialize


class PriorModelProbabilities(keras.Layer):
    def __init__(self, *, prior_weights, **kwargs):
        super().__init__(**kwargs)
        self.prior_weights = prior_weights
        self.prior_probs = [p / sum(prior_weights) for p in prior_weights]
        self.prior_probs = keras.ops.convert_to_tensor([self.prior_probs])

    def __call__(self, conditions: Tensor, *args, **kwargs):
        batch_size = keras.ops.shape(conditions)[0]

        return keras.ops.repeat(self.prior_probs, repeats=batch_size, axis=0)


@serializable("bayesflow.approximators")
class SelfConsistentModelComparison(bf.approximators.Approximator):
    def __init__(
        self,
        num_models: int,
        adapter: bf.Adapter,
        posterior_network: keras.Layer,
        evidence_network: InferenceNetwork | Distribution | Sequence[InferenceNetwork | Distribution],
        prior_network: keras.Layer | Sequence[float] = None,
        summary_network: SummaryNetwork = None,
        loss_schedules: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_models = num_models
        self.adapter = adapter

        self.posterior_network = posterior_network
        self.evidence_network = evidence_network

        if isinstance(prior_network, Sequence):
            prior_network = PriorModelProbabilities(prior_weights=prior_network)
        self.prior_network = prior_network

        self.summary_network = summary_network

        if loss_schedules is None:
            loss_schedules = dict()
        self.loss_schedules = loss_schedules

    def build(self, data_shapes: dict[str, tuple[int]] | dict[str, dict]) -> None:
        summary_outputs_shape = data_shapes["data"]
        if self.summary_network is not None:
            if not self.summary_network.built:
                self.summary_network.build(data_shapes["data"])
            summary_outputs_shape = self.summary_network.compute_output_shape(data_shapes["data"])

        posterior_network_conditions_shape = concatenate_valid_shapes(
            [summary_outputs_shape, data_shapes.get("conditions")], axis=-1
        )

        if not self.posterior_network.built:
            self.posterior_network.build(posterior_network_conditions_shape)

        if not self.prior_network.built:
            self.prior_network.build(data_shapes.get("conditions"))

        evidence_network_conditions_shape = concatenate_valid_shapes(
            [data_shapes.get("model_indices"), data_shapes.get("conditions")], axis=-1
        )
        if isinstance(self.evidence_network, Sequence):
            for net in self.evidence_network:
                if not net.built:
                    net.build(summary_outputs_shape, evidence_network_conditions_shape)
        else:
            if not self.evidence_network.built:
                self.evidence_network.build(summary_outputs_shape, evidence_network_conditions_shape)

        self.built = True

    def build_from_data(self, adapted_data: dict[str, any]):
        self.build(keras.tree.map_structure(keras.ops.shape(adapted_data)))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()

        config = {
            "num_models": self.num_models,
            "adapter": self.adapter,
            "posterior_network": self.posterior_network,
            "evidence_network": self.evidence_network,
            "prior_network": self.prior_network,
            "summary_network": self.summary_network,
            "loss_schedules": self.loss_schedules,
        }

        return base_config | serialize(config)

    def compute_metrics(
        self,
        model_indices: Tensor,
        data: Tensor,
        conditions: Tensor,
        sc_data: Tensor,
        sc_conditions: Tensor,
        stage: str = "training",
    ):
        metrics, total_loss, data_summary = self._summary_metricsy(data, stage)

        # prior
        metric, loss = self._prior_metrics(model_indices, conditions, stage=stage)
        metrics = metrics | metric
        total_loss += loss

        # posterior
        metric, loss = self._posterior_metrics(model_indices, data_summary, conditions, stage=stage)
        metrics = metrics | metric
        total_loss += loss

        # likelihood
        metric, loss = self._evidence_metrics(model_indices, data, data_summary, conditions, stage=stage)
        metrics = metrics | metric
        total_loss += loss

        # self-consistency
        metric, loss = self._self_consistency_metrics(sc_data, sc_conditions)
        metrics = metrics | metric
        total_loss += loss

        metrics = {"loss": total_loss} | metrics

        return metrics

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

        # weight the loss by schedule
        # if isinstance(self.loss_schedules["summary_network"], LearningRateSchedule):
        #     lam = self.loss_schedules["summary_network"](self.step)
        #     metrics = metrics | {"lambda/summary": lam}
        # else:
        #     lam = self.loss_schedules["summary_network"]

        # loss = lam * loss

        return metrics, loss, data

    def _prior_metrics(self, model_indices: Tensor, conditions: Tensor, stage: str) -> tuple[dict, float]:
        logits = self.prior_network(conditions, stage=stage)
        loss = keras.losses.categorical_crossentropy(model_indices, logits, from_logits=True)

        metrics = {"loss/prior": loss}

        return metrics, loss

    def _posterior_metrics(
        self, model_indices: Tensor, data: Tensor, conditions: Tensor, stage: str
    ) -> tuple[dict, float]:
        logits = self.posterior_network(concatenate_valid((data, conditions), axis=-1))
        loss = keras.losses.categorical_crossentropy(model_indices, logits, from_logits=True)

        metrics = {"loss/posterior": loss}

        return metrics, loss

    def _evidence_metrics(
        self, model_indices: Tensor, data: Tensor, data_summary: Tensor, conditions: Tensor, stage: str
    ) -> tuple[dict, float]:
        pass

    def _self_consistency_metrics(self, data: Tensor, conditions: Tensor) -> tuple[dict, float]:
        log_ml = self._log_marginal_likelihood(data, conditions)
        loss = keras.ops.var(log_ml, axis=-1)
        loss = keras.ops.mean(loss)

        metrics = {"loss/self-consistency_loss": loss}

        return metrics, loss

    def _log_marginal_likelihood(self, data: Tensor, conditions: Tensor) -> Tensor:
        _, _, data_summary = self._summary_metrics(data, stage="inference")
        data_summary = keras.ops.stop_gradient(data_summary)

        batch_size = keras.ops.shape(data_summary)[0]

        model_indices = keras.ops.eye(self.num_models)
        model_indices = repeat_valid(model_indices, batch_size)

        data = repeat_valid(data, self.num_models)
        data_summary = repeat_valid(data, self.num_models)
        conditions = repeat_valid(data, self.num_models)

        logit_prior = self.prior_network(conditions)

        logit_posterior = self.posterior_network(concatenate_valid((data_summary, conditions), axis=-1))

        log_evidences = self._evidences(model_indices, data, data_summary, conditions)

        log_ml = logit_prior + log_evidences - logit_posterior
        log_ml = keras.ops.reshape(log_ml, newshape=(batch_size, self.num_models))

        return log_ml

    def _evidences(self, model_indices: Tensor, data: Tensor, data_summary: Tensor, conditions: Tensor) -> Tensor:
        if isinstance(self.evidence_network, InferenceNetwork):
            pass
        elif isinstance(self.evidence_network, Distribution):
            pass
        elif isinstance(self.evidence_network, Sequence):
            for net in self.evidence_network:
                pass
        else:
            raise ValueError(
                "evidence network must be an inference network, a distribution, "
                "or a sequence of inference networks or distributions."
            )
