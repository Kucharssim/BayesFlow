from collections.abc import Sequence, Mapping

import keras
import numpy as np

from keras.optimizers.schedules import LearningRateSchedule  # type: ignore

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.networks.inference_network import InferenceNetwork
from bayesflow.networks.summary_network import SummaryNetwork
from bayesflow.distributions import Distribution, Categorical
from bayesflow.types import Tensor
from bayesflow.utils import concatenate_valid_shapes, concatenate_valid, repeat_valid
from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("bayesflow.approximators")
class SelfConsistentModelComparison(Approximator):
    def __init__(
        self,
        num_models: int,
        adapter: Adapter,
        posterior_network: keras.Layer,
        evidence_network: InferenceNetwork | Distribution | Sequence[InferenceNetwork | Distribution],
        prior_network: keras.Layer | Sequence[float] = None,
        summary_network: SummaryNetwork = None,
        compute_sc: bool = True,
        loss_schedules: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_models = num_models
        self.adapter = adapter

        self.posterior_network = posterior_network
        self.evidence_network = evidence_network

        if prior_network is None:
            prior_weights = [1.0 for _ in range(num_models)]
            prior_network = Categorical(prob_weights=prior_weights)
        elif isinstance(prior_network, Sequence):
            prior_network = Categorical(prob_weights=prior_network)
        self.prior_network = prior_network

        self.summary_network = summary_network

        self.compute_sc = compute_sc

        if loss_schedules is None:
            loss_schedules = dict()
        self.loss_schedules = loss_schedules

    def build(self, data_shapes: dict[str, tuple[int]] | dict[str, dict]) -> None:
        summary_outputs_shape = data_shapes["data"]
        if self.summary_network is not None:
            if not self.summary_network.built:
                self.summary_network.build(data_shapes["data"])
            summary_outputs_shape = self.summary_network.compute_output_shape(data_shapes["data"])

            if not self.loss_schedules.get("summary_network"):
                self.loss_schedules["summary_network"] = 1.0

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

        # add fixed schedules if not defined
        if not self.loss_schedules.get("prior_network"):
            self.loss_schedules["prior_network"] = 1.0
        if not self.loss_schedules.get("evidence_network"):
            self.loss_schedules["evidence_network"] = 1.0
        if not self.loss_schedules.get("posterior_network"):
            self.loss_schedules["posterior_network"] = 1.0
        if not self.loss_schedules.get("self-consistency"):
            self.loss_schedules["self-consistency"] = 1.0

        self.step = keras.Variable(0, trainable=False)

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
            "compute_sc": self.compute_sc,
            "loss_schedules": self.loss_schedules,
        }

        return base_config | serialize(config)

    def compute_metrics(
        self,
        model_indices: Tensor,
        data: Tensor,
        conditions: Tensor = None,
        sc_data: Tensor = None,
        sc_conditions: Tensor = None,
        stage: str = "training",
    ):
        metrics, total_loss, data_summary = self._summary_metrics(data, stage)

        # prior
        metric, loss = self._prior_metrics(model_indices, conditions, stage=stage)
        metrics = metrics | metric
        total_loss += loss

        # posterior
        metric, loss = self._posterior_metrics(model_indices, data_summary, conditions, stage=stage)
        metrics = metrics | metric
        total_loss += loss

        # likelihood
        metric, loss = self._evidence_metrics(model_indices, data_summary, conditions, stage=stage)
        metrics = metrics | metric
        total_loss += loss

        # self-consistency
        if self.compute_sc:
            metric, loss = self._self_consistency_metrics(sc_data, sc_conditions)
            metrics = metrics | metric
            total_loss += loss

        metrics = {"loss": total_loss} | metrics

        # tick step counter (for sc_lambda schedule)
        if stage == "training":
            self.step.assign_add(1)

        return metrics

    def _summary_metrics(self, data: Tensor, stage: str) -> tuple[dict, float, Tensor]:
        if self.summary_network is None:
            return {}, keras.ops.zeros(()), data

        if data is None:
            raise ValueError("Sumary variables are required when summary network is present.")

        metrics = self.summary_network.compute_metrics(data, stage=stage)
        data = metrics.pop("outputs")

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

    def _prior_metrics(self, model_indices: Tensor, conditions: Tensor, stage: str) -> tuple[dict, float]:
        if isinstance(self.prior_network, Distribution):
            return {}, keras.ops.zeros(())
        logits = self.prior_network(conditions, training=stage == "training")

        # in case conditions are None, the prior_network might not know how to return output with batch_size shape
        if keras.ops.shape(logits)[0] == 1:
            logits = repeat_valid(logits, keras.ops.shape(model_indices)[0])
        loss = keras.losses.categorical_crossentropy(model_indices, logits, from_logits=True)

        metrics = {"loss/prior": loss}

        if isinstance(self.loss_schedules["prior_network"], LearningRateSchedule):
            lam = self.loss_schedules["prior_network"](self.step)
            metrics = metrics | {"lambda/prior": lam}
        else:
            lam = self.loss_schedules["prior_network"]

        loss = lam * loss

        return metrics, loss

    def _posterior_metrics(
        self, model_indices: Tensor, data: Tensor, conditions: Tensor, stage: str
    ) -> tuple[dict, float]:
        if isinstance(self.posterior_network, Distribution):
            return {}, keras.ops.zeros(())

        logits = self.posterior_network(concatenate_valid((data, conditions), axis=-1), training=stage == "training")
        loss = keras.losses.categorical_crossentropy(model_indices, logits, from_logits=True)

        metrics = {"loss/posterior": loss}

        if isinstance(self.loss_schedules["posterior_network"], LearningRateSchedule):
            lam = self.loss_schedules["posterior_network"](self.step)
            metrics = metrics | {"lambda/posterior": lam}
        else:
            lam = self.loss_schedules["posterior_network"]

        loss = lam * loss

        return metrics, loss

    def _evidence_metrics(
        self, model_indices: Tensor, data_summary: Tensor, conditions: Tensor, stage: str
    ) -> tuple[dict, float]:
        if isinstance(self.evidence_network, InferenceNetwork):
            metrics = self.evidence_network.compute_metrics(
                data_summary, concatenate_valid((model_indices, conditions), axis=-1), stage=stage
            )
            loss = metrics.get("loss", keras.ops.zeros(()))
            metrics = {f"{key}/evidence_{key}": value for key, value in metrics.items()}
        elif isinstance(self.evidence_network, Distribution):
            metrics = {}
            loss = keras.ops.zeros(())
        elif isinstance(self.evidence_network, Sequence):
            metrics = {}
            loss = keras.ops.zeros(())

            for model_id in range(self.num_models):
                # Select rows where model_id is active (model_indices[:, model_id] == 1)
                mask = keras.ops.equal(model_indices[:, model_id], 1.0)
                mask = keras.ops.cast(mask, "bool")

                # Get indices for current model
                indices = keras.ops.where(mask)[0]

                # Gather matching rows
                subset_model_indices = keras.ops.take(model_indices, indices, axis=0)
                subset_data_summary = keras.ops.take(data_summary, indices, axis=0)
                if conditions:
                    subset_conditions = keras.ops.take(conditions, indices, axis=0)
                else:
                    subset_conditions = None

                # Compute metrics for this model's evidence network
                evidence_network = self.evidence_network[model_id]

                if isinstance(evidence_network, InferenceNetwork):
                    sub_metrics = evidence_network.compute_metrics(
                        subset_data_summary,
                        concatenate_valid((subset_model_indices, subset_conditions), axis=-1),
                        stage=stage,
                    )
                    loss += sub_metrics.get("loss", keras.ops.zeros(()))
                else:
                    sub_metrics = {}

                for key, value in sub_metrics.items():
                    metrics[f"{key}/evidence_{key}/model_{model_id}"] = value

        if isinstance(self.loss_schedules["evidence_network"], LearningRateSchedule):
            lam = self.loss_schedules["evidence_network"](self.step)
            metrics = metrics | {"lambda/evidence": lam}
        else:
            lam = self.loss_schedules["evidence_network"]

        loss = lam * loss

        return metrics, loss

    def _self_consistency_metrics(self, data: Tensor, conditions: Tensor) -> tuple[dict, float]:
        log_ml = self._log_marginal_likelihood(data, conditions)
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

    def _log_marginal_likelihood(self, data: Tensor, conditions: Tensor) -> Tensor:
        _, _, data_summary = self._summary_metrics(data, stage="inference")
        data_summary = keras.ops.stop_gradient(data_summary)

        logit_prior = self.prior_network(conditions)
        logit_posterior = self.posterior_network(concatenate_valid((data_summary, conditions), axis=-1))
        log_evidences = self._evidences(data, data_summary, conditions)

        log_ml = logit_prior + log_evidences - logit_posterior

        return log_ml

    def _evidences(self, data: Tensor, data_summary: Tensor, conditions: Tensor) -> Tensor:
        batch_size = keras.ops.shape(data)[0]
        model_indices = keras.ops.eye(self.num_models)

        evidence_list = []

        for model_index in range(self.num_models):
            if isinstance(self.evidence_network, Sequence):
                net = self.evidence_network[model_index]
            else:
                net = self.evidence_network

            model_reps = model_indices[model_index : model_index + 1]  # (1, num_models)
            model_reps = repeat_valid(model_reps, batch_size)  # (batch_size, num_models)

            evidence = self._evidence(net, model_reps, data, data_summary, conditions)
            evidence_list.append(evidence)

        evidences = keras.ops.stack(evidence_list, axis=0)
        evidences = keras.ops.transpose(evidences, [1, 0])

        return evidences

    def _evidence(
        self,
        evidence_network: keras.Layer,
        model_indices: Tensor,
        data: Tensor,
        data_summary: Tensor,
        conditions: Tensor,
    ):
        if isinstance(evidence_network, InferenceNetwork):
            return evidence_network.log_prob(data_summary, concatenate_valid((model_indices, conditions), axis=-1))
        elif isinstance(evidence_network, Distribution):
            return evidence_network.log_prob(data, model_indices, conditions)
        else:
            raise ValueError(
                "evidence network must be an instance of Inference network or an instance of a Distribution"
            )

    def _batch_size_from_data(self, data: Mapping[str, any]) -> int:
        """
        Fetches the current batch size from an input dictionary. Can only be used during training when
        model indices as present.
        """
        return keras.ops.shape(data["model_indices"])[0]

    def predict(
        self,
        *,
        conditions: Mapping[str, np.ndarray],
        probs: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Predicts posterior model probabilities given input conditions. The `conditions` dictionary is preprocessed
        using the `adapter`. The output is converted to NumPy array after inference.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            Dictionary of conditioning variables as NumPy arrays.
        probs: bool, optional
            A flag indicating whether model probabilities (True) or logits (False) are returned. Default is True.
        **kwargs : dict
            Additional keyword arguments for the adapter and classifier.

        Returns
        -------
        outputs: np.ndarray
            Predicted posterior model probabilities given `conditions`.
        """

        # Apply adapter transforms to raw simulated / real quantities
        conditions = self.adapter(conditions, strict=False, **kwargs)
        conditions = keras.tree.map_structure(keras.ops.convert_to_tensor, conditions)

        output = self._predict(**conditions, **kwargs)

        if probs:
            output = keras.ops.softmax(output)

        return keras.ops.convert_to_numpy(output)

    def _predict(self, data, conditions=None, **kwargs) -> Tensor:
        if self.summary_network:
            data = self.summary_network(data, stage="inference")

        logits = self.posterior_network(concatenate_valid((data, conditions), axis=-1), training=False)

        return logits
