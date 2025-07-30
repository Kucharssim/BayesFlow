from collections.abc import Sequence, Mapping

import keras
import numpy as np

from keras.optimizers.schedules import LearningRateSchedule  # type: ignore

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.networks.inference_network import InferenceNetwork
from bayesflow.networks.summary_network import SummaryNetwork
from bayesflow.distributions import Distribution
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
        evidence_network: InferenceNetwork | Distribution,
        prior_weights: Sequence[float] = None,
        summary_network: SummaryNetwork = None,
        compute_sc: bool = True,
        evidence_summary: bool = False,
        loss_schedules: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_models = num_models
        self.adapter = adapter

        self.posterior_network = posterior_network
        self.evidence_network = evidence_network

        if prior_weights is None:
            prior_weights = [-np.log(num_models) for _ in range(num_models)]

        if len(prior_weights) != num_models:
            raise ValueError("There must be exactly one prior weight for every model")

        self.log_prior = keras.ops.convert_to_tensor([prior_weights])

        self.summary_network = summary_network

        self.compute_sc = compute_sc
        self.evidence_summary = evidence_summary

        if loss_schedules is None:
            loss_schedules = dict()
        self.loss_schedules = loss_schedules

    def build(self, data_shapes: dict[str, tuple[int]] | dict[str, dict]) -> None:
        summary_outputs_shape = data_shapes["data"]
        if self.summary_network is not None:
            if not self.summary_network.built:
                self.summary_network.build(data_shapes["data"])
            summary_outputs_shape = self.summary_network.compute_output_shape(data_shapes["data"])

            if self.loss_schedules.get("summary_network") is None:
                self.loss_schedules["summary_network"] = 1.0

        posterior_network_conditions_shape = concatenate_valid_shapes(
            [summary_outputs_shape, data_shapes.get("conditions")], axis=-1
        )

        if not self.posterior_network.built:
            self.posterior_network.build(posterior_network_conditions_shape)

        evidence_network_conditions_shape = concatenate_valid_shapes(
            [data_shapes.get("model_indices"), data_shapes.get("conditions")], axis=-1
        )
        if not self.evidence_network.built:
            if self.evidence_summary:
                self.evidence_network.build(summary_outputs_shape, evidence_network_conditions_shape)
            else:
                self.evidence_network.build(data_shapes.get("data"), evidence_network_conditions_shape)

        # add fixed schedules if not defined
        if self.loss_schedules.get("evidence_network") is None:
            self.loss_schedules["evidence_network"] = 1.0
        if self.loss_schedules.get("posterior_network") is None:
            self.loss_schedules["posterior_network"] = 1.0
        if self.loss_schedules.get("self-consistency") is None:
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
            "log_prior": self.log_prior,
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

        # posterior
        metric, loss = self._posterior_metrics(model_indices, data_summary, conditions, stage=stage)
        metrics = metrics | metric
        total_loss += loss

        # likelihood
        metric, loss = self._evidence_metrics(
            model_indices,
            keras.ops.stop_gradient(data_summary) if self.evidence_summary else data,
            conditions,
            stage=stage,
        )
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

    def _posterior_metrics(
        self, model_indices: Tensor, data: Tensor, conditions: Tensor, stage: str
    ) -> tuple[dict, float]:
        if isinstance(self.posterior_network, Distribution):
            return {}, keras.ops.zeros(())

        logits = self.posterior_network(concatenate_valid((data, conditions), axis=-1), training=stage == "training")
        loss = keras.losses.categorical_crossentropy(model_indices, logits, from_logits=True)

        metrics = {"loss/posterior_loss": loss}

        if isinstance(self.loss_schedules["posterior_network"], LearningRateSchedule):
            lam = self.loss_schedules["posterior_network"](self.step)
            metrics = metrics | {"lambda/posterior": lam}
        else:
            lam = self.loss_schedules["posterior_network"]

        loss = lam * loss

        return metrics, loss

    def _evidence_metrics(
        self, model_indices: Tensor, data: Tensor, conditions: Tensor, stage: str
    ) -> tuple[dict, float]:
        if isinstance(self.evidence_network, InferenceNetwork):
            metrics = self.evidence_network.compute_metrics(
                data, concatenate_valid((model_indices, conditions), axis=-1), stage=stage
            )
            loss = metrics.get("loss", keras.ops.zeros(()))
            metrics = {f"{key}/evidence_{key}": value for key, value in metrics.items()}
        elif isinstance(self.evidence_network, Distribution):
            metrics = {}
            loss = keras.ops.zeros(())

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

        logit_posterior = self.posterior_network(concatenate_valid((data_summary, conditions), axis=-1))
        log_evidences = self._evidences(data_summary if self.evidence_summary else data, conditions)
        log_prior = keras.ops.cast(self.log_prior, dtype=keras.ops.dtype(logit_posterior))

        log_ml = log_prior + log_evidences - logit_posterior

        return log_ml

    def _evidences(self, data: Tensor, conditions: Tensor) -> Tensor:
        # for each data set, calculate evidence for each model
        batch_size = keras.ops.shape(data)[0]
        model_indices = keras.ops.eye(self.num_models)

        evidence_list = []

        for model_index in range(self.num_models):
            model_reps = model_indices[model_index : model_index + 1]  # (1, num_models)
            model_reps = repeat_valid(model_reps, batch_size)  # (batch_size, num_models)

            if isinstance(self.evidence_network, InferenceNetwork):
                evidence = self.evidence_network.log_prob(data, concatenate_valid((model_reps, conditions), axis=-1))
            elif isinstance(self.evidence_network, Distribution):
                evidence = self.evidence_network.log_prob(data, model_reps, conditions)
            else:
                raise ValueError(
                    "evidence network must be an instance of Inference network or an instance of a Distribution"
                )

            evidence_list.append(evidence)

        evidences = keras.ops.stack(evidence_list, axis=0)
        evidences = keras.ops.transpose(evidences, [1, 0])

        return evidences

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
            data = self.summary_network(data)

        logits = self.posterior_network(concatenate_valid((data, conditions), axis=-1), training=False)

        return logits

    def bayes_factors(
        self,
        *,
        conditions: Mapping[str, np.ndarray],
        log: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Predicts pairwise Bayes factors given input conditions. The `conditions` dictionary is preprocessed
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
            Predicted Bayes factors given `conditions`.
        """

        logit_posterior = self.predict(conditions=conditions, probs=False, **kwargs)
        log_prior = keras.ops.cast(self.log_prior, dtype=keras.ops.dtype(logit_posterior))

        evidences = logit_posterior - log_prior
        evidences = keras.ops.convert_to_numpy(evidences)

        log_bf = np.zeros((logit_posterior.shape[0], self.num_models, self.num_models))

        for evidence_for in range(self.num_models):
            for evidence_against in range(self.num_models):
                log_bf[:, evidence_for, evidence_against] = (
                    evidences[..., evidence_for] - evidences[..., evidence_against]
                )

        if log:
            return log_bf

        return np.exp(log_bf)
