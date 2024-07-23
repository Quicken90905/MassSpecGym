import typing as T
from abc import ABC

import torch
from torchmetrics import CosineSimilarity, MeanMetric
from torchmetrics.retrieval import RetrievalHitRate

from massspecgym.models.base import MassSpecGymModel, Stage
import massspecgym.utils as utils


class RetrievalMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(
        self,
        at_ks: T.Iterable[int] = (1, 5, 20), 
        myopic_mces_kwargs: T.Optional[T.Mapping] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.at_ks = at_ks
        self.myopic_mces = utils.MyopicMCES(**(myopic_mces_kwargs or {}))

    def on_batch_end(
        self, outputs: T.Any, batch: dict, batch_idx: int, stage: Stage
    ) -> None:
        """
        Compute evaluation metrics for the retrieval model based on the batch and corresponding
        predictions.
        """
        self.log(
            f"{stage.to_pref()}loss",
            outputs['loss'],
            batch_size=batch['spec'].size(0) if 'spec' in batch else batch['spec_tree'].size(0),
            # CHANGED FROM '''batch_size=batch['spec'].size(0)''' TO SUPPORT CURRENT IMPLEMENTATION OF MSnDataset---------------------------------
            sync_dist=True,
            prog_bar=True,
        )
        if stage in self.log_only_loss_at_stages:
            return

        self.evaluate_retrieval_step(
            outputs["scores"],
            #batch["labels"], ---------------------------------------UNCOMMENT THIS---------------------------------------------------------------
            #batch["batch_ptr"], ------------------------------------UNCOMMENT THIS---------------------------------------------------------------
            stage=stage,
        )
        self.evaluate_mces_at_1(
            outputs["scores"],
            batch["labels"],
            batch["smiles"],
            batch["candidates_smiles"],
            batch["batch_ptr"],
            stage=stage,
        )

    def evaluate_retrieval_step(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        batch_ptr: torch.Tensor,
        stage: Stage,
    ) -> None:
        """
        Main evaluation method for the retrieval models. The retrieval step is evaluated by 
        computing the hit rate at different top-k values.

        Args:
            scores (torch.Tensor): Concatenated scores for all candidates for all samples in the 
                batch
            labels (torch.Tensor): Concatenated True/False labels for all candidates for all samples
                 in the batch
            batch_ptr (torch.Tensor): Number of each sample's candidates in the concatenated tensors
        """
        # Evaluate hitrate at different top-k values
        indexes = torch.arange(batch_ptr.size(0), device=batch_ptr.device)
        indexes = torch.repeat_interleave(indexes, batch_ptr)
        for at_k in self.at_ks:
            self._update_metric(
                stage.to_pref() + f"hit_rate@{at_k}",
                RetrievalHitRate,
                (scores, labels, indexes),
                batch_size=batch_ptr.size(0),
                metric_kwargs=dict(top_k=at_k),
            )

    def evaluate_mces_at_1(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        smiles: list[str],
        candidates_smiles: list[str],
        batch_ptr: torch.Tensor,
        stage: Stage,
    ) -> None:
        """
        TODO
        """
        if labels.sum() != len(smiles):
            raise ValueError("MCES@1 evaluation currently supports exactly 1 positive candidate per sample.")
        
        # Get top-1 predicted molecules for each ground-truth sample
        smiles_pred_top_1 = []
        batch_ptr = torch.cumsum(batch_ptr, dim=0)
        for i, j in zip(torch.cat([torch.tensor([0], device=batch_ptr.device), batch_ptr]), batch_ptr):
            scores_sample = scores[i:j]
            top_1_idx = i + torch.argmax(scores_sample)
            smiles_pred_top_1.append(candidates_smiles[top_1_idx])

        # Calculate MCES distance between top-1 predicted molecules and ground truth
        mces_dists = [
            self.myopic_mces(sm, sm_pred)
            for sm, sm_pred in zip(smiles, smiles_pred_top_1)
        ]
        self._update_metric(
            f"{stage.to_pref()}mces_at_1",
            MeanMetric,
            (mces_dists,),
            batch_size=len(mces_dists)
        )


    def evaluate_fingerprint_step(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        stage: Stage,
    ) -> None:
        """
        Utility evaluation method to assess the quality of predicted fingerprints. This method is
        not a part of the necessary evaluation logic (not called in the `on_batch_end` method)
        since retrieval models are not bound to predict fingerprints.

        Args:
            y_true (torch.Tensor): [batch_size, fingerprint_size] tensor of true fingerprints
            y_pred (torch.Tensor): [batch_size, fingerprint_size] tensor of predicted fingerprints
        """
        # Cosine similarity between predicted and true fingerprints
        self._update_metric(
            f"{stage.to_pref()}fingerprint_cos_sim",
            CosineSimilarity,
            (y_pred, y_true),
            batch_size=y_true.size(0),
            metric_kwargs=dict(reduction="mean"),
        )
