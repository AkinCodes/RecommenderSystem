"""Tests for PrepConfig and DLRM interaction upgrade."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestPrepConfig:
    def test_default_values(self):
        from data.preprocessing import PrepConfig
        config = PrepConfig()
        assert config.num_features == 8
        assert config.rating_max == 5.0
        assert config.like_threshold == 4
        assert config.min_ratings == 1000
        assert config.min_interactions_warn == 5
        assert config.max_rating_var == 4.0
        assert config.train_ratio == 0.8

    def test_custom_values(self):
        from data.preprocessing import PrepConfig
        config = PrepConfig(min_ratings=500, train_ratio=0.9)
        assert config.min_ratings == 500
        assert config.train_ratio == 0.9
        assert config.num_features == 8  # unchanged defaults

    def test_validate_uses_config(self):
        from data.preprocessing import PrepConfig, validate_raw_data
        raw = np.array([[1, 1, 5, 100], [2, 2, 4, 200]], dtype=np.int64)
        # Default min_ratings=1000 should fail with only 2 rows
        with pytest.raises(ValueError, match="need at least 1000"):
            validate_raw_data(raw)
        # Custom config with min_ratings=1 should pass
        config = PrepConfig(min_ratings=1)
        validate_raw_data(raw, config=config)  # should not raise

    def test_backward_compat_constants(self):
        from data.preprocessing import LIKE_THRESHOLD, NUM_FEATURES, RATING_MAX
        assert NUM_FEATURES == 8
        assert RATING_MAX == 5.0
        assert LIKE_THRESHOLD == 4


class TestDLRMInteraction:
    def test_interaction_changes_output_shape(self):
        from models.dlrm import DLRMModel
        model = DLRMModel(
            num_features=8,
            embedding_sizes=[100, 200],
            mlp_layers=[128, 64, 32],
            dropout=0.0,
        )
        cont = torch.randn(4, 8)
        cat = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long)
        out = model(cont, cat)
        assert out.shape == (4, 1)

    def test_interaction_improves_gradient_flow(self):
        from models.dlrm import DLRMModel
        model = DLRMModel(
            num_features=8,
            embedding_sizes=[100, 200],
            mlp_layers=[128, 64, 32],
        )
        cont = torch.randn(4, 8)
        cat = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long)
        out = model(cont, cat)
        loss = out.sum()
        loss.backward()
        # Both embeddings should have gradients
        assert model.embeddings[0].weight.grad is not None
        assert model.embeddings[1].weight.grad is not None

    def test_output_range_zero_to_one(self):
        from models.dlrm import DLRMModel
        model = DLRMModel(
            num_features=8,
            embedding_sizes=[100, 200],
            mlp_layers=[128, 64, 32],
        )
        cont = torch.randn(16, 8)
        cat = torch.randint(0, 50, (16, 2))
        with torch.no_grad():
            out = model(cont, cat)
        assert (out >= 0).all() and (out <= 1).all()

    def test_model_parameter_count_increased(self):
        from models.dlrm import DLRMModel
        # With interaction, MLP input is larger so more params
        model = DLRMModel(
            num_features=8,
            embedding_sizes=[100, 200],
            mlp_layers=[128, 64, 32],
        )
        total_params = sum(p.numel() for p in model.parameters())
        # Original was: continuous(128) + embeds + MLP(128*64 + 64*32) + output
        # Now MLP first layer is wider by 128 (interaction_size)
        # MLP input = 128 + 128 + 128 + 128 = 512 (was 384)
        assert total_params > 70_000

    def test_interaction_changes_output_value(self):
        from models.dlrm import DLRMModel
        torch.manual_seed(42)
        model = DLRMModel(
            num_features=8,
            embedding_sizes=[100, 200],
            mlp_layers=[128, 64, 32],
            dropout=0.0,
        )
        cont = torch.randn(4, 8)
        cat = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long)
        with torch.no_grad():
            out_normal = model(cont, cat).clone()
            # Zero out one embedding to remove interaction signal
            model.embeddings[1].weight.data.zero_()
            out_zeroed = model(cont, cat)
        assert not torch.allclose(out_normal, out_zeroed), \
            "Interaction term should change output when embeddings differ"

    def test_single_embedding_no_interaction(self):
        from models.dlrm import DLRMModel
        model = DLRMModel(
            num_features=8,
            embedding_sizes=[100],
            mlp_layers=[128, 64, 32],
            dropout=0.0,
        )
        cont = torch.randn(4, 8)
        cat = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)
        out = model(cont, cat)
        assert out.shape == (4, 1)


class TestBuildUserItemFeatures:
    def test_output_shape(self):
        from api.app import _build_user_item_features
        user_feats = [0.6, 0.15, 0.08, 0.34]
        item_features = {0: [0.7, 0.5, 0.8], 1: [0.4, 0.2, 0.3]}
        result = _build_user_item_features(user_feats, [0, 1], item_features, 8)
        assert result.shape == (2, 8)
        assert result.dtype == torch.float32

    def test_user_features_broadcast(self):
        from api.app import _build_user_item_features
        user_feats = [0.6, 0.15, 0.08, 0.34]
        item_features = {0: [0.5, 0.3, 0.7], 1: [0.5, 0.3, 0.7]}
        result = _build_user_item_features(user_feats, [0, 1], item_features, 8)
        assert result[0, 0].item() == pytest.approx(0.6)
        assert result[1, 0].item() == pytest.approx(0.6)
        assert result[0, 3].item() == pytest.approx(0.34)
        assert result[1, 3].item() == pytest.approx(0.34)

    def test_item_features_per_row(self):
        from api.app import _build_user_item_features
        user_feats = [0.5, 0.5, 0.5, 0.5]
        item_features = {0: [0.9, 0.1, 0.2], 1: [0.3, 0.8, 0.6]}
        result = _build_user_item_features(user_feats, [0, 1], item_features, 8)
        assert result[0, 4].item() == pytest.approx(0.9)
        assert result[1, 4].item() == pytest.approx(0.3)
        assert result[0, 6].item() == pytest.approx(0.2)
        assert result[1, 6].item() == pytest.approx(0.6)

    def test_interaction_feature(self):
        from api.app import _build_user_item_features
        user_feats = [0.8, 0.5, 0.5, 0.5]
        item_features = {0: [0.6, 0.5, 0.5], 1: [0.2, 0.5, 0.5]}
        result = _build_user_item_features(user_feats, [0, 1], item_features, 8)
        expected_0 = (0.8 - 0.6 + 1.0) / 2.0
        expected_1 = (0.8 - 0.2 + 1.0) / 2.0
        assert result[0, 7].item() == pytest.approx(expected_0)
        assert result[1, 7].item() == pytest.approx(expected_1)

    def test_missing_item_uses_default(self):
        from api.app import _build_user_item_features
        user_feats = [0.5, 0.5, 0.5, 0.5]
        item_features = {}
        result = _build_user_item_features(user_feats, [999], item_features, 8)
        assert result[0, 4].item() == pytest.approx(0.6)
        assert result[0, 5].item() == pytest.approx(0.0)
        assert result[0, 6].item() == pytest.approx(0.0)

    def test_empty_items(self):
        from api.app import _build_user_item_features
        user_feats = [0.5, 0.5, 0.5, 0.5]
        result = _build_user_item_features(user_feats, [], {}, 8)
        assert result.shape == (0, 8)
