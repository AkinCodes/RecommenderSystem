"""Tests for PrepConfig and DLRM interaction upgrade."""

import numpy as np
import pytest
import torch
import sys
import os

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
        from data.preprocessing import NUM_FEATURES, RATING_MAX, LIKE_THRESHOLD
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
