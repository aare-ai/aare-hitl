"""Tests for inference engine."""

import pytest
from unittest.mock import MagicMock, patch
import torch

from aare.core.inference import (
    get_device,
    ModelState,
    InferenceEngine,
    CompareEngine,
    get_inference_engine,
    get_compare_engine,
)


class TestGetDevice:
    """Tests for device selection."""

    def test_cpu_explicit(self):
        """Explicitly request CPU."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_auto_returns_valid_device(self):
        """Auto mode returns a valid device."""
        device = get_device("auto")
        assert device.type in ("cpu", "mps", "cuda")

    def test_mps_fallback_to_cpu(self):
        """MPS falls back to CPU when unavailable."""
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            device = get_device("mps")
            assert device.type == "cpu"

    def test_cuda_fallback_to_cpu(self):
        """CUDA falls back to CPU when unavailable."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            device = get_device("cuda")
            assert device.type == "cpu"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_when_available(self):
        """MPS is used when available."""
        device = get_device("mps")
        assert device.type == "mps"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_when_available(self):
        """CUDA is used when available."""
        device = get_device("cuda")
        assert device.type == "cuda"


class TestModelState:
    """Tests for ModelState dataclass."""

    def test_default_state(self):
        """Default state is unloaded."""
        state = ModelState()
        assert state.loaded is False
        assert state.model_path == ""
        assert state.adapter_path == ""
        assert state.model is None
        assert state.tokenizer is None
        assert state.device.type == "cpu"

    def test_custom_state(self):
        """Create state with custom values."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        device = torch.device("mps")

        state = ModelState(
            loaded=True,
            model_path="test/model",
            adapter_path="test/adapter",
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=device,
        )

        assert state.loaded is True
        assert state.model_path == "test/model"
        assert state.adapter_path == "test/adapter"
        assert state.model is mock_model
        assert state.tokenizer is mock_tokenizer
        assert state.device.type == "mps"


class TestInferenceEngine:
    """Tests for InferenceEngine."""

    def test_init_default_device(self):
        """Initialize with default device."""
        engine = InferenceEngine()
        assert engine.device.type in ("cpu", "mps", "cuda")
        assert engine.state.loaded is False

    def test_init_cpu_device(self):
        """Initialize with CPU device."""
        engine = InferenceEngine(device="cpu")
        assert engine.device.type == "cpu"

    def test_load_empty_path(self):
        """Load with empty path returns error."""
        engine = InferenceEngine(device="cpu")
        result = engine.load("")
        assert "Error" in result

    def test_generate_without_load(self):
        """Generate without loaded model returns error."""
        engine = InferenceEngine(device="cpu")
        result = engine.generate("test prompt")
        assert "Error" in result or "No model" in result

    def test_unload(self):
        """Unload resets state."""
        engine = InferenceEngine(device="cpu")
        engine.state = ModelState(
            loaded=True,
            model_path="test",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        engine.unload()

        assert engine.state.loaded is False
        assert engine.state.model is None
        assert engine.state.tokenizer is None

    def test_skip_reload_same_model(self):
        """Skip reload if same model already loaded."""
        engine = InferenceEngine(device="cpu")
        engine.state = ModelState(
            loaded=True,
            model_path="test/model",
            adapter_path="",
        )

        result = engine.load("test/model", "")
        assert "Already loaded" in result


class TestCompareEngine:
    """Tests for CompareEngine."""

    def test_init(self):
        """Initialize compare engine."""
        engine = CompareEngine(device="cpu")
        assert engine.device.type == "cpu"
        assert engine.base_model is None
        assert engine.ft_model is None
        assert engine.tokenizer is None
        assert engine._loaded is False

    def test_generate_base_not_loaded(self):
        """Generate from base when not loaded."""
        engine = CompareEngine(device="cpu")
        result = engine.generate_base("test")
        assert "not loaded" in result.lower()

    def test_generate_finetuned_no_adapter(self):
        """Generate from finetuned without adapter."""
        engine = CompareEngine(device="cpu")
        engine._loaded = True
        engine.ft_model = None
        result = engine.generate_finetuned("test")
        assert "No adapter" in result

    def test_unload(self):
        """Unload clears all models."""
        engine = CompareEngine(device="cpu")
        engine.base_model = MagicMock()
        engine.ft_model = MagicMock()
        engine.tokenizer = MagicMock()
        engine._loaded = True

        engine.unload()

        assert engine.base_model is None
        assert engine.ft_model is None
        assert engine.tokenizer is None
        assert engine._loaded is False


class TestGlobalEngines:
    """Tests for global engine instances."""

    def test_get_inference_engine_singleton(self):
        """get_inference_engine returns same instance."""
        # Reset global state
        import aare.core.inference as inference_module
        inference_module._engine = None

        engine1 = get_inference_engine("cpu")
        engine2 = get_inference_engine("cpu")

        assert engine1 is engine2

    def test_get_compare_engine_singleton(self):
        """get_compare_engine returns same instance."""
        # Reset global state
        import aare.core.inference as inference_module
        inference_module._compare_engine = None

        engine1 = get_compare_engine("cpu")
        engine2 = get_compare_engine("cpu")

        assert engine1 is engine2


class TestInferenceEngineWithMocks:
    """Tests for InferenceEngine with mocked transformers.

    Note: These tests use state manipulation rather than import mocking
    since the imports happen inside functions.
    """

    def test_generate_with_preloaded_state(self):
        """Generate with a pre-configured state (simulating loaded model)."""
        engine = InferenceEngine(device="cpu")

        # Manually set up a mock state
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Generated response"

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])

        engine.state = ModelState(
            loaded=True,
            model_path="test/model",
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
        )

        result = engine.generate("Hello")

        assert result == "Generated response"
        mock_model.generate.assert_called_once()

    def test_generate_empty_response(self):
        """Handle empty generated response."""
        engine = InferenceEngine(device="cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = ""

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

        engine.state = ModelState(
            loaded=True,
            model_path="test",
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
        )

        result = engine.generate("test")
        assert result == "(Empty response)"

    def test_generate_with_error(self):
        """Handle generation error gracefully."""
        engine = InferenceEngine(device="cpu")

        mock_model = MagicMock()
        mock_model.parameters.side_effect = Exception("Test error")

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        engine.state = ModelState(
            loaded=True,
            model_path="test",
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
        )

        result = engine.generate("test")
        assert "Error" in result
