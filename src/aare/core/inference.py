"""Inference module for model generation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

logger = logging.getLogger(__name__)


def get_device(preferred: str = "auto") -> torch.device:
    """Get the device to use for inference.

    Args:
        preferred: Device preference - "auto", "cpu", "mps", or "cuda"

    Returns:
        torch.device for the selected device
    """
    if preferred == "cpu":
        return torch.device("cpu")
    elif preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS not available, falling back to CPU")
        return torch.device("cpu")
    elif preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    else:  # auto
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


@dataclass
class ModelState:
    """State for a loaded model."""
    loaded: bool = False
    model_path: str = ""
    adapter_path: str = ""
    model: Any = None
    tokenizer: Any = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


class CompareEngine:
    """Engine for comparing base and fine-tuned models.

    Keeps both models loaded in memory for fast switching.
    """

    def __init__(self, device: str = "auto"):
        self.base_model: Any = None
        self.ft_model: Any = None
        self.tokenizer: Any = None
        self.device: torch.device = get_device(device)
        self.model_path: str = ""
        self.adapter_path: str = ""
        self._loaded = False

    def load_models(self, model_path: str, adapter_path: str = "") -> str:
        """Load both base and fine-tuned models for comparison.

        Args:
            model_path: HuggingFace model ID or local path.
            adapter_path: Path to LoRA adapter.

        Returns:
            Status message.
        """
        if self._loaded and self.model_path == model_path and self.adapter_path == adapter_path:
            return "Models already loaded"

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            device_name = str(self.device)
            logger.info(f"Using device: {device_name}")
            print(f"  Device: {device_name}")

            # Load tokenizer
            logger.info(f"Loading tokenizer: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            logger.info(f"Loading base model: {model_path}")
            print(f"  Loading base model...")

            # Use float16 on MPS/CUDA for speed, float32 on CPU
            dtype = torch.float16 if self.device.type in ("mps", "cuda") else torch.float32

            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            ).to(dtype).to(self.device)
            self.base_model.eval()

            # Load fine-tuned model (base + adapter)
            adapter_exists = adapter_path and Path(adapter_path).exists()
            if adapter_exists and (Path(adapter_path) / "adapter_config.json").exists():
                logger.info(f"Loading fine-tuned model with adapter: {adapter_path}")
                print(f"  Loading fine-tuned model...")

                # Load a fresh copy for fine-tuned
                ft_base = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                ).to(dtype)
                self.ft_model = PeftModel.from_pretrained(ft_base, adapter_path)
                self.ft_model.to(self.device)
                self.ft_model.eval()
            else:
                self.ft_model = None
                print(f"  (No adapter found at {adapter_path})")

            self.model_path = model_path
            self.adapter_path = adapter_path
            self._loaded = True

            status = f"Loaded on {device_name}"
            return status

        except Exception as e:
            logger.exception("Error loading models")
            return f"Error: {e}"

    def generate(self, model: Any, prompt: str, max_tokens: int = 128) -> str:
        """Generate text from a model."""
        if model is None:
            return "(Model not loaded)"

        try:
            tokenizer = self.tokenizer

            # Use chat template if available
            messages = [{"role": "user", "content": prompt}]

            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
            else:
                full_prompt = f"User: {prompt}\n\nAssistant:"
                input_ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"]

            input_ids = input_ids.to(self.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(max_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_len = input_ids.shape[1]
            new_tokens = outputs[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            return response if response else "(Empty response)"

        except Exception as e:
            logger.exception("Generation error")
            return f"Error: {e}"

    def generate_base(self, prompt: str, max_tokens: int = 128) -> str:
        """Generate from base model."""
        return self.generate(self.base_model, prompt, max_tokens)

    def generate_finetuned(self, prompt: str, max_tokens: int = 128) -> str:
        """Generate from fine-tuned model."""
        if self.ft_model is None:
            return "(No adapter - train first)"
        return self.generate(self.ft_model, prompt, max_tokens)

    def unload(self) -> None:
        """Unload all models."""
        if self.base_model is not None:
            del self.base_model
        if self.ft_model is not None:
            del self.ft_model
        if self.tokenizer is not None:
            del self.tokenizer
        self.base_model = None
        self.ft_model = None
        self.tokenizer = None
        self._loaded = False


class InferenceEngine:
    """Engine for running inference on trained models."""

    def __init__(self, device: str = "auto"):
        self.state = ModelState()
        self.device = get_device(device)

    def load(
        self,
        model_path: str,
        adapter_path: str = "",
    ) -> str:
        """Load a model for inference.

        Args:
            model_path: HuggingFace model ID or local path.
            adapter_path: Optional path to LoRA adapter.

        Returns:
            Status message.
        """
        # Skip reload if same model/adapter already loaded
        if (self.state.loaded and
            self.state.model_path == model_path and
            self.state.adapter_path == adapter_path):
            return f"Already loaded: {model_path}"

        if not model_path:
            return "Error: No model path specified"

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            logger.info(f"Loading model: {model_path}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Use float16 on MPS/CUDA, float32 on CPU
            dtype = torch.float16 if self.device.type in ("mps", "cuda") else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            ).to(dtype).to(self.device)

            # Load adapter if specified
            adapter_loaded = False
            if adapter_path and Path(adapter_path).exists():
                adapter_config = Path(adapter_path) / "adapter_config.json"
                if adapter_config.exists():
                    try:
                        logger.info(f"Loading adapter: {adapter_path}")
                        model = PeftModel.from_pretrained(model, adapter_path)
                        adapter_loaded = True
                    except Exception as e:
                        logger.warning(f"Could not load adapter: {e}")

            model.eval()

            self.state = ModelState(
                loaded=True,
                model_path=model_path,
                adapter_path=adapter_path if adapter_loaded else "",
                model=model,
                tokenizer=tokenizer,
                device=self.device,
            )

            status = f"Loaded: {model_path}"
            if adapter_loaded:
                status += f" + adapter"
            status += f" on {self.device}"
            return status

        except Exception as e:
            logger.exception("Error loading model")
            return f"Error: {e}"

    def unload(self) -> None:
        """Unload the current model."""
        if self.state.model is not None:
            del self.state.model
        if self.state.tokenizer is not None:
            del self.state.tokenizer
        self.state = ModelState()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from the loaded model.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """
        if not self.state.loaded or self.state.model is None:
            return "Error: No model loaded"

        try:
            model = self.state.model
            tokenizer = self.state.tokenizer

            # Use chat template if available
            messages = [{"role": "user", "content": prompt}]

            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
            else:
                full_prompt = f"User: {prompt}\n\nAssistant:"
                input_ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"]

            # Move to model device
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones_like(input_ids)

            # Generate with greedy decoding for stability
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(max_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only new tokens
            input_len = input_ids.shape[1]
            new_tokens = outputs[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            return response if response else "(Empty response)"

        except Exception as e:
            logger.exception("Generation error")
            return f"Error: {e}"


# Global instances
_engine: InferenceEngine | None = None
_compare_engine: CompareEngine | None = None


def get_inference_engine(device: str = "auto") -> InferenceEngine:
    """Get the global inference engine instance."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine(device)
    return _engine


def get_compare_engine(device: str = "auto") -> CompareEngine:
    """Get the global compare engine instance."""
    global _compare_engine
    if _compare_engine is None:
        _compare_engine = CompareEngine(device)
    return _compare_engine
