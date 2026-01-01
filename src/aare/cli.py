"""Command-line interface for Aare Model Trainer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from aare.utils.logging import setup_logging, get_logger


logger = get_logger(__name__)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="aare",
        description="Aare HITL - Human-in-the-Loop Fine-Tuning for LLMs",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--log-format",
        choices=["text", "json"],
        default="text",
        help="Log output format (default: text)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", required=True, help="YAML config file")

    # Compare command (interactive HITL)
    compare_parser = subparsers.add_parser("compare", help="Interactive HITL comparison")
    compare_parser.add_argument("--config", required=True, help="YAML config file")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate from a trained model")
    gen_parser.add_argument("--config", required=True, help="YAML config file")

    # Data command
    data_parser = subparsers.add_parser("data", help="Data utilities")
    data_subparsers = data_parser.add_subparsers(dest="data_command")
    validate_parser = data_subparsers.add_parser("validate", help="Validate a dataset")
    validate_parser.add_argument("file", help="Dataset file to validate")

    # Merge subcommand
    merge_parser = data_subparsers.add_parser("merge", help="Merge/filter datasets based on HITL results")
    merge_parser.add_argument("base", help="Base dataset file")
    merge_parser.add_argument("--add", help="Add samples from this file")
    merge_parser.add_argument("--remove", help="Remove samples matching this file")
    merge_parser.add_argument("--output", "-o", required=True, help="Output file")

    # Import subcommand (CSV to JSON)
    import_parser = data_subparsers.add_parser("import", help="Import CSV or Google Sheets to JSON dataset format")
    import_parser.add_argument("source", help="CSV file path or Google Sheets URL")
    import_parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    import_parser.add_argument("--instruction-col", default="instruction", help="Column name for instruction (default: instruction)")
    import_parser.add_argument("--output-col", default="output", help="Column name for output (default: output)")
    import_parser.add_argument("--sheet", default=None, help="Sheet name or GID for Google Sheets (default: first sheet)")
    import_parser.add_argument("--api-key", default=None, help="Google API key (or set GOOGLE_API_KEY env var)")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    json_format = args.log_format == "json"
    setup_logging(level=log_level, json_format=json_format)

    if args.command == "train":
        return run_train(args)
    elif args.command == "compare":
        return run_compare(args)
    elif args.command == "generate":
        return run_generate(args)
    elif args.command == "data":
        return run_data(args)
    else:
        parser.print_help()
        return 0


def load_config(path: str) -> dict:
    """Load and validate a YAML config file."""
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix not in (".yaml", ".yml"):
        raise ValueError(f"Config must be YAML (.yaml or .yml), got: {config_path.suffix}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def run_train(args: argparse.Namespace) -> int:
    """Run training from a YAML config file."""
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Required fields
    model = config.get("model")
    dataset = config.get("dataset")
    output_dir = config.get("output_dir", "./output")

    if not model or not dataset:
        logger.error("Config must include 'model' and 'dataset' fields")
        return 1

    # Optional fields with defaults
    method = config.get("method", "lora")
    epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 2)
    learning_rate = config.get("learning_rate", 2e-4)
    lora_rank = config.get("lora_rank", 8)
    device_pref = config.get("device", "auto")

    logger.info(f"Starting training...")
    logger.info(f"  Model: {model}")
    logger.info(f"  Dataset: {dataset}")
    logger.info(f"  Method: {method}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Output: {output_dir}")

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        import torch
        from aare.core.inference import get_device

        # Determine device
        device = get_device(device_pref)
        logger.info(f"  Device: {device}")

        # Use float16 on GPU, float32 on CPU
        dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        logger.info("Loading model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
        ).to(dtype).to(device)

        # Apply LoRA
        if method in ["lora", "qlora"]:
            logger.info(f"Applying LoRA (rank={lora_rank})...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                inference_mode=False,
            )
            base_model = get_peft_model(base_model, lora_config)
            base_model.print_trainable_parameters()

        # Load dataset
        logger.info("Loading dataset...")
        with open(dataset) as f:
            raw_data = json.load(f)

        def format_example(ex):
            if "instruction" in ex and "output" in ex:
                text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
            elif "text" in ex:
                text = ex["text"]
            else:
                text = str(ex)
            return {"text": text}

        formatted = [format_example(ex) for ex in raw_data]
        train_dataset = Dataset.from_list(formatted)

        # Tokenize
        logger.info("Tokenizing...")
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

        tokenized = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)

        # Training
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Disable pin_memory on MPS (not supported)
        use_pin_memory = device.type not in ("mps",)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=use_pin_memory,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=base_model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        logger.info("Training...")
        trainer.train()

        logger.info(f"Saving to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info("Training complete!")
        return 0

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


def run_compare(args: argparse.Namespace) -> int:
    """Run interactive HITL comparison."""
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    model = config.get("model")
    adapter = config.get("adapter", "./output")
    dataset = config.get("dataset")
    output_dir = config.get("output_dir", "./compare_results")
    device = config.get("device", "auto")  # auto, cpu, mps, cuda

    if not model or not dataset:
        logger.error("Config must include 'model' and 'dataset' fields")
        return 1

    # Load dataset
    try:
        with open(dataset) as f:
            samples = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1

    if not samples:
        logger.error("Dataset is empty")
        return 1

    from aare.core.inference import get_compare_engine

    engine = get_compare_engine(device)

    print(f"\nLoading models...")
    print(f"  Base: {model}")
    print(f"  Adapter: {adapter}")

    # Load both models upfront for fast comparison
    engine.load_models(model, adapter)

    # Track results - store actual samples, not just counts
    accepted_samples: list[dict] = []
    rejected_samples: list[dict] = []
    skipped_indices: set[int] = set()
    reviewed_indices: set[int] = set()
    current_idx = 0

    # Cache for generated responses (idx -> (base, finetuned))
    response_cache: dict[int, tuple[str, str]] = {}

    def generate_responses(idx: int) -> tuple[str, str]:
        """Generate responses from both models (with caching)."""
        if idx in response_cache:
            return response_cache[idx]

        sample = samples[idx]
        prompt = sample.get("instruction", sample.get("prompt", sample.get("text", "")))

        # Generate from both models - use shorter max_tokens for speed
        base_resp = engine.generate_base(prompt, max_tokens=256)
        ft_resp = engine.generate_finetuned(prompt, max_tokens=256)

        response_cache[idx] = (base_resp, ft_resp)
        return base_resp, ft_resp

    def show_sample(idx: int, show_expected: bool = False) -> None:
        """Display a sample with model responses."""
        if idx < 0 or idx >= len(samples):
            return

        sample = samples[idx]
        prompt = sample.get("instruction", sample.get("prompt", sample.get("text", "")))
        expected = sample.get("output", sample.get("response", ""))

        # Show review status
        status = ""
        if idx in reviewed_indices:
            if samples[idx] in accepted_samples:
                status = " [ACCEPTED]"
            elif samples[idx] in rejected_samples:
                status = " [REJECTED]"
            elif idx in skipped_indices:
                status = " [SKIPPED]"

        print(f"\n{'=' * 60}")
        print(f"Sample {idx + 1}/{len(samples)}{status}")
        print(f"[a]ccept  [r]eject  [s]kip  [e]xpected  [n]ext  [p]rev  [q]uit")
        print(f"{'=' * 60}")
        print(f"\nPROMPT:\n{prompt}")

        # Generate responses (cached after first call)
        if idx not in response_cache:
            print(f"\n  Generating responses...")

        base_resp, ft_resp = generate_responses(idx)

        print(f"\n{'─' * 40}")
        print(f"FINE-TUNED:\n{ft_resp[:600]}{'...' if len(ft_resp) > 600 else ''}")
        print(f"\n{'─' * 40}")
        print(f"BASE MODEL:\n{base_resp[:600]}{'...' if len(base_resp) > 600 else ''}")

        if show_expected:
            print(f"\n{'─' * 40}")
            print(f"EXPECTED (training data):\n{expected[:600]}{'...' if len(expected) > 600 else ''}")

    def save_results() -> None:
        """Save accepted and rejected samples to files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        accepted_path = Path(output_dir) / "accepted.json"
        rejected_path = Path(output_dir) / "rejected.json"

        if accepted_samples:
            with open(accepted_path, "w") as f:
                json.dump(accepted_samples, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(accepted_samples)} accepted samples to {accepted_path}")

        if rejected_samples:
            with open(rejected_path, "w") as f:
                json.dump(rejected_samples, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(rejected_samples)} rejected samples to {rejected_path}")

    # Main loop
    print(f"\nResults will be saved to: {output_dir}/")
    show_expected = False

    while True:
        show_sample(current_idx, show_expected=show_expected)
        show_expected = False  # Reset after showing

        try:
            cmd = input(f"\n[a/r/s/e/n/p/q] > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if cmd == "q":
            break
        elif cmd == "a":
            sample = samples[current_idx]
            # Remove from rejected if previously rejected
            if sample in rejected_samples:
                rejected_samples.remove(sample)
            # Add to accepted if not already there
            if sample not in accepted_samples:
                accepted_samples.append(sample)
            skipped_indices.discard(current_idx)
            reviewed_indices.add(current_idx)
            print("  Accepted")
            current_idx = min(current_idx + 1, len(samples) - 1)
        elif cmd == "r":
            sample = samples[current_idx]
            # Remove from accepted if previously accepted
            if sample in accepted_samples:
                accepted_samples.remove(sample)
            # Add to rejected if not already there
            if sample not in rejected_samples:
                rejected_samples.append(sample)
            skipped_indices.discard(current_idx)
            reviewed_indices.add(current_idx)
            print("  Rejected")
            current_idx = min(current_idx + 1, len(samples) - 1)
        elif cmd == "s":
            sample = samples[current_idx]
            # Remove from both lists if previously categorized
            if sample in accepted_samples:
                accepted_samples.remove(sample)
            if sample in rejected_samples:
                rejected_samples.remove(sample)
            skipped_indices.add(current_idx)
            reviewed_indices.add(current_idx)
            print("  Skipped")
            current_idx = min(current_idx + 1, len(samples) - 1)
        elif cmd == "e":
            show_expected = True
        elif cmd == "n":
            current_idx = min(current_idx + 1, len(samples) - 1)
        elif cmd == "p":
            current_idx = max(current_idx - 1, 0)

    # Summary and save
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Accepted: {len(accepted_samples)}")
    print(f"  Rejected: {len(rejected_samples)}")
    print(f"  Skipped:  {len(skipped_indices)}")
    print(f"  Not reviewed: {len(samples) - len(reviewed_indices)}")
    print()

    # Save results to files
    if accepted_samples or rejected_samples:
        save_results()
    else:
        print("  No samples to save.")
    print()

    return 0


def run_generate(args: argparse.Namespace) -> int:
    """Generate from a trained model."""
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    model = config.get("model")
    adapter = config.get("adapter", "")
    prompt = config.get("prompt", "")
    max_tokens = config.get("max_tokens", 512)
    device = config.get("device", "auto")  # auto, cpu, mps, cuda

    if not model:
        logger.error("Config must include 'model' field")
        return 1

    if not prompt:
        logger.error("Config must include 'prompt' field")
        return 1

    from aare.core.inference import get_inference_engine

    engine = get_inference_engine(device)

    logger.info(f"Loading model: {model}")
    if adapter:
        logger.info(f"With adapter: {adapter}")

    status = engine.load(model, adapter)
    logger.info(status)

    logger.info("Generating...")
    response = engine.generate(prompt, max_tokens=max_tokens)

    # Output the response
    print(f"\n{response}\n")

    return 0


def run_data(args: argparse.Namespace) -> int:
    """Data utilities."""
    if args.data_command == "validate":
        return validate_dataset(args.file)
    elif args.data_command == "merge":
        return merge_datasets(args.base, args.add, args.remove, args.output)
    elif args.data_command == "import":
        return import_data(args.source, args.output, args.instruction_col, args.output_col, args.sheet, args.api_key)
    else:
        logger.error("Unknown data command. Use: aare data validate|merge|import")
        return 1


def validate_dataset(file_path: str) -> int:
    """Validate a dataset file (JSON or YAML)."""
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found: {path}")
        return 1

    try:
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to parse file: {e}")
        return 1

    if not isinstance(data, list):
        logger.error("Dataset must be an array/list")
        return 1

    if len(data) == 0:
        logger.error("Dataset is empty")
        return 1

    # Check format
    valid = 0
    issues = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            issues.append(f"Item {i}: not a dict")
            continue

        has_instruction = "instruction" in item or "prompt" in item or "text" in item
        has_output = "output" in item or "response" in item

        if has_instruction and has_output:
            valid += 1
        elif has_instruction:
            issues.append(f"Item {i}: missing output/response field")
        else:
            issues.append(f"Item {i}: missing instruction/prompt/text field")

    # Report
    print(f"\nDataset: {path}")
    print(f"  Total samples: {len(data)}")
    print(f"  Valid samples: {valid}")

    if issues:
        print(f"\nIssues ({len(issues)}):")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    if valid == len(data):
        print("\n  Dataset is valid!")
        return 0
    else:
        print(f"\n  {len(data) - valid} samples have issues")
        return 1


def load_dataset(file_path: str) -> list[dict]:
    """Load a dataset file (JSON or YAML)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f)
        else:
            return json.load(f)


def samples_match(a: dict, b: dict) -> bool:
    """Check if two samples match based on instruction/prompt field."""
    # Get the instruction/prompt from each
    a_key = a.get("instruction", a.get("prompt", a.get("text", "")))
    b_key = b.get("instruction", b.get("prompt", b.get("text", "")))
    return a_key == b_key


def merge_datasets(base_path: str, add_path: str | None, remove_path: str | None, output_path: str) -> int:
    """Merge or filter datasets based on HITL results.

    Usage:
        # Remove rejected samples from training data
        aare data merge original.json --remove rejected.json -o cleaned.json

        # Add new samples to existing dataset
        aare data merge existing.json --add new_samples.json -o combined.json

        # Both: remove bad, add good
        aare data merge original.json --remove rejected.json --add accepted.json -o curated.json
    """
    try:
        base_data = load_dataset(base_path)
        logger.info(f"Loaded base dataset: {len(base_data)} samples")
    except Exception as e:
        logger.error(f"Failed to load base dataset: {e}")
        return 1

    result = list(base_data)

    # Remove samples if specified
    if remove_path:
        try:
            remove_data = load_dataset(remove_path)
            logger.info(f"Loaded remove dataset: {len(remove_data)} samples")

            # Filter out matching samples
            original_count = len(result)
            result = [
                sample for sample in result
                if not any(samples_match(sample, r) for r in remove_data)
            ]
            removed_count = original_count - len(result)
            logger.info(f"Removed {removed_count} samples")

        except Exception as e:
            logger.error(f"Failed to load remove dataset: {e}")
            return 1

    # Add samples if specified
    if add_path:
        try:
            add_data = load_dataset(add_path)
            logger.info(f"Loaded add dataset: {len(add_data)} samples")

            # Add samples that don't already exist
            added_count = 0
            for sample in add_data:
                if not any(samples_match(sample, existing) for existing in result):
                    result.append(sample)
                    added_count += 1

            logger.info(f"Added {added_count} new samples")

        except Exception as e:
            logger.error(f"Failed to load add dataset: {e}")
            return 1

    # Write output
    try:
        output = Path(output_path)
        with open(output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nMerge complete:")
        print(f"  Base: {len(base_data)} samples")
        print(f"  Output: {output} ({len(result)} samples)")

        return 0

    except Exception as e:
        logger.error(f"Failed to write output: {e}")
        return 1


def parse_google_sheets_url(url: str) -> tuple[str, str | None]:
    """Parse a Google Sheets URL to extract spreadsheet ID and optional GID.

    Supports formats:
        https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit#gid=SHEET_GID
        https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit
        https://docs.google.com/spreadsheets/d/SPREADSHEET_ID

    Returns:
        Tuple of (spreadsheet_id, gid or None)
    """
    import re

    # Extract spreadsheet ID
    match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
    if not match:
        raise ValueError(f"Could not parse Google Sheets URL: {url}")

    spreadsheet_id = match.group(1)

    # Extract GID if present
    gid = None
    gid_match = re.search(r'[#&]gid=(\d+)', url)
    if gid_match:
        gid = gid_match.group(1)

    return spreadsheet_id, gid


def fetch_google_sheet_csv(spreadsheet_id: str, gid: str | None = None, api_key: str | None = None) -> str:
    """Fetch a Google Sheet as CSV content.

    Args:
        spreadsheet_id: The Google Sheets spreadsheet ID
        gid: Optional sheet GID (defaults to first sheet)
        api_key: Google API key (optional for public sheets)

    Returns:
        CSV content as string
    """
    import requests

    # Build the export URL
    # For public sheets or with API key, we can use the export URL directly
    base_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export"
    params = {"format": "csv"}

    if gid:
        params["gid"] = gid

    if api_key:
        params["key"] = api_key

    logger.info(f"Fetching Google Sheet: {spreadsheet_id}" + (f" (gid={gid})" if gid else ""))

    response = requests.get(base_url, params=params, timeout=30)

    if response.status_code == 404:
        raise ValueError(
            f"Spreadsheet not found. Make sure the sheet exists and is shared as "
            f"'Anyone with the link can view' or provide a valid API key."
        )
    elif response.status_code == 403:
        raise ValueError(
            f"Access denied. The sheet may be private. Either:\n"
            f"  1. Share the sheet as 'Anyone with the link can view', or\n"
            f"  2. Provide a Google API key with --api-key or GOOGLE_API_KEY env var"
        )
    elif response.status_code != 200:
        raise ValueError(f"Failed to fetch sheet: HTTP {response.status_code}")

    return response.text


def import_data(
    source: str,
    output_path: str,
    instruction_col: str,
    output_col: str,
    sheet: str | None = None,
    api_key: str | None = None,
) -> int:
    """Import CSV file or Google Sheets URL to JSON dataset format.

    Usage:
        # From local CSV file
        aare data import sheet.csv -o dataset.json

        # From Google Sheets URL (public sheet)
        aare data import "https://docs.google.com/spreadsheets/d/ABC123/edit" -o dataset.json

        # With custom columns
        aare data import source.csv -o dataset.json --instruction-col question --output-col answer

        # With specific sheet and API key
        aare data import "https://..." -o dataset.json --sheet 123456 --api-key YOUR_KEY
    """
    import csv
    import io

    # Check if source is a Google Sheets URL
    is_google_sheets = "docs.google.com/spreadsheets" in source

    if is_google_sheets:
        # Fetch from Google Sheets
        try:
            spreadsheet_id, url_gid = parse_google_sheets_url(source)

            # Use sheet arg if provided, otherwise use GID from URL
            gid = sheet if sheet else url_gid

            # Get API key from arg or environment
            key = api_key or os.environ.get("GOOGLE_API_KEY")

            csv_content = fetch_google_sheet_csv(spreadsheet_id, gid, key)
            csv_file = io.StringIO(csv_content)
            source_display = f"Google Sheet {spreadsheet_id}" + (f" (gid={gid})" if gid else "")

        except ValueError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Failed to fetch Google Sheet: {e}")
            return 1
    else:
        # Local CSV file
        path = Path(source)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return 1

        if path.suffix.lower() != ".csv":
            logger.error(f"Expected CSV file, got: {path.suffix}")
            return 1

        csv_file = open(path, newline="", encoding="utf-8")
        source_display = str(path)

    try:
        samples = []
        reader = csv.DictReader(csv_file)

        # Check columns exist
        if reader.fieldnames is None:
            logger.error("CSV data appears to be empty")
            return 1

        if instruction_col not in reader.fieldnames:
            logger.error(f"Column '{instruction_col}' not found. Available: {', '.join(reader.fieldnames)}")
            return 1
        if output_col not in reader.fieldnames:
            logger.error(f"Column '{output_col}' not found. Available: {', '.join(reader.fieldnames)}")
            return 1

        for row in reader:
            instruction = row[instruction_col].strip()
            output = row[output_col].strip()

            # Skip empty rows
            if not instruction or not output:
                continue

            samples.append({
                "instruction": instruction,
                "output": output,
            })

        if not samples:
            logger.error("No valid samples found in data")
            return 1

        # Write JSON output
        output = Path(output_path)
        with open(output, "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        print(f"\nImport complete:")
        print(f"  Source: {source_display}")
        print(f"  Output: {output} ({len(samples)} samples)")

        return 0

    except Exception as e:
        logger.error(f"Failed to import data: {e}")
        return 1
    finally:
        if not is_google_sheets:
            csv_file.close()


if __name__ == "__main__":
    sys.exit(main())
