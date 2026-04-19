"""Human labeling schema and paths for calibrating LLM-as-judge evaluators."""

from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "data" / "human_labels.schema.json"
EXAMPLE_LABELS_PATH = Path(__file__).resolve().parent / "data" / "human_labels.example.jsonl"
