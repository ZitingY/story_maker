"""Tests for NLU-1: intent training data augmentation."""
import json
import pytest
import tempfile
from pathlib import Path

from training.data_augmenter import generate_dataset, _GENERATORS


class TestDataAugmenter:
    """Test the data augmenter generates valid data."""

    def test_generate_returns_path(self, tmp_path):
        output = str(tmp_path / "test_data.jsonl")
        path = generate_dataset(num_per_class=10, output_path=output, seed=42)
        assert path == output

    def test_file_is_valid_jsonl(self, tmp_path):
        output = str(tmp_path / "test_data.jsonl")
        generate_dataset(num_per_class=10, output_path=output, seed=42)
        with open(output) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 10 * len(_GENERATORS)

    def test_all_intents_present(self, tmp_path):
        output = str(tmp_path / "test_data.jsonl")
        generate_dataset(num_per_class=10, output_path=output, seed=42)
        with open(output) as f:
            intents = {json.loads(line)["intent"] for line in f if line.strip()}
        assert intents == set(_GENERATORS.keys())

    def test_samples_have_text_and_intent(self, tmp_path):
        output = str(tmp_path / "test_data.jsonl")
        generate_dataset(num_per_class=5, output_path=output, seed=42)
        with open(output) as f:
            for line in f:
                sample = json.loads(line)
                assert "text" in sample
                assert "intent" in sample
                assert len(sample["text"]) > 0

    def test_balanced_distribution(self, tmp_path):
        output = str(tmp_path / "test_data.jsonl")
        generate_dataset(num_per_class=50, output_path=output, seed=42)
        from collections import Counter
        with open(output) as f:
            counts = Counter(json.loads(line)["intent"] for line in f if line.strip())
        for intent in _GENERATORS:
            assert counts[intent] == 50

    def test_reproducible_with_seed(self, tmp_path):
        out1 = str(tmp_path / "data1.jsonl")
        out2 = str(tmp_path / "data2.jsonl")
        generate_dataset(num_per_class=20, output_path=out1, seed=123)
        generate_dataset(num_per_class=20, output_path=out2, seed=123)
        with open(out1) as f1, open(out2) as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
        assert lines1 == lines2

    def test_default_500_per_class(self, tmp_path):
        output = str(tmp_path / "test_data.jsonl")
        generate_dataset(output_path=output, seed=42)
        with open(output) as f:
            count = sum(1 for _ in f)
        assert count == 500 * len(_GENERATORS)


class TestIndividualGenerators:
    """Test each intent generator produces valid output."""

    @pytest.mark.parametrize("intent", list(_GENERATORS.keys()))
    def test_generator_returns_list(self, intent):
        gen = _GENERATORS[intent]
        samples = gen(10)
        assert isinstance(samples, list)
        assert len(samples) == 10
        for s in samples:
            assert isinstance(s, str)
            assert len(s) > 3

    @pytest.mark.parametrize("intent", list(_GENERATORS.keys()))
    def test_generator_diversity(self, intent):
        gen = _GENERATORS[intent]
        samples = gen(100)
        unique = set(samples)
        # At least 30% unique samples (lower for "other" which has more fixed templates)
        assert len(unique) >= 30


class TestLoadDataFromJsonl:
    """Test the JSONL loading function in train_intent."""

    def test_load_data(self, tmp_path):
        from training.train_intent import load_data_from_jsonl
        output = str(tmp_path / "test.jsonl")
        generate_dataset(num_per_class=20, output_path=output, seed=42)
        texts, labels = load_data_from_jsonl(output)
        assert len(texts) == 20 * len(_GENERATORS)
        assert len(labels) == len(texts)
        assert all(0 <= l < 8 for l in labels)

    def test_load_empty_file(self, tmp_path):
        from training.train_intent import load_data_from_jsonl
        output = str(tmp_path / "empty.jsonl")
        Path(output).write_text("")
        texts, labels = load_data_from_jsonl(output)
        assert texts == []
        assert labels == []
