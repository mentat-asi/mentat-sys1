import argparse
import json
import tomllib
from pathlib import Path

import mentat_sys1
from mentat_sys1.cli import build_parser

ROOT = Path(__file__).parents[2]


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(actions) == 1
    return actions[0].choices


def test_public_identity_is_lowercase_sys1_v01() -> None:
    assert mentat_sys1.__version__ == "0.1.1"
    assert mentat_sys1.MODEL_ID == "mentat-sys1-v0.1"


def test_public_cli_omits_training_commands() -> None:
    commands = _subcommands(build_parser())

    assert set(commands) == {
        "calibrate",
        "serve",
        "audit",
        "release",
    }


def test_public_config_omits_training_recipe() -> None:
    payload = json.loads(
        (ROOT / "configs/mentat-sys1-v0.1.json").read_text(encoding="utf-8")
    )
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["project_id"] == "mentat-sys1-v0.1"
    assert payload["base_model"] == {
        "repository": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }
    assert payload["runtime"] == {
        "max_length": 16384,
        "readout_codes": 256,
    }
    assert "training" not in payload
    assert "resources" not in payload
    assert "/data/" not in encoded
    assert "/Users/" not in encoded


def test_public_config_filename_matches_identity() -> None:
    assert (ROOT / "configs/mentat-sys1-v0.1.json").is_file()
    assert not (
        ROOT / "configs" / ("mentat-" + "jev-v0.1.json")
    ).exists()
    assert not (ROOT / "configs" / ("mentat-" + "jev-v1.json")).exists()


def test_public_package_and_command_use_v01_identity() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert payload["project"]["name"] == "mentat-sys1-v0.1"
    assert payload["project"]["version"] == "0.1.1"
    assert (
        "torchvision==0.23.0"
        in payload["project"]["optional-dependencies"]["serve"]
    )
    assert payload["project"]["scripts"] == {
        "mentat-sys1": "mentat_sys1.cli:main",
    }


def test_public_documents_do_not_advertise_v1_identity() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "docs/DATA_CARD.md",
        ROOT / "docs/JEVBENCH_SUBMISSION.md",
        ROOT / "docs/MODEL_CARD.md",
        ROOT / "docs/REPRODUCIBILITY.md",
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "Mentat-" not in text
        assert "mentat-" + "jev" not in text


def test_public_documents_use_approved_hugging_face_repository() -> None:
    paths = [
        ROOT / "docs/JEVBENCH_SUBMISSION.md",
        ROOT / "docs/MODEL_CARD.md",
        ROOT / "docs/REPRODUCIBILITY.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "huggingface.co/yunqu/mentat-sys1-v0.1" in text
    assert "hf download yunqu/mentat-sys1-v0.1" in text
    assert "huggingface.co/mentat-asi/mentat-sys1-v0.1" not in text
    assert "hf download mentat-asi/mentat-sys1-v0.1" not in text


def test_public_documents_pin_code_patch_separately_from_model() -> None:
    submission = (ROOT / "docs/JEVBENCH_SUBMISSION.md").read_text(encoding="utf-8")
    model_card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
    reproducibility = (ROOT / "docs/REPRODUCIBILITY.md").read_text(
        encoding="utf-8"
    )

    assert (
        "bound to the code `v0.1.1` and model `v0.1.0` release tags"
        in submission
    )
    assert "- Code revision: `v0.1.1`" in submission
    assert "- Model revision: `v0.1.0`" in submission
    assert "git checkout v0.1.1" in submission
    assert "git checkout v0.1.1" in reproducibility
    assert (
        "- Code: `https://github.com/mentat-asi/mentat-sys1-v0.1`, "
        "tag `v0.1.1`"
        in model_card
    )
    assert (
        "- Model: `https://huggingface.co/yunqu/mentat-sys1-v0.1`, "
        "tag `v0.1.0`"
        in model_card
    )


def test_readme_exposes_results_context_and_frozen_release_commands() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_markers = (
        "https://github.com/mentat-asi/mentat-sys1-v0.1",
        "https://huggingface.co/yunqu/mentat-sys1-v0.1",
        "https://github.com/fstandhartinger/jevbench/issues/186",
        "## Public-231 context",
        "same 231 public items",
        "submitter-reported",
        "not an official leaderboard",
        "| mentat-sys1-v0.1 | **200** | 48 | 66 | 86 |",
        "https://github.com/fstandhartinger/jevbench/issues/84",
        "https://github.com/fstandhartinger/jevbench/issues/119",
        "https://github.com/fstandhartinger/jevbench/issues/159",
        "git checkout v0.1.1",
        "hf download yunqu/mentat-sys1-v0.1",
        "--revision v0.1.0",
        "hf download Qwen/Qwen3.5-4B",
    )

    for marker in required_markers:
        assert marker in readme


def test_model_card_links_code_and_explains_the_two_downloads() -> None:
    model_card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")

    required_markers = (
        "https://github.com/mentat-asi/mentat-sys1-v0.1/tree/v0.1.1",
        "https://huggingface.co/yunqu/mentat-sys1-v0.1/tree/v0.1.0",
        "https://github.com/fstandhartinger/jevbench/issues/186",
        "hf download yunqu/mentat-sys1-v0.1",
        "--revision v0.1.0",
        "hf download Qwen/Qwen3.5-4B",
        "does not download the Qwen base weights",
        "## Public-231 context",
    )

    for marker in required_markers:
        assert marker in model_card


def test_public_surface_omits_private_recipe_and_plans() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "NOTICE",
        ROOT / "pyproject.toml",
        ROOT / "configs/mentat-sys1-v0.1.json",
        ROOT / "data/dataset-manifest.json",
        ROOT / "data/sources.lock.json",
        ROOT / "docs/DATA_CARD.md",
        ROOT / "docs/JEVBENCH_SUBMISSION.md",
        ROOT / "docs/MODEL_CARD.md",
        ROOT / "docs/REPRODUCIBILITY.md",
    ]
    forbidden = (
        "learning_rate",
        "warmup_steps",
        "batch_size",
        "rationale_weight",
        "gpu_indices",
        "training_min_free_memory_mib",
        "/data/" + "quy/",
        "/Users/" + "bytedance/",
        "TD" + "4B",
        "td" + "4b",
        "transfer-training",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for marker in forbidden:
        assert marker not in text

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "train" not in metadata["project"]["optional-dependencies"]
    assert (ROOT / "src/mentat_sys1/__init__.py").is_file()
    assert not (ROOT / "src" / ("mentat_" + "jev_v1")).exists()
    assert not (ROOT / "src/mentat_sys1/training").exists()
    assert not (ROOT / "src/mentat_sys1/audit/resources.py").exists()
