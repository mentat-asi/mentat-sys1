import argparse
import json
import re
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


def test_public_identity_registers_current_and_legacy_releases() -> None:
    assert mentat_sys1.__version__ == "0.2.0"
    assert mentat_sys1.MODEL_ID == "mentat-sys1-v0.1"
    assert {
        "mentat-sys1-v0.1",
        "mentat-sys1-v0.2",
    } == mentat_sys1.SUPPORTED_MODEL_IDS


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


def test_public_package_and_command_use_version_neutral_identity() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert payload["project"]["name"] == "mentat-sys1"
    assert payload["project"]["version"] == "0.2.0"
    assert (
        "torchvision==0.23.0"
        in payload["project"]["optional-dependencies"]["serve"]
    )
    assert payload["project"]["scripts"] == {
        "mentat-sys1": "mentat_sys1.cli:main",
    }


def test_readme_documents_current_and_legacy_releases() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    reproducibility = (ROOT / "docs/REPRODUCIBILITY.md").read_text(
        encoding="utf-8"
    )

    for marker in (
        "https://github.com/mentat-asi/mentat-sys1",
        "https://huggingface.co/yunqu/mentat-sys1-v0.1",
        "https://huggingface.co/yunqu/mentat-sys1-v0.2",
        "206/231",
        "configs/mentat-sys1-v0.2.json",
    ):
        assert marker in readme
    assert "temperature_by_type-v2" in reproducibility


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


def test_readme_exposes_current_release_and_reproducibility() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_markers = (
        "SemIF 187/231",
        "Mentat v0.1 200/231",
        "Mentat v0.2 206/231",
        "git clone https://github.com/mentat-asi/mentat-sys1.git",
        "hf download yunqu/mentat-sys1-v0.2",
        '"model": "mentat-sys1-v0.2"',
        "231/231 valid",
        "zero prediction differences",
        "results/v1.4.2.2/jevbench-v1.4.2.2-results.json",
        "| 1 | Plumb-4B | 207 | 89.61% |",
        "| **2** | **Mentat v0.2** | **206** | **89.18%** |",
        "| 3 | Mentat v0.1 | 200 | 86.58% |",
        "| 3 | Jev 1.13.0 | 200 | 86.58% |",
        "| 4 | Imajev-4B | 199 | 86.15% |",
        "| 5 | JevK5 v0.2.0 | 197 | 85.28% |",
        "| 6 | SemIF | 187 | 80.95% |",
        "git checkout v0.2.0",
        "--revision v0.2.0",
        "hf download Qwen/Qwen3.5-4B",
    )

    for marker in required_markers:
        assert marker in readme

    forbidden_markers = (
        "git clone https://github.com/mentat-asi/mentat-sys1-v0.1.git",
        "submitter-reported",
        "Current composite leaderboard",
        "Current Composite Leaderboard",
        "JevBench Score",
        "results/v1.2/jevbench-v1.2-per-task.json",
    )

    for marker in forbidden_markers:
        assert marker not in readme

    api_section = readme.split("## Call the API", maxsplit=1)[1]
    api_section = api_section.split("\n## ", maxsplit=1)[0]
    assert '"model": "mentat-sys1-v0.1"' not in api_section


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
        "## JevBench Public-231 Leaderboard",
        "results/v1.4.2.2/jevbench-v1.4.2.2-results.json",
        "| 1 | Plumb-4B | 207 | 89.61% |",
        "| **2** | **mentat-sys1-v0.1** | **200** | **86.58%** |",
        "| 2 | Jev 1.13.0 | 200 | 86.58% |",
        "| 4 | Imajev-4B | 199 | 86.15% |",
        "| 5 | JevK5 v0.2.0 | 197 | 85.28% |",
        "| 18 | ZeroEntropy zerank-2 | 162 | 70.13% |",
    )

    for marker in required_markers:
        assert marker in model_card

    forbidden_markers = (
        "submitter-reported",
        "Current composite leaderboard",
        "Current Composite Leaderboard",
        "JevBench Score",
        "results/v1.2/jevbench-v1.2-per-task.json",
    )

    for marker in forbidden_markers:
        assert marker not in model_card

    issue_ids = re.findall(
        r"github\.com/fstandhartinger/jevbench/issues/(\d+)",
        model_card,
    )
    assert set(issue_ids) == {"186"}


def test_public_docs_show_a_runnable_jev_compatible_request() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
    common_markers = (
        "same TypeSafe wire format as Jev",
        "https://docs.typesafe.ai/api",
        "Self-hosted Mentat does not require an API key.",
        "curl --fail --request POST",
        '"type": "choice"',
        '"choice": "shirt"',
        '"probabilities"',
        '"unknown_probability"',
        '"abstained": false',
        "`noul`",
        "`score`",
        "Mentat adds diagnostic fields",
    )

    for document in (readme, model_card):
        normalized = " ".join(document.split())
        for marker in common_markers:
            assert marker in normalized

    assert "## Call the API" in readme
    assert '"model": "mentat-sys1-v0.2"' in readme
    assert "## Use the model" in model_card
    assert '"model": "mentat-sys1-v0.1"' in model_card


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
