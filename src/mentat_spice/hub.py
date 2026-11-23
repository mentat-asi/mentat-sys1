"""Download only inference artifacts and the qualified backbone revision."""
from pathlib import Path

MODEL_ID = "yunqu/mentat-Spice-4B"
BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "decision_readout.json",
    "inference_config.json",
)


def resolve_pretrained(
    model_id=MODEL_ID, *, base_model=None, revision=None, cache_dir=None,
    local_files_only=False, token=None,
):
    from huggingface_hub import snapshot_download

    options = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "token": token,
    }
    package = Path(model_id).expanduser()
    if not package.is_dir():
        package = Path(snapshot_download(
            repo_id=str(model_id), revision=revision,
            allow_patterns=[f"model/{name}" for name in MODEL_FILES], **options,
        ))
    adapter = package / "model"
    missing = [name for name in MODEL_FILES if not (adapter / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete Spice model package: missing {', '.join(missing)}")
    if base_model is None:
        base_model = snapshot_download(
            repo_id=BASE_MODEL_ID, revision=BASE_REVISION, **options,
        )
    base = Path(base_model).expanduser()
    if not base.is_dir():
        raise ValueError("base_model must point to a downloaded backbone directory")
    return str(base), str(adapter)
