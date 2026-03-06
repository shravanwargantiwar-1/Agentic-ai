from pathlib import Path


def openvino_model_exists(model_dir: str) -> bool:
    base = Path(model_dir)
    return (base / "model.xml").exists() and (base / "model.bin").exists()
