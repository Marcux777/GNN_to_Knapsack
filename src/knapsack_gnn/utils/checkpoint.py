"""
Enhanced checkpoint utilities with reproducibility metadata.

Saves comprehensive information for experiment reproducibility including
configuration, environment, git state, and hardware info.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


def get_git_info(repo_path: Path | None = None) -> dict[str, Any]:
    """
    Get git repository information for reproducibility.

    Args:
        repo_path: Path to git repository (defaults to current directory)

    Returns:
        Dictionary with git information:
        - commit_hash: Current commit SHA
        - branch: Current branch name
        - is_dirty: Whether there are uncommitted changes
        - diff: Unified diff of uncommitted changes (if any)
        - remote_url: URL of origin remote
    """
    if not GIT_AVAILABLE:
        return {"error": "GitPython not installed"}

    try:
        repo = git.Repo(repo_path or ".", search_parent_directories=True)

        git_info = {
            "commit_hash": repo.head.commit.hexsha,
            "commit_message": repo.head.commit.message.strip(),
            "branch": repo.active_branch.name if not repo.head.is_detached else "detached",
            "is_dirty": repo.is_dirty(),
            "timestamp": datetime.fromtimestamp(repo.head.commit.committed_date).isoformat(),
        }

        # Get remote URL if available
        try:
            git_info["remote_url"] = repo.remotes.origin.url
        except (AttributeError, ValueError):
            git_info["remote_url"] = None

        # Get diff if repo is dirty
        if repo.is_dirty():
            git_info["diff"] = repo.git.diff()
        else:
            git_info["diff"] = None

        return git_info

    except (git.InvalidGitRepositoryError, git.GitCommandError) as e:
        return {"error": f"Git error: {e}"}


def get_environment_info() -> dict[str, Any]:
    """
    Get Python environment information.

    Returns:
        Dictionary with environment information:
        - python_version: Python version
        - platform: OS platform
        - hostname: Machine hostname
        - packages: Installed packages and versions
    """
    env_info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }

    # Get pip freeze output
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        env_info["packages"] = result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        env_info["packages"] = []

    return env_info


def get_hardware_info() -> dict[str, Any]:
    """
    Get hardware information (CPU, GPU, memory).

    Returns:
        Dictionary with hardware information
    """
    hw_info = {
        "cpu_count": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        hw_info["cuda_version"] = torch.version.cuda
        hw_info["cudnn_version"] = torch.backends.cudnn.version()
        hw_info["gpu_count"] = torch.cuda.device_count()
        hw_info["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
    else:
        hw_info["cuda_version"] = None
        hw_info["cudnn_version"] = None
        hw_info["gpu_count"] = 0
        hw_info["gpus"] = []

    return hw_info


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex string of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_checkpoint_metadata(
    checkpoint_dir: Path,
    config: Any | None = None,
    seed: int | None = None,
    deterministic: bool | None = None,
    additional_info: dict[str, Any] | None = None,
) -> None:
    """
    Save comprehensive reproducibility metadata to checkpoint directory.

    Creates the following files in checkpoint_dir:
    - config.yaml: Experiment configuration (if provided)
    - environment.txt: pip freeze output
    - git_info.json: Git repository state
    - reproducibility.json: Seeds, determinism flags, hardware info
    - metadata.json: Timestamp and other metadata

    Args:
        checkpoint_dir: Directory to save metadata files
        config: Experiment configuration (ExperimentConfig or dict)
        seed: Random seed used
        deterministic: Whether deterministic mode was enabled
        additional_info: Additional metadata to save

    Example:
        >>> from knapsack_gnn.config import load_config
        >>> config = load_config("configs/train_default.yaml")
        >>> checkpoint_dir = Path("checkpoints/run_001")
        >>> save_checkpoint_metadata(
        ...     checkpoint_dir,
        ...     config=config,
        ...     seed=42,
        ...     deterministic=True
        ... )
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    if config is not None:
        if hasattr(config, "model_dump"):  # Pydantic model
            import yaml

            config_dict = config.model_dump()
            with open(checkpoint_dir / "config.yaml", "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif isinstance(config, dict):
            import yaml

            with open(checkpoint_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Save environment info
    env_info = get_environment_info()
    with open(checkpoint_dir / "environment.txt", "w") as f:
        f.write(f"Python: {env_info['python_version']}\n")
        f.write(f"Platform: {env_info['platform']}\n")
        f.write(f"Hostname: {env_info['hostname']}\n\n")
        f.write("Installed packages:\n")
        for pkg in env_info["packages"]:
            f.write(f"  {pkg}\n")

    # Save git info
    git_info = get_git_info()
    with open(checkpoint_dir / "git_info.json", "w") as f:
        json.dump(git_info, f, indent=2)

    # Save diff separately if exists
    if git_info.get("diff"):
        with open(checkpoint_dir / "git_diff.patch", "w") as f:
            f.write(git_info["diff"])

    # Save reproducibility info
    repro_info = {
        "seed": seed,
        "deterministic": deterministic,
        "timestamp": datetime.now().isoformat(),
        "hardware": get_hardware_info(),
        "pytorch_version": torch.__version__,
    }

    if additional_info:
        repro_info["additional"] = additional_info

    with open(checkpoint_dir / "reproducibility.json", "w") as f:
        json.dump(repro_info, f, indent=2)

    # Save general metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "files_created": [
            "config.yaml",
            "environment.txt",
            "git_info.json",
            "reproducibility.json",
        ],
    }

    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    """
    Load all metadata from a checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Dictionary containing all metadata

    Example:
        >>> metadata = load_checkpoint_metadata(Path("checkpoints/run_001"))
        >>> print(f"Seed used: {metadata['reproducibility']['seed']}")
        >>> print(f"Commit: {metadata['git']['commit_hash']}")
    """
    checkpoint_dir = Path(checkpoint_dir)

    metadata = {}

    # Load reproducibility info
    repro_file = checkpoint_dir / "reproducibility.json"
    if repro_file.exists():
        with open(repro_file) as f:
            metadata["reproducibility"] = json.load(f)

    # Load git info
    git_file = checkpoint_dir / "git_info.json"
    if git_file.exists():
        with open(git_file) as f:
            metadata["git"] = json.load(f)

    # Load config
    config_file = checkpoint_dir / "config.yaml"
    if config_file.exists():
        import yaml

        with open(config_file) as f:
            metadata["config"] = yaml.safe_load(f)

    # Load general metadata
    metadata_file = checkpoint_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata["metadata"] = json.load(f)

    return metadata
