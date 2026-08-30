from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile.frontend"
DOCKERIGNORE = ROOT / "Dockerfile.frontend.dockerignore"
REQUIREMENTS_INPUT = ROOT / "deployment" / "frontend-requirements.in"
REQUIREMENTS_LOCK = ROOT / "deployment" / "frontend-requirements.lock"

HEAVYWEIGHT_PACKAGES = {
    "chromadb",
    "langchain",
    "langgraph",
    "onnxruntime",
    "sentence-transformers",
    "torch",
    "transformers",
}


def locked_package_names() -> set[str]:
    names = set()
    for line in REQUIREMENTS_LOCK.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([a-zA-Z0-9_.-]+)==", line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


def test_frontend_dockerfile_exists_and_uses_python_311():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM python:3.11-slim AS runtime" in dockerfile
    assert "deployment/frontend-requirements.lock" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--no-deps" in dockerfile


def test_frontend_dependencies_are_minimal_and_exclude_backend_ml():
    requirements_input = REQUIREMENTS_INPUT.read_text(encoding="utf-8")
    locked = locked_package_names()

    assert {"pandas", "python-dotenv", "streamlit"}.issubset(locked)
    assert not HEAVYWEIGHT_PACKAGES.intersection(locked)
    assert "pandas==3.0.5" in requirements_input
    assert "python-dotenv==1.2.2" in requirements_input
    assert "streamlit==1.60.0" in requirements_input


def test_every_frontend_version_exists_in_root_uv_lock():
    root_lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    root_versions: dict[str, set[str]] = {}
    for package in root_lock["package"]:
        root_versions.setdefault(package["name"], set()).add(package["version"])

    frontend_versions = dict(
        re.findall(
            r"^([a-zA-Z0-9_.-]+)==([^ ;\\]+)",
            REQUIREMENTS_LOCK.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )

    assert frontend_versions
    for name, version in frontend_versions.items():
        normalized_name = name.lower().replace("_", "-")
        assert version in root_versions[normalized_name]


def test_frontend_image_copies_only_ui_source():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    dockerignore = DOCKERIGNORE.read_text(encoding="utf-8")

    assert "COPY src " not in dockerfile
    assert "data/chroma" not in dockerfile
    assert "deployment/prepare_model_assets" not in dockerfile
    assert "src/graph" not in dockerfile
    assert "src/ingestion" not in dockerfile
    assert "src/llm" not in dockerfile
    assert "!src/ui/app.py" in dockerignore
    assert "!src/ui/api_client.py" in dockerignore
    assert "!deployment/frontend-requirements.lock" in dockerignore
    assert "!data/" not in dockerignore


def test_frontend_runs_streamlit_non_root_on_port_8501():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "USER finguard-ui" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert '"--server.port=8501"' in dockerfile
    assert '"--server.headless=true"' in dockerfile
    assert '"--browser.gatherUsageStats=false"' in dockerfile
    assert "8501/_stcore/health" in dockerfile


def test_frontend_image_has_no_backend_url_or_secret():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "FINGUARD_API_BASE_URL" not in dockerfile
    assert "XAI_API_KEY" not in dockerfile
    assert "OPENAI_API_KEY" not in dockerfile
    assert "AWS_" not in dockerfile
    assert ".env" not in dockerfile
