from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[1]
TERRAFORM_DIR = ROOT / "infrastructure" / "terraform"
VARIABLES = TERRAFORM_DIR / "variables.tf"
EXAMPLE_VARIABLES = TERRAFORM_DIR / "terraform.tfvars.example"
DEPLOYMENT_RUNBOOK = TERRAFORM_DIR / "DEPLOYMENT.md"
ECR_LIFECYCLE_POLICY = ROOT / "infrastructure" / "ecr-lifecycle-policy.json"


def read_terraform() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(TERRAFORM_DIR.glob("*.tf"))
    )


def test_fargate_task_contains_thin_frontend_and_backend():
    terraform = read_terraform()

    assert 'requires_compatibilities = ["FARGATE"]' in terraform
    assert 'network_mode             = "awsvpc"' in terraform
    assert "var.backend_image_uri" in terraform
    assert "var.frontend_image_uri" in terraform
    assert 'value = "http://127.0.0.1:8000"' in terraform
    assert 'condition     = "HEALTHY"' in terraform
    assert "http://127.0.0.1:8000/ready" in terraform


def test_backend_runtime_contract_is_preserved():
    terraform = read_terraform()

    expected_environment = {
        '        { name = "FINGUARD_CACHE_MODE", value = "memory" },',
        '        { name = "FINGUARD_MODEL_LOCAL_ONLY", value = "1" },',
        '        { name = "HF_HUB_OFFLINE", value = "1" },',
        '        { name = "TRANSFORMERS_OFFLINE", value = "1" },',
    }
    assert expected_environment.issubset(set(terraform.splitlines()))
    assert 'name      = "XAI_API_KEY"' in terraform
    assert "valueFrom = var.xai_secret_arn" in terraform
    assert "FINGUARD_CACHE_MODE=redis" not in terraform


def test_alb_targets_only_streamlit():
    terraform = read_terraform()

    assert 'target_type = "ip"' in terraform
    assert 'container_name   = local.frontend_container_name' in terraform
    assert "container_port   = 8501" in terraform
    assert 'path                = "/_stcore/health"' in terraform
    assert "container_port   = 8000" not in terraform


def test_security_groups_do_not_expose_fastapi():
    terraform = read_terraform()

    assert 'description                  = "Streamlit from ALB"' in terraform
    assert "from_port                    = 8501" in terraform
    assert "to_port                      = 8501" in terraform
    assert "from_port                    = 8000" not in terraform
    assert "to_port                      = 8000" not in terraform


def test_execution_role_has_narrow_secret_access_and_no_task_role():
    terraform = read_terraform()

    assert 'actions   = ["secretsmanager:GetSecretValue"]' in terraform
    assert "resources = [var.xai_secret_arn]" in terraform
    assert "task_role_arn" not in terraform
    assert "aws_secretsmanager_secret" not in terraform


def test_cloudwatch_logs_are_separate_for_each_container():
    terraform = read_terraform()

    assert 'resource "aws_cloudwatch_log_group" "backend"' in terraform
    assert 'resource "aws_cloudwatch_log_group" "frontend"' in terraform
    assert terraform.count('logDriver = "awslogs"') == 2


def test_task_sizing_accounts_for_packaged_runtime():
    terraform = read_terraform()

    assert 'cpu                      = "2048"' in terraform
    assert 'memory                   = "4096"' in terraform
    assert "size_in_gib = 30" in terraform
    assert "cpu               = 1792" in terraform
    assert "memoryReservation = 3072" in terraform
    assert "cpu               = 256" in terraform
    assert "memoryReservation = 512" in terraform


def test_image_inputs_are_required_and_digest_pinned():
    variables = VARIABLES.read_text(encoding="utf-8")

    backend_block, frontend_and_later = variables.split(
        'variable "backend_image_uri"', maxsplit=1
    )[1].split('variable "frontend_image_uri"', maxsplit=1)
    frontend_block = frontend_and_later.split(
        'variable "xai_secret_arn"', maxsplit=1
    )[0]

    assert "default" not in backend_block
    assert "default" not in frontend_block
    assert "finguard-api@sha256:" in backend_block
    assert "finguard-ui@sha256:" in frontend_block
    assert "000000000000" in backend_block
    assert "000000000000" in frontend_block


def test_example_inputs_are_synthetic_and_cannot_be_applied_unchanged():
    example = EXAMPLE_VARIABLES.read_text(encoding="utf-8")

    assert "<account-id>" in example
    assert "<backend-image-digest>" in example
    assert "<frontend-image-digest>" in example
    assert "<existing-secrets-manager-secret-arn>" in example
    assert not re.search(r"\b\d{12}\.dkr\.ecr\.", example)


def test_runbook_requires_reviewed_plan_and_has_no_auto_apply():
    runbook = DEPLOYMENT_RUNBOOK.read_text(encoding="utf-8")

    assert "aws sts get-caller-identity" in runbook
    assert "--image-tag-mutability IMMUTABLE" in runbook
    assert "--image-scanning-configuration scanOnPush=true" in runbook
    assert "@$BACKEND_DIGEST" in runbook
    assert "@$FRONTEND_DIGEST" in runbook
    assert "terraform -chdir=infrastructure/terraform plan" in runbook
    assert "explicit approval" in runbook
    assert "-auto-approve" not in runbook
    assert "XAI_API_KEY=" not in runbook


def test_ecr_lifecycle_policy_limits_demo_image_storage():
    policy = json.loads(ECR_LIFECYCLE_POLICY.read_text(encoding="utf-8"))

    untagged, tagged = policy["rules"]
    assert untagged["selection"] == {
        "tagStatus": "untagged",
        "countType": "sinceImagePushed",
        "countUnit": "days",
        "countNumber": 14,
    }
    assert tagged["selection"] == {
        "tagStatus": "tagged",
        "tagPrefixList": ["deployment-"],
        "countType": "imageCountMoreThan",
        "countNumber": 3,
    }
