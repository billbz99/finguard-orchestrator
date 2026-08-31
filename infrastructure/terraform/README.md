# FinGuard ECS/Fargate skeleton

This root module defines the smallest portfolio-demo AWS topology for running
the thin Streamlit frontend and packaged FastAPI backend as two containers in
one Fargate task. The public ALB targets Streamlit only. In `awsvpc` mode,
containers in a task share a network stack, so Streamlit calls FastAPI at
`http://127.0.0.1:8000` without exposing port 8000 through a security group.

The module does not create container images or secret values. Before applying,
publish immutable backend and frontend images, create an XAI API key secret in
AWS Secrets Manager, and provide their URIs/ARN through a local `.tfvars` file.
Do not commit that file if it contains account-specific or sensitive values.
Use digest-pinned ECR image URIs; tag-only and placeholder values are rejected.
Follow `DEPLOYMENT.md` for the controlled publication, planning, approval,
validation, and teardown sequence.

Initial sizing is 2 vCPU, 4 GiB task memory, and 30 GiB ephemeral storage. The
validated backend used roughly 0.4--0.9 GiB at readiness; a separate model probe
raised the observed total to about 1.7 GiB. Streamlit used about 0.06 GiB. Soft
reservations assign 1.75 vCPU/3 GiB to FastAPI and 0.25 vCPU/0.5 GiB to
Streamlit, leaving 0.5 GiB shared memory headroom. Thirty GiB of ephemeral
storage accommodates the approximately 5.49 GB backend image and its unpacked
runtime layers.

Validation is local and does not create AWS resources:

```bash
terraform -chdir=infrastructure/terraform fmt -check
terraform -chdir=infrastructure/terraform init -backend=false
terraform -chdir=infrastructure/terraform validate
pytest tests/test_deployment_aws_offline.py -q
```

Do not run `terraform apply` until image publication, secret creation, HTTPS,
budget expectations, and account-specific configuration have been reviewed.
