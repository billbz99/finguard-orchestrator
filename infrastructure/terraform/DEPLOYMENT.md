# Controlled AWS deployment runbook

This runbook is a command template, not an automated deployment. Run each
mutating step only after reviewing the selected AWS account, region, expected
cost, and command arguments. Never commit credentials, real `.tfvars` files,
secret values, login tokens, account-specific image URIs, or Terraform state.

## 1. Verify identity and region

```bash
aws --version
aws sts get-caller-identity
aws configure get region
```

Confirm the returned account and principal are the intended deployment target.
Set the region for the remaining shell session and derive the registry without
hardcoding an account ID:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
```

## 2. Create or verify the xAI secret separately

Create the xAI API key secret through an approved secret-management workflow,
not Terraform variables or files. Record only its ARN for later use. Do not put
the secret value on a command line, in shell history, or in Terraform state.

## 3. Create the two private ECR repositories

The intended repositories use immutable tags, basic scan-on-push, and the
checked-in small-project lifecycle policy:

```bash
aws ecr create-repository \
  --region "$AWS_REGION" \
  --repository-name finguard-api \
  --image-tag-mutability IMMUTABLE \
  --image-scanning-configuration scanOnPush=true

aws ecr create-repository \
  --region "$AWS_REGION" \
  --repository-name finguard-ui \
  --image-tag-mutability IMMUTABLE \
  --image-scanning-configuration scanOnPush=true

aws ecr put-lifecycle-policy \
  --region "$AWS_REGION" \
  --repository-name finguard-api \
  --lifecycle-policy-text file://infrastructure/ecr-lifecycle-policy.json

aws ecr put-lifecycle-policy \
  --region "$AWS_REGION" \
  --repository-name finguard-ui \
  --lifecycle-policy-text file://infrastructure/ecr-lifecycle-policy.json
```

## 4. Authenticate Docker and publish the validated images

The ECR token flows directly to Docker and must not be printed or stored in the
repository:

```bash
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

docker tag finguard-api:deployment-v1 \
  "$ECR_REGISTRY/finguard-api:deployment-v1"
docker tag finguard-ui:deployment-v1 \
  "$ECR_REGISTRY/finguard-ui:deployment-v1"

docker push "$ECR_REGISTRY/finguard-api:deployment-v1"
docker push "$ECR_REGISTRY/finguard-ui:deployment-v1"
```

## 5. Capture immutable digests

```bash
export BACKEND_DIGEST="$(aws ecr describe-images \
  --region "$AWS_REGION" \
  --repository-name finguard-api \
  --image-ids imageTag=deployment-v1 \
  --query 'imageDetails[0].imageDigest' \
  --output text)"

export FRONTEND_DIGEST="$(aws ecr describe-images \
  --region "$AWS_REGION" \
  --repository-name finguard-ui \
  --image-ids imageTag=deployment-v1 \
  --query 'imageDetails[0].imageDigest' \
  --output text)"

export BACKEND_IMAGE_URI="$ECR_REGISTRY/finguard-api@$BACKEND_DIGEST"
export FRONTEND_IMAGE_URI="$ECR_REGISTRY/finguard-ui@$FRONTEND_DIGEST"
```

Inspect both values and confirm they end in a `sha256:` digest before using
them. Terraform rejects tags and placeholder image references.

## 6. Prepare local Terraform inputs

Copy `terraform.tfvars.example` to an ignored local `.tfvars` file and replace
every angle-bracket placeholder. Supply only the existing secret ARN; never put
the xAI secret value in Terraform configuration.

Required inputs are:

- `backend_image_uri`: digest-pinned `finguard-api` ECR URI
- `frontend_image_uri`: digest-pinned `finguard-ui` ECR URI
- `xai_secret_arn`: existing Secrets Manager secret ARN

Review optional defaults for region, name prefix, CIDRs, desired count, model,
and log retention before planning.

## 7. Initialize, validate, and review a saved plan

Use a local plan filename ending in `.tfplan`; it is ignored by Git. Treat the
plan as potentially sensitive metadata.

```bash
terraform -chdir=infrastructure/terraform init
terraform -chdir=infrastructure/terraform fmt -check
terraform -chdir=infrastructure/terraform validate
terraform -chdir=infrastructure/terraform plan \
  -var-file=local.tfvars \
  -out=deployment.tfplan
terraform -chdir=infrastructure/terraform show deployment.tfplan
```

Do not apply until the plan, account, region, costs, image digests, secret ARN,
network exposure, and IAM changes receive explicit approval.

## 8. Apply only after approval and validate runtime

After explicit approval, apply the reviewed saved plan without adding automatic
approval flags. Validate the ECS task rollout, target health, CloudWatch logs,
ALB page, Streamlit-to-FastAPI localhost path, backend `/health`, backend
`/ready`, and one deterministic audit before any provider-backed audit.

## 9. Teardown

Before teardown, confirm the Terraform workspace and AWS account again. Review
a destroy plan, obtain explicit approval, and destroy the demo infrastructure
when it is no longer needed to stop Fargate, ALB, public IPv4, logging, and
Container Insights charges. ECR repositories and the separately managed secret
are outside this Terraform module and require separate retention/deletion
decisions.
