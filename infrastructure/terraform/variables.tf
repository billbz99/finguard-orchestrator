variable "aws_region" {
  description = "AWS region for the FinGuard demo deployment."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix applied to FinGuard infrastructure names."
  type        = string
  default     = "finguard-demo"
}

variable "vpc_cidr" {
  description = "CIDR block for the demo VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "Two public subnet CIDRs used by the ALB and Fargate tasks."
  type        = list(string)
  default     = ["10.42.0.0/24", "10.42.1.0/24"]

  validation {
    condition     = length(var.public_subnet_cidrs) == 2
    error_message = "Exactly two public subnet CIDRs are required."
  }
}

variable "backend_image_uri" {
  description = "Digest-pinned private ECR URI for the packaged FastAPI backend."
  type        = string

  validation {
    condition = can(regex(
      "^[0-9]{12}\\.dkr\\.ecr\\.[a-z0-9-]+\\.amazonaws\\.com/finguard-api@sha256:[0-9a-f]{64}$",
      var.backend_image_uri,
    )) && !startswith(var.backend_image_uri, "000000000000.")
    error_message = "backend_image_uri must be a non-placeholder, digest-pinned private ECR finguard-api URI."
  }
}

variable "frontend_image_uri" {
  description = "Digest-pinned private ECR URI for the thin Streamlit frontend."
  type        = string

  validation {
    condition = can(regex(
      "^[0-9]{12}\\.dkr\\.ecr\\.[a-z0-9-]+\\.amazonaws\\.com/finguard-ui@sha256:[0-9a-f]{64}$",
      var.frontend_image_uri,
    )) && !startswith(var.frontend_image_uri, "000000000000.")
    error_message = "frontend_image_uri must be a non-placeholder, digest-pinned private ECR finguard-ui URI."
  }
}

variable "xai_secret_arn" {
  description = "ARN of an existing Secrets Manager secret containing the XAI API key."
  type        = string
  sensitive   = true

  validation {
    condition     = can(regex("^arn:[^:]+:secretsmanager:[^:]+:[0-9]{12}:secret:", var.xai_secret_arn))
    error_message = "xai_secret_arn must be a Secrets Manager secret ARN."
  }
}

variable "xai_model" {
  description = "Configured xAI model name passed to the backend."
  type        = string
  default     = "grok-4.3"
}

variable "desired_count" {
  description = "Number of two-container FinGuard tasks to run."
  type        = number
  default     = 1

  validation {
    condition     = var.desired_count >= 1
    error_message = "desired_count must be at least one."
  }
}

variable "log_retention_days" {
  description = "CloudWatch log retention for both application containers."
  type        = number
  default     = 14
}
