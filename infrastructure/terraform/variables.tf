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
  description = "Immutable container image URI for the packaged FastAPI backend."
  type        = string
}

variable "frontend_image_uri" {
  description = "Immutable container image URI for the thin Streamlit frontend."
  type        = string
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
