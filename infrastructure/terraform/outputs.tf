output "alb_dns_name" {
  description = "Public DNS name for the Streamlit load balancer."
  value       = aws_lb.main.dns_name
}

output "ecs_cluster_name" {
  description = "FinGuard ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "FinGuard ECS service name."
  value       = aws_ecs_service.main.name
}

output "xai_secret_arn" {
  description = "Referenced Secrets Manager secret ARN; Terraform does not create its value."
  value       = var.xai_secret_arn
  sensitive   = true
}
