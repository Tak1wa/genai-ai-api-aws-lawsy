terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Terraform state の保存先（必要に応じて変更）
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "lawsy-aws/terraform.tfstate"
  #   region = "ap-northeast-1"
  # }
}

provider "aws" {
  region = var.aws_region
}

module "lawsy" {
  source = "../../modules"

  aws_region           = var.aws_region
  project_name         = var.project_name
  vector_bucket_arn    = var.vector_bucket_arn
  vector_index_name    = var.vector_index_name
  model_id             = var.model_id
  embedding_model_id   = var.embedding_model_id
  allowed_ip_addresses = var.allowed_ip_addresses
}

output "api_endpoint" {
  description = "API Gateway エンドポイント URL"
  value       = module.lawsy.api_endpoint
}

output "api_key_value" {
  description = "API Key の値"
  value       = module.lawsy.api_key_value
  sensitive   = true
}

output "lambda_function_name" {
  description = "Lambda 関数名"
  value       = module.lawsy.lambda_function_name
}
