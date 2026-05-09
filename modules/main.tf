terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ---------------------------
# データソース
# ---------------------------

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ---------------------------
# IAM Role for Lambda
# ---------------------------

resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:Converse"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3vectors:QueryVectors",
          "s3vectors:GetVectors",
          "s3vectors:ListVectors"
        ]
        Resource = "*"
      }
    ]
  })
}

# ---------------------------
# Lambda Function
# ---------------------------

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda"
  output_path = "${path.module}/../.build/lambda.zip"
}

resource "aws_lambda_function" "lawsy" {
  function_name    = "${var.project_name}-api"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  handler          = "src.handler.lambda_handler"
  runtime          = "python3.12"
  role             = aws_iam_role.lambda_role.arn
  timeout          = 300
  memory_size      = 512

  environment {
    variables = {
      AWS_REGION_NAME      = var.aws_region
      MODEL_ID             = var.model_id
      EMBEDDING_MODEL_ID   = var.embedding_model_id
      VECTOR_BUCKET_ARN    = var.vector_bucket_arn
      VECTOR_INDEX_NAME    = var.vector_index_name
      TEMPERATURE          = "0.0"
      MAX_TOKENS           = "8192"
      TOP_P                = "1.0"
    }
  }

  layers = [aws_lambda_layer_version.dependencies.arn]
}

# ---------------------------
# Lambda Layer (dependencies)
# ---------------------------

resource "null_resource" "pip_install" {
  triggers = {
    requirements = filemd5("${path.module}/../lambda/requirements.txt")
  }

  provisioner "local-exec" {
    command = <<-EOT
      rm -rf ${path.module}/../.build/layer
      mkdir -p ${path.module}/../.build/layer/python
      pip install -r ${path.module}/../lambda/requirements.txt -t ${path.module}/../.build/layer/python --quiet
    EOT
  }
}

data "archive_file" "layer_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../.build/layer"
  output_path = "${path.module}/../.build/layer.zip"
  depends_on  = [null_resource.pip_install]
}

resource "aws_lambda_layer_version" "dependencies" {
  layer_name          = "${var.project_name}-dependencies"
  filename            = data.archive_file.layer_zip.output_path
  source_code_hash    = data.archive_file.layer_zip.output_base64sha256
  compatible_runtimes = ["python3.12"]
  depends_on          = [null_resource.pip_install]
}

# ---------------------------
# API Gateway (REST API)
# ---------------------------

resource "aws_api_gateway_rest_api" "lawsy" {
  name        = "${var.project_name}-api"
  description = "Lawsy 法令レポート生成 API"

  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

resource "aws_api_gateway_resource" "invoke" {
  rest_api_id = aws_api_gateway_rest_api.lawsy.id
  parent_id   = aws_api_gateway_rest_api.lawsy.root_resource_id
  path_part   = "invoke"
}

resource "aws_api_gateway_method" "invoke_post" {
  rest_api_id      = aws_api_gateway_rest_api.lawsy.id
  resource_id      = aws_api_gateway_resource.invoke.id
  http_method      = "POST"
  authorization    = "NONE"
  api_key_required = true
}

resource "aws_api_gateway_method" "invoke_options" {
  rest_api_id   = aws_api_gateway_rest_api.lawsy.id
  resource_id   = aws_api_gateway_resource.invoke.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "invoke_post" {
  rest_api_id             = aws_api_gateway_rest_api.lawsy.id
  resource_id             = aws_api_gateway_resource.invoke.id
  http_method             = aws_api_gateway_method.invoke_post.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.lawsy.invoke_arn
}

resource "aws_api_gateway_integration" "invoke_options" {
  rest_api_id = aws_api_gateway_rest_api.lawsy.id
  resource_id = aws_api_gateway_resource.invoke.id
  http_method = aws_api_gateway_method.invoke_options.http_method
  type        = "MOCK"

  request_templates = {
    "application/json" = "{\"statusCode\": 200}"
  }
}

resource "aws_api_gateway_method_response" "options_200" {
  rest_api_id = aws_api_gateway_rest_api.lawsy.id
  resource_id = aws_api_gateway_resource.invoke.id
  http_method = aws_api_gateway_method.invoke_options.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_integration_response" "options_200" {
  rest_api_id = aws_api_gateway_rest_api.lawsy.id
  resource_id = aws_api_gateway_resource.invoke.id
  http_method = aws_api_gateway_method.invoke_options.http_method
  status_code = aws_api_gateway_method_response.options_200.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,x-api-key'"
    "method.response.header.Access-Control-Allow-Methods" = "'POST,OPTIONS'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
}

# ---------------------------
# API Gateway Deployment
# ---------------------------

resource "aws_api_gateway_deployment" "lawsy" {
  rest_api_id = aws_api_gateway_rest_api.lawsy.id

  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.invoke.id,
      aws_api_gateway_method.invoke_post.id,
      aws_api_gateway_integration.invoke_post.id,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [
    aws_api_gateway_integration.invoke_post,
    aws_api_gateway_integration.invoke_options,
  ]
}

resource "aws_api_gateway_stage" "prod" {
  deployment_id = aws_api_gateway_deployment.lawsy.id
  rest_api_id   = aws_api_gateway_rest_api.lawsy.id
  stage_name    = "prod"
}

# ---------------------------
# API Key & Usage Plan
# ---------------------------

resource "aws_api_gateway_api_key" "lawsy" {
  name    = "${var.project_name}-api-key"
  enabled = true
}

resource "aws_api_gateway_usage_plan" "lawsy" {
  name = "${var.project_name}-usage-plan"

  api_stages {
    api_id = aws_api_gateway_rest_api.lawsy.id
    stage  = aws_api_gateway_stage.prod.stage_name
  }

  throttle_settings {
    burst_limit = 10
    rate_limit  = 5
  }
}

resource "aws_api_gateway_usage_plan_key" "lawsy" {
  key_id        = aws_api_gateway_api_key.lawsy.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.lawsy.id
}

# ---------------------------
# Lambda Permission for API Gateway
# ---------------------------

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lawsy.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.lawsy.execution_arn}/*/*"
}

# ---------------------------
# WAF (IP制限) - オプション
# ---------------------------

resource "aws_wafv2_ip_set" "allowed_ips" {
  count              = length(var.allowed_ip_addresses) > 0 ? 1 : 0
  name               = "${var.project_name}-allowed-ips"
  scope              = "REGIONAL"
  ip_address_version = "IPV4"
  addresses          = var.allowed_ip_addresses
}

resource "aws_wafv2_web_acl" "api_waf" {
  count = length(var.allowed_ip_addresses) > 0 ? 1 : 0
  name  = "${var.project_name}-waf"
  scope = "REGIONAL"

  default_action {
    block {}
  }

  rule {
    name     = "allow-specific-ips"
    priority = 1

    action {
      allow {}
    }

    statement {
      ip_set_reference_statement {
        arn = aws_wafv2_ip_set.allowed_ips[0].arn
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.project_name}-allowed-ips"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project_name}-waf"
    sampled_requests_enabled   = true
  }
}

resource "aws_wafv2_web_acl_association" "api_waf" {
  count        = length(var.allowed_ip_addresses) > 0 ? 1 : 0
  resource_arn = aws_api_gateway_stage.prod.arn
  web_acl_arn  = aws_wafv2_web_acl.api_waf[0].arn
}
