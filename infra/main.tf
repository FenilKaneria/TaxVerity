# Step 17.5 (ADR-076, ADR-111). Minimal Terraform describing the deployed
# infra: ECR repo, Lambda container function, function URL (RESPONSE_STREAM,
# rule 04), IAM execution role, CloudWatch log group, Secrets Manager entries.
# No VPC (ADR-076 — a NAT gateway would be needed for every vendor call).
# Vercel (17.6) is not here; it's configured through its own CLI/UI.
#
# These resources already exist (created imperatively via the AWS CLI while
# building this phase). This file documents them as IaC for reproducibility
# on a fresh account; it has not been `terraform apply`'d against the live
# resources (would need `terraform import` first — out of SIMPLIFIED scope,
# ADR-110).

terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "ap-south-1"
}

variable "image_tag" {
  description = "ECR image tag or digest to deploy"
  type        = string
  default     = "latest"
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

# --- ECR ---

resource "aws_ecr_repository" "taxverity" {
  name                 = "taxverity"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# --- IAM execution role ---

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = "taxverity-lambda-exec"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "lambda_basic_exec" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# --- Secrets Manager ---
# Values are set out of band (aws secretsmanager put-secret-value / the
# console) — never committed to this file or to git.

resource "aws_secretsmanager_secret" "groq_api_key" {
  name = "taxverity/groq-api-key"
}

resource "aws_secretsmanager_secret" "jina_api_key" {
  name = "taxverity/jina-api-key"
}

resource "aws_secretsmanager_secret" "jwt_secret" {
  name = "taxverity/jwt-secret"
}

resource "aws_secretsmanager_secret" "database_url" {
  name = "taxverity/database-url"
}

data "aws_secretsmanager_secret_version" "groq_api_key" {
  secret_id = aws_secretsmanager_secret.groq_api_key.id
}

data "aws_secretsmanager_secret_version" "jina_api_key" {
  secret_id = aws_secretsmanager_secret.jina_api_key.id
}

data "aws_secretsmanager_secret_version" "jwt_secret" {
  secret_id = aws_secretsmanager_secret.jwt_secret.id
}

data "aws_secretsmanager_secret_version" "database_url" {
  secret_id = aws_secretsmanager_secret.database_url.id
}

# --- CloudWatch log group ---
# Created automatically by AWSLambdaBasicExecutionRole on first invoke, but
# declared explicitly so retention is controlled rather than left at
# "never expire".

resource "aws_cloudwatch_log_group" "taxverity" {
  name              = "/aws/lambda/taxverity"
  retention_in_days = 30
}

# --- Lambda function ---
# No VPC config (ADR-076): every dependency (Supabase, Jina, Groq, Gemini,
# Langfuse) is reached over the public internet, and a VPC would need a NAT
# gateway for each of those calls to leave.

resource "aws_lambda_function" "taxverity" {
  function_name = "taxverity"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.taxverity.repository_url}:${var.image_tag}"
  architectures = ["x86_64"]
  memory_size   = 2048
  timeout       = 120

  environment {
    variables = {
      TAXVERITY_GROQ_API_KEY              = data.aws_secretsmanager_secret_version.groq_api_key.secret_string
      TAXVERITY_JINA_API_KEY              = data.aws_secretsmanager_secret_version.jina_api_key.secret_string
      TAXVERITY_JWT_SECRET                = data.aws_secretsmanager_secret_version.jwt_secret.secret_string
      TAXVERITY_DATABASE_URL              = data.aws_secretsmanager_secret_version.database_url.secret_string
      TAXVERITY_SERVING_CORPUS_VERSION    = var.serving_corpus_version
      TAXVERITY_SERVING_EMBEDDING_SET_ID  = var.serving_embedding_set_id
    }
  }

  depends_on = [aws_cloudwatch_log_group.taxverity]
}

variable "serving_corpus_version" {
  type = string
}

variable "serving_embedding_set_id" {
  type    = string
  default = "1"
}

# --- Function URL ---
# RESPONSE_STREAM invoke mode is required, not an optimisation: claim-level
# SSE (rule 04) must reach the client incrementally. auth_type NONE because
# the Vercel frontend calls this directly with `fetch`, not SigV4.

resource "aws_lambda_function_url" "taxverity" {
  function_name      = aws_lambda_function.taxverity.function_name
  authorization_type = "NONE"
  invoke_mode        = "RESPONSE_STREAM"

  cors {
    allow_origins = var.cors_allow_origins
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["content-type", "authorization"]
    allow_credentials = true
    max_age       = 86400
  }
}

variable "cors_allow_origins" {
  description = "Vercel frontend origin(s), set once the frontend is deployed (Step 17.6)"
  type        = list(string)
  default     = []
}

resource "aws_lambda_permission" "function_url_public" {
  statement_id           = "FunctionURLAllowPublicAccess"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.taxverity.function_name
  principal               = "*"
  function_url_auth_type = "NONE"
}

output "function_url" {
  value = aws_lambda_function_url.taxverity.function_url
}

output "ecr_repository_url" {
  value = aws_ecr_repository.taxverity.repository_url
}
