# AWS 部署说明

这个应用推荐先用 **AWS Lambda Function URL** 部署成一个带界面的单函数应用：

- `GET /` 返回 HTML 页面
- `POST /api/calculate` 执行显存计算
- `POST /api/extract-hf-config` 从 HuggingFace `config.json` 提取模型参数
- `POST /api/calculate-markdown` 返回 Markdown 结果

计算逻辑在 `model_memory_core.py` 里，`aws_lambda_handler.py` 只做 Lambda HTTP 适配，HTML 页面复用 `model_memory_http_server.py` 中的 `INDEX_HTML`。

## 1. 本地打包

```bash
zip -r memory-calculator-lambda.zip \
  aws_lambda_handler.py \
  model_memory_core.py \
  model_memory_http_server.py
```

当前代码只依赖 Python 标准库，不需要安装第三方依赖。

## 2. 创建 IAM Role

```bash
aws iam create-role \
  --role-name memory-calculator-lambda-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name memory-calculator-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

记下 Role ARN，后续创建函数会用到。

## 3. 创建 Lambda 函数

```bash
aws lambda create-function \
  --function-name model-memory-calculator \
  --runtime python3.11 \
  --handler aws_lambda_handler.lambda_handler \
  --zip-file fileb://memory-calculator-lambda.zip \
  --role arn:aws:iam::<account-id>:role/memory-calculator-lambda-role \
  --timeout 30 \
  --memory-size 512
```

## 4. 开启 Function URL

```bash
aws lambda create-function-url-config \
  --function-name model-memory-calculator \
  --auth-type NONE \
  --cors '{
    "AllowOrigins": ["*"],
    "AllowMethods": ["GET", "POST", "OPTIONS"],
    "AllowHeaders": ["content-type"]
  }'
```

查看 URL：

```bash
aws lambda get-function-url-config \
  --function-name model-memory-calculator
```

浏览器打开返回的 `FunctionUrl` 即可访问页面。

## 5. 更新代码

```bash
zip -r memory-calculator-lambda.zip \
  aws_lambda_handler.py \
  model_memory_core.py \
  model_memory_http_server.py

aws lambda update-function-code \
  --function-name model-memory-calculator \
  --zip-file fileb://memory-calculator-lambda.zip
```

## 后续增强

如果要给多人长期使用，建议升级为：

```text
CloudFront
    |
    +-- /              -> Lambda Function URL 或 S3 静态页面
    +-- /api/*         -> Lambda Function URL
```

如果需要访问控制，可以把 Function URL 的 `auth-type` 改成 `AWS_IAM`，或者改用 API Gateway 接入鉴权、限流和访问日志。
