@echo off
REM Load environment variables from .env file in sibling folder
for /f "usebackq tokens=1,2 delims==" %%A in ("..\0_ci_cd_formation\.env") do (
    set "%%A=%%B"
)

REM Deploy the CloudFormation stack for ECS Task, Lambda, S3, and SQS
aws cloudformation deploy ^
  --template-file 1_ecs_task_and_lambda.yaml ^
  --stack-name semantic-search-ecs-%BRANCH_NAME% ^
  --capabilities CAPABILITY_NAMED_IAM ^
  --parameter-overrides ^
    VpcId=%VPC_ID% ^
    SubnetIds="%SUBNET_IDS%" ^
    BranchName=%BRANCH_NAME%

pause