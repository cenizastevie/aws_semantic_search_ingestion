@echo off
REM Deploy the CI/CD CloudFormation stack using parameters from .env

REM Prompt for branch name
set /p BRANCH_NAME=Enter the branch name for this stack (default: main): 
if "%BRANCH_NAME%"=="" set BRANCH_NAME=main

REM Load environment variables from .env
for /f "usebackq tokens=1,2 delims==" %%A in (".env") do set %%A=%%B

REM Append branch name to stack name
set STACK_NAME=%STACK_NAME%-%BRANCH_NAME%

REM Deploy the stack
aws cloudformation deploy ^
  --template-file 0_ci_cd_formation.yaml ^
  --stack-name %STACK_NAME% ^
  --region %REGION% ^
  --capabilities CAPABILITY_NAMED_IAM ^
  --parameter-overrides ^
    ConnectionID=%CONNECTION_ID% ^
    GitHubUser=%GITHUB_USER% ^
    GitHubRepo=%GITHUB_REPO% ^
    BranchName=%BRANCH_NAME% ^
    AWSAccountId=%AWS_ACCOUNT_ID% ^
    VpcId=%VPC_ID% ^
    SubnetIds="%SUBNET_IDS%"
