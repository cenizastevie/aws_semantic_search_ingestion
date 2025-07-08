# aws_semantic_search_ingestion
Repository of my semantic search app ingestion pipleline.

```
aws cloudformation deploy --template-file 1_fargate_task/1_fargate_task_formation.yaml --stack-name assi-fargate-task-main --capabilities CAPABILITY_NAMED_IAM --parameter-overrides BranchName=main
```
