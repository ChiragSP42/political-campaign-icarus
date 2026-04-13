import * as cdk from 'aws-cdk-lib/core';
import * as dotenv from 'dotenv';
import * as path from 'path';
import { Construct } from 'constructs';
import * as aws_iam from 'aws-cdk-lib/aws-iam';
import * as aws_lambda from 'aws-cdk-lib/aws-lambda';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
import * as aws_apigateway from 'aws-cdk-lib/aws-apigateway';
import * as aws_cognito from 'aws-cdk-lib/aws-cognito';
import * as aws_s3 from 'aws-cdk-lib/aws-s3';
import * as aws_s3_deployment from 'aws-cdk-lib/aws-s3-deployment';
import * as aws_dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as aws_lambda_event_sources from 'aws-cdk-lib/aws-lambda-event-sources';
dotenv.config({
  path: path.join(__dirname, "../../../.env")
})

export class IcarusDannerInfraStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // The code that defines your stack goes here

    // example resource
    // const queue = new sqs.Queue(this, 'InfraQueue', {
    //   visibilityTimeout: cdk.Duration.seconds(300)
    // });

    // =====================================================
    // 1. COGNITO USER POOL (Authentication)
    // =====================================================

    const userPool = new aws_cognito.UserPool(this, 'UserPool', {
      userPoolName: 'icarus-users',
      signInAliases: { email: true },
      autoVerify: { email: true },
      passwordPolicy: {
        minLength: 8,
        requireLowercase: true,
        requireUppercase: true,
        requireDigits: true,
        requireSymbols: false
      },
      accountRecovery: aws_cognito.AccountRecovery.EMAIL_ONLY,
      email: aws_cognito.UserPoolEmail.withCognito(),
      selfSignUpEnabled: true,
      standardAttributes: {
        email: { required: true, mutable: true }
      },
      removalPolicy: cdk.RemovalPolicy.DESTROY
    })

    const userPoolClient = userPool.addClient('IcarusWebClient', {
      userPoolClientName: 'icarus-web',
      authFlows: {
        userPassword: true,
        userSrp: true,
        custom: false,
        adminUserPassword: false
      },
      oAuth: {
        flows: { authorizationCodeGrant: true },
        scopes: [
          aws_cognito.OAuthScope.EMAIL,
          aws_cognito.OAuthScope.OPENID
        ]
      },
      preventUserExistenceErrors: true
    })

    // =====================================================
    // 2. S3 BUCKETS
    // =====================================================

    // Store election data - NEED TO DEPRECATE
    const election_data_bucket = new aws_s3.Bucket(this, 'ElectionDataBucket', {
      bucketName: `icarus-election-data-${this.account}`,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true
    })

    // Store prompts and 
    const prompt_bucket = new aws_s3.Bucket(this, 'PromptBucket', {
      bucketName: `prompt-bucket-${this.account}`,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true
    })

    // Store relevant election rules and regulations
    const election_laws = new aws_s3.Bucket(this, 'ElectionLawsBucket', {
      bucketName: `election-laws-${this.account}`,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true
    })

    // Load local prompt files to prompt-bucket
    new aws_s3_deployment.BucketDeployment(this, 'DeployFiles', {
      sources: [
        aws_s3_deployment.Source.asset(path.join(__dirname, '../../local-files'))
      ],
      destinationBucket: prompt_bucket
    })

    // =====================================================
    // 3. IAM ROLE FOR LAMBDA FUNCTIONS
    // =====================================================

    const bedrock_policy = new aws_iam.Policy(this, 'bedrock-policy', {
      policyName: 'lambda-bedrock-policy',
      statements: [
        new aws_iam.PolicyStatement({
          actions: [
            'bedrock:InvokeModel',
            'bedrock:InvokeModelWithResponseStream',
            'bedrock-agent-runtime:Retrieve',
            'bedrock:Retrieve'
          ],
          resources: ['*']
        })
      ]
    })

    const lambda_role = new aws_iam.Role(this, 'lambda-role', {
      roleName: 'lambda-role',
      assumedBy: new aws_iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'Lambda service role to provide access to Bedrock, S3',
      managedPolicies: [
        aws_iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
        aws_iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaRole')
      ],
      inlinePolicies: {
        'icarus-s3-policy': new aws_iam.PolicyDocument({
          statements: [
            new aws_iam.PolicyStatement({
              effect: aws_iam.Effect.ALLOW,
              actions: ["s3:*",
                "s3-object-lambda:*"],
              resources: ['*']
            })
          ]
        })
      }
    })

    bedrock_policy.attachToRole(lambda_role)

    // =====================================================
    // 3b. DYNAMODB TABLE FOR CHAT HISTORY
    // =====================================================

    const chatHistoryTable = new aws_dynamodb.Table(this, 'ChatHistoryTable', {
      tableName: `chat-history-${this.account}`,
      partitionKey: { name: 'chatId', type: aws_dynamodb.AttributeType.STRING },
      sortKey: { name: 'timestamp', type: aws_dynamodb.AttributeType.STRING },
      billingMode: aws_dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    })

    chatHistoryTable.addGlobalSecondaryIndex({
      indexName: 'userId-index',
      partitionKey: { name: 'userId', type: aws_dynamodb.AttributeType.STRING },
      sortKey: { name: 'createdAt', type: aws_dynamodb.AttributeType.STRING },
    })

    chatHistoryTable.grantReadWriteData(lambda_role)

    // =====================================================
    // 3c. DYNAMODB TABLE FOR USERS
    // =====================================================

    const mainTable = new aws_dynamodb.Table(this, 'MainTable', {
      tableName: `main-${this.account}`,
      partitionKey: { name: 'userId', type: aws_dynamodb.AttributeType.STRING },
      sortKey: { name: 'SK', type: aws_dynamodb.AttributeType.STRING },
      billingMode: aws_dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    })

    mainTable.grantReadWriteData(lambda_role)

    // =====================================================
    // 3d. DYNAMODB TABLE FOR QUESTIONNAIRE
    // =====================================================

    const questionnaireTable = new aws_dynamodb.Table(this, 'QuestionnaireTable', {
      tableName: `questionnaire-${this.account}`,
      partitionKey: { name: 'userId', type: aws_dynamodb.AttributeType.STRING },
      sortKey: { name: 'SK', type: aws_dynamodb.AttributeType.STRING },
      billingMode: aws_dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      stream: aws_dynamodb.StreamViewType.NEW_IMAGE,
    })

    questionnaireTable.grantReadWriteData(lambda_role)

    // =====================================================
    // 4. LAMBDA FUNCTIONS
    // =====================================================

    // Check questionnaire Lambda
    const check_questionnaire_lambda = new aws_lambda.Function(this, 'check-questionnaire-lambda', {
      functionName: 'check-questionnaire-lambda',
      description: 'This function checks if a user has already completed the questionnaire',
      code: aws_lambda.Code.fromAsset(path.join(__dirname, '../../services/lambdas/')),
      handler: 'check_questionnaire_lambda.lambda_handler',
      runtime: aws_lambda.Runtime.PYTHON_3_13,
      timeout: cdk.Duration.minutes(15),
      memorySize: 1024,
      role: lambda_role,
      environment: {
        S3_QUESTIONNAIRES: process.env.S3_QUESTIONNAIRES || 'icarus-questionnaires'
      }
    })

    // Save questionnaire Lambda
    const save_questionnaire_lambda = new aws_lambda.Function(this, 'save-questionnaire-lambda', {
      functionName: 'save-questionnaire-lambda',
      description: 'This function stores questionnaire answers in S3',
      code: aws_lambda.Code.fromAsset(path.join(__dirname, '../../services/lambdas/')),
      handler: 'save_questionnaire_lambda.lambda_handler',
      runtime: aws_lambda.Runtime.PYTHON_3_13,
      timeout: cdk.Duration.minutes(15),
      memorySize: 1024,
      role: lambda_role,
      environment: {
        QUESTIONNAIRE_TABLE_NAME: questionnaireTable.tableName
      }
    })

    // Main chatbot lambda
    const chatbot_lambda = new aws_lambda.Function(this, 'chatbot-lambda', {
      functionName: 'chatbot-lambda',
      description: 'Main chatbot functionality',
      code: aws_lambda.Code.fromAsset(path.join(__dirname, '../../services/lambdas/')),
      handler: 'chatbot_lambda.lambda_handler',
      runtime: aws_lambda.Runtime.PYTHON_3_13,
      timeout: cdk.Duration.minutes(15),
      memorySize: 2048,
      role: lambda_role,
      ephemeralStorageSize: cdk.Size.mebibytes(1024),
      environment: {
        S3_GENERATED_INSIGHTS: process.env.S3_GENERATED_INSIGHTS || 'generated-insights',
        S3_QUESTIONNAIRES: process.env.S3_QUESTIONNAIRES || 'icarus-questionnaires',
        CHATBOT_PROMPT: process.env.CHATBOT_PROMPT || 'campaign_advisor_prompt.md',
        PROMPT_BUCKET: process.env.PROMPT_BUCKET || 'prompt-bucket',
        MODEL_ID: process.env.MODEL_ID || 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        KB_ID: process.env.KB_ID || 'AXGUO9J7Q1',
        CHAT_HISTORY_TABLE: chatHistoryTable.tableName,
      }
    })

    // Generate insights lambda
    const generate_insights_lambda = new aws_lambda.Function(this, 'generate-insights-lambda', {
      functionName: 'generate-insights-lambda',
      description: 'Generate insights lambda',
      code: aws_lambda.Code.fromAsset(path.join(__dirname, '../../services/lambdas/')),
      handler: 'generate_insights_lambda.lambda_handler',
      runtime: aws_lambda.Runtime.PYTHON_3_13,
      timeout: cdk.Duration.minutes(15),
      memorySize: 2048,
      role: lambda_role,
      ephemeralStorageSize: cdk.Size.mebibytes(1024),
      environment: {
        INSIGHTS_GENERALISED_PROMPT: process.env.INSIGHTS_GENERALISED_PROMPT || 'campaign_insights_prompt.md',
        KB_INSIGHTS_PROMPT: process.env.KB_INSIGHTS_PROMPT || 'kb_election_laws_prompt.md',
        PROMPT_BUCKET: process.env.PROMPT_BUCKET || 'prompt-bucket',
        ELECTION_CYCLE_FILENAME: process.env.ELECTION_CYCLE_FILENAME || 'election_cycles.json',
        MODEL_ID: process.env.MODEL_ID || 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        KB_ID: process.env.KB_ID || 'AXGUO9J7Q1',
        MAIN_TABLE_NAME: mainTable.tableName,
        QUESTIONNAIRE_TABLE_NAME: questionnaireTable.tableName
      }
    })

    // Trigger generate-insights-lambda when questionnaire is saved to DynamoDB
    generate_insights_lambda.addEventSource(
      new aws_lambda_event_sources.DynamoEventSource(questionnaireTable, {
        startingPosition: aws_lambda.StartingPosition.LATEST,
        batchSize: 1,
        retryAttempts: 2,
      })
    )

    // Trigger chatbot-lambda
    const trigger_chatbot_lambda = new aws_lambda.Function(this, 'trigger-chatbot-lambda', {
      functionName: 'trigger-chatbot-lambda',
      description: 'Lambda that will start async process of chatbot response',
      code: aws_lambda.Code.fromAsset(path.join(__dirname, '../../services/lambdas/')),
      handler: 'trigger_chatbot.lambda_handler',
      runtime: aws_lambda.Runtime.PYTHON_3_13,
      timeout: cdk.Duration.minutes(15),
      memorySize: 1024,
      ephemeralStorageSize: cdk.Size.mebibytes(1024),
      role: lambda_role,
      environment: {
        CHATBOT_LAMBDA_NAME: chatbot_lambda.functionName,
        CHAT_HISTORY_TABLE: chatHistoryTable.tableName,
      }
    })

    // Check chatbot response
    const check_chatbot_response_lambda = new aws_lambda.Function(this, 'check-llm-response-lambda', {
      functionName: 'check-llm-response-lambda',
      description: 'Check is chatbot-lambda stored response in S3',
      code: aws_lambda.Code.fromAsset(path.join(__dirname, "../../services/lambdas/")),
      handler: 'check_LLM_response_lambda.lambda_handler',
      runtime: aws_lambda.Runtime.PYTHON_3_13,
      timeout: cdk.Duration.minutes(15),
      memorySize: 1024,
      role: lambda_role,
      ephemeralStorageSize: cdk.Size.mebibytes(1024),
      environment: {
        S3_RESPONSES: process.env.S3_RESPONSES || 'chatbot-responses',
        CHAT_HISTORY_TABLE: chatHistoryTable.tableName,
      }
    })

    // Session manager lambda
    const session_manager_lambda = new aws_lambda.Function(this, 'session-manager-lambda', {
      functionName: 'session-manager-lambda',
      description: 'Manages chat session CRUD operations',
      code: aws_lambda.Code.fromAsset(path.join(__dirname, '../../services/lambdas/')),
      handler: 'session_manager_lambda.lambda_handler',
      runtime: aws_lambda.Runtime.PYTHON_3_13,
      timeout: cdk.Duration.minutes(15),
      memorySize: 1024,
      role: lambda_role,
      environment: {
        CHAT_HISTORY_TABLE: chatHistoryTable.tableName,
      }
    })

    // =====================================================
    // 5. API GATEWAY
    // =====================================================

    const api = new aws_apigateway.RestApi(this, 'IcarusApi', {
      restApiName: 'icarus-api',
      description: 'Project Icarus Campaign Chatbot API',
      defaultCorsPreflightOptions: {
        allowOrigins: aws_apigateway.Cors.ALL_ORIGINS,
        allowMethods: aws_apigateway.Cors.ALL_METHODS,
        allowHeaders: [
          'Content-Type',
          'X-Amz-Date',
          'Authorization',
          'X-Api-Key',
          'X-Amz-Security-Token'
        ]
      },
      deployOptions: {
        stageName: 'dev',
        throttlingRateLimit: 1000,
        throttlingBurstLimit: 2000,
        loggingLevel: aws_apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: true
      }
    })

    // Lambda Integrations (proxy: true avoids circular dependencies)
    const check_questionnaire_integration = new aws_apigateway.LambdaIntegration(check_questionnaire_lambda, { proxy: true })
    const save_questionnaire_integration = new aws_apigateway.LambdaIntegration(save_questionnaire_lambda, { proxy: true })
    const chatbot_integration = new aws_apigateway.LambdaIntegration(trigger_chatbot_lambda, { proxy: true })
    const check_chatbot_response_integration = new aws_apigateway.LambdaIntegration(check_chatbot_response_lambda, {proxy: true})

    // API Resources
    const check_resource = api.root.addResource('check-questionnaire')
    check_resource.addMethod('GET', check_questionnaire_integration)

    const save_resource = api.root.addResource('save-questionnaire')
    save_resource.addMethod('POST', save_questionnaire_integration)

    const chatbot_resource = api.root.addResource('chat')
    chatbot_resource.addMethod('POST', chatbot_integration)

    const check_chatbot_resource = api.root.addResource('check-response')
    check_chatbot_resource.addMethod('GET', check_chatbot_response_integration)

    // Session management routes
    const session_manager_integration = new aws_apigateway.LambdaIntegration(session_manager_lambda, { proxy: true })

    const sessions_resource = api.root.addResource('sessions')
    sessions_resource.addMethod('GET', session_manager_integration)
    sessions_resource.addMethod('DELETE', session_manager_integration)

    const sessions_messages_resource = sessions_resource.addResource('messages')
    sessions_messages_resource.addMethod('GET', session_manager_integration)

    // =====================================================
    // 6. OUTPUTS
    // =====================================================

    new cdk.CfnOutput(this, 'UserPoolId', {
      value: userPool.userPoolId,
      description: 'Cognito User Pool ID',
      exportName: 'IcarusUserPoolId'
    })

    new cdk.CfnOutput(this, 'UserPoolClientId', {
      value: userPoolClient.userPoolClientId,
      description: 'Cognito User Pool Client ID',
      exportName: 'IcarusUserPoolClientId'
    })

    new cdk.CfnOutput(this, 'ApiEndpoint', {
      value: api.url,
      description: 'API Gateway endpoint',
      exportName: 'IcarusApiEndpoint'
    })

  }
}
