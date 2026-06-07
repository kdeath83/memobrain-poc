import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigw from 'aws-cdk-lib/aws-apigateway';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export interface MemoBrainStackProps extends cdk.StackProps {
  /** LLM API key (Fireworks or OpenAI). Stored as env var; use Secrets Manager in production. */
  readonly llmApiKey?: string;
  /** LLM base URL (default: Fireworks) */
  readonly llmBaseUrl?: string;
  /** Model name (default: llama-v3p1-70b-instruct) */
  readonly modelName?: string;
  /** Lambda memory size in MB (default: 1024) */
  readonly lambdaMemory?: number;
  /** Lambda timeout in seconds (default: 60) */
  readonly lambdaTimeout?: number;
}

export class MemoBrainStack extends cdk.Stack {
  /** API Gateway endpoint URL */
  public readonly apiEndpoint: string;

  constructor(scope: Construct, id: string, props: MemoBrainStackProps = {}) {
    super(scope, id, props);

    const llmApiKey = props.llmApiKey || process.env.FIREWORKS_API_KEY || process.env.OPENAI_API_KEY || '';
    const llmBaseUrl = props.llmBaseUrl || process.env.OPENAI_BASE_URL || 'https://api.fireworks.ai/inference/v1';
    const modelName = props.modelName || process.env.MEMOBRAIN_MODEL || 'accounts/fireworks/models/llama-v3p1-70b-instruct';
    const lambdaMemory = props.lambdaMemory || 1024;
    const lambdaTimeout = props.lambdaTimeout || 60;

    // Lambda function
    const fn = new lambda.Function(this, 'MemoBrainLambda', {
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('../lambda'),
      memorySize: lambdaMemory,
      timeout: cdk.Duration.seconds(lambdaTimeout),
      environment: {
        FIREWORKS_API_KEY: llmApiKey,
        OPENAI_BASE_URL: llmBaseUrl,
        MEMOBRAIN_MODEL: modelName,
        PYTHONUNBUFFERED: '1',
      },
      logRetention: logs.RetentionDays.ONE_WEEK,
      architecture: lambda.Architecture.ARM_64,
    });

    // API Gateway
    const api = new apigw.RestApi(this, 'MemoBrainApi', {
      restApiName: 'memobrain-api',
      description: 'MemoBrain agentic reasoning API',
      deployOptions: {
        stageName: 'prod',
        throttlingRateLimit: 10,
        throttlingBurstLimit: 20,
      },
      defaultCorsPreflightOptions: {
        allowOrigins: apigw.Cors.ALL_ORIGINS,
        allowMethods: ['POST', 'OPTIONS'],
        allowHeaders: ['Content-Type', 'Authorization'],
      },
    });

    // /solve endpoint
    const solve = api.root.addResource('solve');
    solve.addMethod('POST', new apigw.LambdaIntegration(fn, {
      proxy: true,
    }), {
      methodResponses: [
        { statusCode: '200' },
        { statusCode: '400' },
        { statusCode: '500' },
      ],
    });

    // /health endpoint
    const health = api.root.addResource('health');
    health.addMethod('GET', new apigw.LambdaIntegration(fn, {
      proxy: true,
    }));

    // Output
    this.apiEndpoint = api.url;
    new cdk.CfnOutput(this, 'ApiEndpoint', {
      value: api.url,
      description: 'MemoBrain API Gateway endpoint',
    });

    new cdk.CfnOutput(this, 'SolveEndpoint', {
      value: `${api.url}solve`,
      description: 'POST /solve endpoint',
    });
  }
}
