#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { MemoBrainStack } from '../stack';

const app = new cdk.App();
new MemoBrainStack(app, 'MemoBrainStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT || process.env.AWS_ACCOUNT_ID,
    region: process.env.CDK_DEFAULT_REGION || 'us-east-1',
  },
  description: 'MemoBrain PoC: Executive Memory as an Agentic Brain for Reasoning',
});
