#!/usr/bin/env bash

source .env

############################################################################################################
# Create Data Science Project
params=""
LLM_PROJECT_NAME=${LLM_PROJECT_NAME:-"llm"}
[ -n "${LLM_PROJECT_NAME}" ] && params="${params} -p PROJECT_NAME=${LLM_PROJECT_NAME}"
oc process ${params} -f templates/llms/llm_ds_project.yaml | oc apply  -f -



############################################################################################################
# Create S3 Data Connection
params=""

if [ -n "${AWS_S3_ENDPOINT}" ]; then
    params="${params} -p AWS_S3_ENDPOINT=${AWS_S3_ENDPOINT}"
else
    echo "AWS_S3_ENDPOINT missing"
    exit 1
fi

if [ -n "${AWS_ACCESS_KEY_ID}" ]; then
    params="${params} -p AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}"
else
    echo "AWS_ACCESS_KEY_ID missing"
    exit 1
fi

if [ -n "${AWS_SECRET_ACCESS_KEY}" ]; then
    params="${params} -p AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}"
else
    echo "AWS_SECRET_ACCESS_KEY missing"
    exit 1
fi

if [ -n "${AWS_DEFAULT_REGION}" ]; then
    params="${params} -p AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}"
else
    echo "AWS_DEFAULT_REGION missing"
    exit 1
fi

if [ -n "${AWS_S3_BUCKET}" ]; then
    params="${params} -p AWS_S3_BUCKET=${AWS_S3_BUCKET}"
else
    echo "AWS_S3_BUCKET missing"
    exit 1
fi

oc process ${params} -f templates/llms/s3_ds_connection.yaml | oc apply -n ${LLM_PROJECT_NAME}  -f -


############################################################################################################
# Served Fine-tuned Model

params=""

FINETUNED_MODEL_NAME=${FINETUNED_MODEL_NAME:-"finetuned"}
params="${params} -p MODEL_NAME=${FINETUNED_MODEL_NAME}"

if [ -n "${FINETUNED_MODEL_S3_PREFIX}" ]; then
    params="${params} -p AWS_S3_PREFIX=${FINETUNED_MODEL_S3_PREFIX}"
else
    echo "FINETUNED_MODEL_S3_PREFIX missing"
    exit 1
fi

oc process ${params} -f templates/llms/served_model.yaml | oc apply -n ${LLM_PROJECT_NAME}  -f -


############################################################################################################
# Create Auth token for served fine-tuned model

INFERENCE_SERVICE_UID=$(oc get inferenceservice ${FINETUNED_MODEL_NAME} -n ${LLM_PROJECT_NAME} -o jsonpath='{.metadata.uid}')
echo ${INFERENCE_SERVICE_UID}

params=""

FINETUNED_MODEL_NAME=${FINETUNED_MODEL_NAME:-"finetuned"}
params="${params} -p MODEL_NAME=${FINETUNED_MODEL_NAME}"

if [ -n "${INFERENCE_SERVICE_UID}" ]; then
    params="${params} -p INFERENCE_SERVICE_UID=${INFERENCE_SERVICE_UID}"
else
    echo "INFERENCE_SERVICE_UID missing"
    exit 1
fi

oc process ${params} -f templates/llms/served_model_auth.yaml | oc apply -n ${LLM_PROJECT_NAME}  -f -