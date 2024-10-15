# RHEL AI POC 

This repository contains the code and documentation for creating RHEL AI Proofs of Concept (POCs).  This aims to be straightforward and comply with the available documentation for [RHEL AI](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/) and [InstructLab](https://github.com/instructlab/instructlab/blob/main/README.md). The goal is to provide a simple and easy to follow guide for creating successful and reproducible POCs that can be used to demonstrate the capabilities of RHEL AI.

Much of what is here, are some nuance and gotchas that may not be covered in the main documentation.  There are some included scripts and automation, but many of these are anticipated to be removed as the features are integrated into the main RHEL AI product.


## Table of Contents
<!-- TOC -->
* [RHEL AI POC](#rhel-ai-poc-)
  * [Table of Contents](#table-of-contents)
  * [Suggested documentation](#suggested-documentation)
  * [Prerequisites](#prerequisites)
  * [Installation and Configuration](#installation-and-configuration)
    * [Bare Metal](#bare-metal)
    * [IBM Cloud](#ibm-cloud)
    * [AWS](#aws)
      * [Creating an AWS AMI](#creating-an-aws-ami)
      * [Launching an AWS Instance](#launching-an-aws-instance)
  * [Document Collection](#document-collection)
    * [PDF](#pdf-)
    * [qna.yaml](#qnayaml)
    * [Evaluation Questions](#evaluation-questions)
  * [Data Preparation](#data-preparation)
    * [PDF to Markdown Conversion](#pdf-to-markdown-conversion)
  * [Synthetic Data Generation](#synthetic-data-generation)
    * [ilab generate](#ilab-generate)
  * [Model Training](#model-training)
    * [ilab train](#ilab-train)
  * [Deployment and Testing](#deployment-and-testing)
    * [Saving the model to S3 storage](#saving-the-model-to-s3-storage)
    * [Serving the model](#serving-the-model)
      * [ilab serve](#ilab-serve)
      * [OpenShift AI Serving](#openshift-ai-serving)
  * [Evaluation](#evaluation)
    * [Testing the model](#testing-the-model)
    * [RAG with Anything LLM](#rag-with-anything-llm)
    * [RAG testing with InstructLab](#rag-testing-with-instructlab)
<!-- TOC -->

## Suggested documentation
* [RHEL AI](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/) 
* [InstructLab](https://github.com/instructlab/instructlab/blob/main/README.md)


## Prerequisites


## Installation
[Official Installation Documentation](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.1/html/installing/)

### Bare Metal
TBD

### IBM Cloud
TBD

### AWS


#### Creating an AWS AMI


#### Launching an AWS Instance


### Initial Configuration


## Document Collection


### PDF

The easiest data to use for training a custom model, is PDF formatted.  Create a directory in `document_collection` under the name for your data such as `my_org` and place your pdf in the directory.

### qna.yaml

The qna.yaml file is a simple yaml file that contains the questions and answers for the document.  This file is used in the synthetic data generation process to create the training data for the model.  The qna.yaml file should be in the same directory as the PDF file (e.g. `document_collection/my_org`.  For more on the format and structure of the qna.yaml file, see the [documentation](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.2/html-single/creating_a_custom_llm_using_rhel_ai/index#customize_llm_knowledge_example)

TBD qna.yaml best practices and gotchas.

### Evaluation Questions
It is also useful to obtain a set of human generated questions and answers to judge the model.  These questions will be held separate from the model training data and used to evaluate the model.


## Data Preparation

### PDF to Markdown Conversion
[RHEL AI Official Doc](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.2/html-single/creating_a_custom_llm_using_rhel_ai/index#customize_llm_create_knowledge_d)

Since RHEL AI needs markdown of the knowledge data in for synthetic data generation (SDG), the first step is to convert the PDF to markdown.  For this we will use Docling to convert the PDF to chunks of markdown text.  The code to do so is a work in progress and will be added to the InstructLab main project, but for now, we've taken the current versions of the SDG conversion scripts and added them here as a workaround.  You can call the scripts using a simple notebook: [data_preparation/pdf_conversion.ipynb](data_preparation/pdf_conversion.ipynb).

First create a `data_preparation/.env` file similar to the [example](data_preparation/.env.example) specifying the data_directory (e.g.` is where all input documents including the pdf and qna.yaml will be located.  Then set the `DATA_DIR` and `OUTPUT_DIR` variables in the notebook to point to the correct directories.  For example:

```shell
DATA_DIR=document_collection/my_org
OUTPUT_DIR=output/my_org
```

Next run the cells in the notebook to convert the PDF from the `DATA_DIR` to markdown.  The notebook will create an `OUTPUT_DIR` directory for the converted markdown files in a folder named `md`.  The markdown has been broken up in to chunks for better context for the synthetic data generation process. The markdown files will be named document_1.md, document_2.md.

Commit the changes to the `OUTPUT_DIR` to the repository.  Take note of the markdown locations and the commit hash for the training data. 

Now you can clone this repo on the RHEL AI instance and use the markdown files for synthetic data generation.  

### Update the qna.yaml

Now that we've cloned the repository with the markdown files onto your RHEL AI instance, we can reference them from the qna.yaml file.  

Update `document:` section the qna.yaml file to point to the markdown files in the cloned repository.  For example:
```yaml
document:
  repo: 'file:///home/example-user/my_poc'
  commit: abc123
  patterns:
    - 'output/my_org/md/*.md'
```
Now you can copy the qna.yaml file and place it in the appropriate location on the RHEL AI instance.  
`~/.local/share/instructlab/taxonomy/knowledge/<my_org>/qna.yaml`

Verify the qna.yaml and your taxonomy are valid using
```bash
ilab taxonomy diff
```

## Synthetic Data Generation

### ilab generate

Now, that we've finished preparing the data, we're ready to generate the synthetic data.  This is done using the `ilab generate` command.  This command will take the markdown files and the qna.yaml file and generate the synthetic data for training the model.  You can run the command in a background task or tmux to make sure it runs to completion.  For example:

```bash
nohup ilab data generate --chunk-word-count 10000 --server-ctx-size 16384 > generate.log 2>&1 
```

The synthetic data will be in a directory `~/.local/share/instructlab/datasets/` and be named `skills_train_msgs...jsonl` and `knowledge_train_msgs...jsonl`.  These files will be used to train the model in the next step.


## Model Training

### ilab train

```bash
nohup ilab model train --strategy lab-multiphase \
  --phased-phase1-data ~/.local/share/instructlab/datasets/<knowledge-train-messages-jsonl-file> \
  --phased-phase2-data ~/.local/share/instructlab/datasets/<skills-train-messages-jsonl-file> \
  --model-path /home/example-user/.cache/instructlab/models/granite-7b-redhat-lab \
  --gpus 8 \
  -y \
  > training.log 2>&1 &
```





## Deployment and Testing

### Saving the model to S3 storage

### Serving the model

#### ilab serve

#### OpenShift AI Serving



## Evaluation

### Testing the model

### RAG with Anything LLM

### RAG testing with InstructLab

