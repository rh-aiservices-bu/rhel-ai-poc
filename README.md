# RHEL AI POC

This repository contains the code and documentation for creating RHEL AI Proofs of Concept (POCs).  This aims to be straightforward and comply with the available documentation for [RHEL AI](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/) and [InstructLab](https://github.com/instructlab/instructlab/blob/main/README.md). The goal is to provide a simple and easy to follow guide for creating successful and reproducible POCs that can be used to demonstrate the capabilities of RHEL AI.

Much of what is here, are some nuance and gotchas that may not be covered in the main documentation.  There are some included scripts and automation, but many of these are anticipated to be removed as the features are integrated into the main RHEL AI product.


## Table of Contents
<!-- TOC -->
* [RHEL AI POC](#rhel-ai-poc)
  * [Table of Contents](#table-of-contents)
  * [Suggested documentation](#suggested-documentation)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
    * [Bare Metal](#bare-metal)
    * [IBM Cloud](#ibm-cloud)
    * [AWS](#aws)
      * [Creating an AWS AMI](#creating-an-aws-ami)
      * [Launching an AWS Instance](#launching-an-aws-instance)
    * [Initial Configuration](#initial-configuration)
  * [Document Collection](#document-collection)
    * [PDFs](#pdfs)
    * [Markdown](#markdown)
    * [qna.yaml](#qnayaml)
    * [Evaluation Questions](#evaluation-questions)
  * [Data Preparation](#data-preparation)
    * [PDF to Markdown Conversion](#pdf-to-markdown-conversion)
      * [Python script](#python-script)
      * [Notebook](#notebook)
    * [Commit the changes](#commit-the-changes)
    * [Update the your qna.yaml files](#update-the-your-qnayaml-files)
      * [Update the taxonomy on the RHEL AI instance with the new qna.yaml file.](#update-the-taxonomy-on-the-rhel-ai-instance-with-the-new-qnayaml-file)
  * [Synthetic Data Generation](#synthetic-data-generation)
    * [IBM Cloud](#ibm-cloud-1)
    * [RHEL AI Cluster](#rhel-ai-cluster)
  * [Model Training](#model-training)
    * [IBM Cloud](#ibm-cloud-2)
    * [RHEL AI Cluster](#rhel-ai-cluster-1)
  * [Deployment and Testing](#deployment-and-testing)
    * [RHEL AI Cluster](#rhel-ai-cluster-2)
    * [OpenShift AI Serving](#openshift-ai-serving)
      * [Saving the model to object storage](#saving-the-model-to-object-storage)
      * [Kserve and vLLM](#kserve-and-vllm)
  * [Evaluation](#evaluation)
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

### PDFs
The most common data to use for training a custom model, is PDF formatted.  In the directory in `document_collection` add your pdfs.  These will be used to create markdown versions of the data suitable for the RHEL AI synthetic data generation.  During the data prepration step, we will convert the PDFs to markdown.  By default, markdown, broken in chunks will be created in the output folder.  You will refer to these generated markdown files from your "qna.yaml" file.

### Markdown
If your data is in markdown format already, you can refer to markdown documents directly in the qna.yaml

### qna.yaml
[Official Documentation - Customizing your taxonomy tree](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.2/html/creating_a_custom_llm_using_rhel_ai/customize_taxonomy_tree)

The qna.yaml file is a simple yaml file that contains the questions and answers for the document.  This file is used in the synthetic data generation process to create the training data for the model.  They should be placed in the appropriate directory structure in the `taxonomy` folder. (e.g. `taxonomy/knowledge/my_subject/qna.yaml`).  At the end of this process, you should be able to copy this directory structure as needed for the POC either to upload to the cloud service or the RHEL AI instance used in the POC.

TBD qna.yaml best practices and gotchas.

### Evaluation Questions
It is also useful to obtain a set of human generated questions and answers to judge the model.  These questions will be held separate from the model training data and used to evaluate the model.  These questions and "ground truth" answers will be used in our evaluation process to evaluate how our model is doing.  You can format these questions in a CSV format with the columns "question" and "ground_truth" and place them in the `eval/qna` directory.  In addition you can use the same format as a qna.yaml.  This is useful if you would like to use a qna.yaml for evaluation as a trial run or stand in.

```yaml
seed_examples:
  - questions_and_answers:
      - question: >
          relevant question
        answer: >
          reference / ground truth answer
      - question: >
          relevant question
        answer: >
          reference / ground truth answer
```


## Data Preparation

### PDF to Markdown Conversion
[RHEL AI Official Doc](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.2/html-single/creating_a_custom_llm_using_rhel_ai/index#customize_llm_create_knowledge_d)

Since RHEL AI needs markdown of the knowledge data in for synthetic data generation (SDG), the first step is to convert the PDF to a format that is easily digestible.  For this we will use Docling to convert the PDF to chunks of markdown text.  The code to do so is a work in progress and will be , but for now, we've taken the current versions of the SDG conversion scripts and added them here as a workaround.

#### Python script
In the `document_chunker.py` python script will convert pdfs to chunks of markdown ready for SDG.  Enable your python environment and switch to the `data_preparation` folder:
```bash
source venv/bin/activate
cd data_preparation
```

Then run the script a command, such as the example:
```bash
pip install -r requirements.txt
python document_chunker.py --input-dir document_collection --output-dir output
```

#### Notebook
In the event you would like to see the steps of the process or perhaps customize the process, you can convert the PDF using a simple notebook: [data_preparation/pdf_conversion.ipynb](data_preparation/pdf_conversion.ipynb).

Next run the cells in the notebook to convert all PDFs from the `document_collection` directory to markdown.  The notebook will create an `output` directory for the converted markdown files, copied taxonomy, and some intermediate files.  The markdown has been broken up in to chunks for better context for the synthetic data generation process. The markdown files will be named document_1.md, document_2.md.

### Commit the changes
Commit the changes to the `OUTPUT_DIR` to the repository.  Take note of the markdown locations and the commit hash for the training data, and push it up to the repository. 

### Update the your qna.yaml files
[Official Documentation - Creating a konwledge YAML file](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.2/html/creating_a_custom_llm_using_rhel_ai/customize_taxonomy_tree#customize_llm_create_knowledge)
For the copied taxonomy in `output/taxonomy`, you need to refer to your output markdown files and the commit hash in the `document` section.  

If the repository is public, simply refer to the repo URL, commit hash, and the path to markdown files in the qna.yaml file.  For example:

```yaml
document:
  repo: 'https://github.com/user/poc-repo'
  commit: 50e47897b5d5bb359504618fa33a83110e87f5f8
  patterns:
    - 'data_preparation/output/knowledge/my_subject/md/*.md'
```
After updating your qna.yaml files in this project, you can test them out like so:
```bash
ilab taxonomy diff --taxonomy-path ./output/taxonomy
```
After you've validated the qna.yaml files, you can commit them to the taxonomy repository and push

**NOTE:**  file references
In the case of a private repository or a repo that only exists on the RHEL AI server, you can refer to the documents using the `file://` protocol.  For example:

```yaml
document:
  repo: 'file:///home/example-user/poc-repo'
  commit: 50e47897b5d5bb359504618fa33a83110e87f5f8
  patterns:
    - 'data_preparation/output/knowledge/my_subject/md/*.md'
```

#### Update the taxonomy on the RHEL AI instance with the new qna.yaml file.

Now you can copy the output taxonomy folder and place it in the appropriate location on the RHEL AI instance.  
```
rm -rf ~/.local/share/instructlab/taxonomy
cp -r <poc-repo>/data_preparation/taxonomy ~/.local/share/instructlab/
```

After the files are copied, you can do a final check on the taxonomy before proceeding.
```bash
ilab taxonomy diff
```

## Synthetic Data Generation
Now, that we've finished preparing the documents and taxonomy, we're ready to generate the synthetic data.  This is done using the `ilab generate` command.  This command will take the markdown files and the qna.yaml file and generate the synthetic data for training the model.

### IBM Cloud
TBD

### RHEL AI Cluster
You can run the command in a background task or tmux to make sure it runs to completion.  In addition, it is customized to allow each of the previously generated markdown chunks to be consumed whole during the SDG process. 

```bash
ilab data generate --chunk-word-count 10000 --server-ctx-size 16384
```

If you want to run it in the background, so the process does not get interrupted:
```bash
nohup ilab data generate --chunk-word-count 10000 --server-ctx-size 16384 > generate.log 2>&1 
tail -f generate.log
```

The synthetic data will be in a directory `~/.local/share/instructlab/datasets/` and be named `skills_train_msgs...jsonl` and `knowledge_train_msgs...jsonl`.  These files will be used to train the model in the next step.


## Model Training

### IBM Cloud
TBD


### RHEL AI Cluster

```bash
ilab model train --strategy lab-multiphase \
  --phased-phase1-data ~/.local/share/instructlab/datasets/knowledge_train_msgs_2024-11-08T22_55_40.jsonl \
  --phased-phase2-data ~/.local/share/instructlab/datasets/skills_train_msgs_2024-11-08T22_55_40.jsonl   
```

When trying to run `ilab train` in a background process, you'll need to add `-y` in order to skip the interactive prompts and avoid the process not progressing.  For example:

```bash
nohup ilab model train -y \
  --strategy lab-multiphase \
  --phased-phase1-data ~/.local/share/instructlab/datasets/knowledge_train_msgs_2024-11-08T22_55_40.jsonl \
  --phased-phase2-data ~/.local/share/instructlab/datasets/skills_train_msgs_2024-11-08T22_55_40.jsonl \
  > training.log 2>&1 &

tail -f training.log
```

## Deployment and Testing
To test out and evaluate the model for demo purposes, first we need to serve the model.  After training you will have received a message in the training output like the following:


```bash
Training finished! Best final checkpoint: <path-to-best-performed-checkpoint> with score: 6.968152866242038
```

This is the model checkpoint we'll serve.

### RHEL AI Cluster
[Official Documentation - Serving and chatting with your new model](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.2/html/creating_a_custom_llm_using_rhel_ai/serving_chatting_new_model)

Once the model training is complete 
```bash
ilab model serve --model-path <path-to-best-performed-checkpoint>
```

### OpenShift AI Serving
[Official Documentation - Serving large models with OpenShift AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2-latest/html/serving_models/serving-large-models_serving-large-models)

If you choose to serve the model with OpenShift AI, this will give you the flexibility to open up the model as an endpoint secured by an API key.  You will also be able to build applications (such as a RAG chatbot) for demo purposes.

#### Saving the model to object storage
To serve the model with OpenShift AI, you will need to save the model to an object storage bucket that is compatible with the S3 API.  You can edit and use the python script `misc/s3_uploader` if you wish.

#### Kserve and vLLM
Once the model is in storage, you can add an OpenShift AI data connection to the storage bucket.  Once the connection is established, you can deploy it with the vLLM serving image.  The model will be served as an endpoint that you can access with an API key.

## Evaluation
Evaluation notebooks are available in the `eval` directory.  These notebooks will allow you to evaluate the model using the evaluation questions and answers you've collected.  The notebooks will allow you to run the model against the evaluation questions and answers and provide a score for the model in comparison with other models using RAG.

There are two notebooks available for evaluation:
### Evaluation via service
Notebook: [eval_rh_api.ipynb](eval/eval_rh_api.ipynb)

This notebooks is available to run a standard evaluation ([llm-eval-app](https://github.com/cfchase/llm-eval-app)).  To run this notebook, you will need:
  * The URL and API key for the evaluation service. 
  * The information for your fine-tuned model deployed and accessible via an API.
  * A set of reference questions and answers in a common format (csv, jsonl, or qna.yaml).
  * A set of context data in PDF format.  These are generally the documents that the model was trained on.
The notebook includes instructions on how to run the evaluation and how to interpret the results.  The notebook will provide a scores for the model, the model with RAG, Granite 3 with RAG, and ChatGPT with RAG.

### Custom evaluation notebook
Notebook: [local_custom_eval.ipynb](eval/local_custom_eval.ipynb)

This notebook is available for more custom evaluations.  With this notebook you can compare the model to any number of other models of your choosing with any template of your choosing.  You can also customize a ChatGPT scoring template.  To run this notebook, you will need:
  * An OpenAI API key.
  * The information for your fine-tuned model deployed and accessible via an API.
  * The API information for each of the models you wish to compare against.
  * A set of reference questions and answers in a common format (csv, jsonl, or qna.yaml).
  * A set of context data in PDF format.  These are generally the documents that the model was trained on.
  * A set of questions and answers for evaluation in a common format (csv, jsonl, or qna.yaml).

