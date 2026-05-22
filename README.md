# bulkRNAseq-Analysis
This repository was created to analyze the performance of methods used to analyze scRNA-seq data on bulk-RNA-seq data. The relevant analyses were carried out in two stages:
1. The performance of existing methods used to analyze scRNA-seq data on bulk-RNA-seq data was examined.
2. Fine-tuning operations were performed on the model showing the best performance among the applied methods.

### 1. Dataset Used:
We used a dataset created for the Cancer Genome Atlas (TCGA) project, which is available at https://gdc-portal.nci.nih.gov, as our bulk RNA-seq dataset.
The TCGA dataset contains genetic, clinical, and pathological data for over 10,000 patients with 33 different cancer types (also known as the cohort). 

#### Preporcessing of the data
We focused on mRNA expression profiles obtained from patients’ primary tumor samples. Gene expression quantification files were downloaded for 33 TCGA cohorts using the STAR-Counts workflow. For each sample, TPM-based expression values were extracted. Gene identifiers were mapped using a gene conversion table, and cohort-specific mRNA expression matrices were generated. The cohort-level matrices were then merged into a unified TCGA mRNA expression dataset.

The resulting dataset was used as the main bulk RNA-seq input for analyses with pretrained foundation models. Depending on the requirements of each model, additional model-specific preprocessing steps were applied before generating sample-level embeddings. For example, the expression matrix was aligned with the model’s predefined gene vocabulary and converted into AnnData `.h5ad` format. For SCimilarity, the TPM-based mRNA expression matrix was used after being arranged in a compatible input format.

The cohort label was used as the class label in the classification task. 

### 2. scRNA-seq Analysis Methods Used:
1. scMulan (version 1.0) (https://github.com/SuperBianC/scMulan)
2. scGPT (version 0.2.1) (https://github.com/bowang-lab/scGPT)
3. SCimilarity (version 0.4.1) (https://github.com/Genentech/scimilarity)

Files containing the results obtained with these methods are located under embeddings.

### 3. Fine-tunings' Models:
We adapted the encoder layer structure from SCimilarity in our proposed fine-tuning models to ensure compatibility with the TCGA data. We proposed the 
1. No Fine-tuning,
2. Partial Fine-tuning, and
3. All Layers Fine-tuning models.

Files containing the results obtained with these methods are located under finetunings.

The studies were carried out using the Python programming language. For the relevant models to work, the SCimilarity model file (modelv1.1), dataset, and model scripts must be in the same directory.

### 4. Overall Workflow and Script Usage

This pipeline, therefore, includes dataset preparation, dependency control, installation of local analysis functions, embedding generation with pretrained models, model evaluation, and fine-tuning-based performance improvement.

#### 4.1. Dataset Preparation
TCGA mRNA expression data were downloaded and processed to construct a unified bulk RNA-seq dataset. 

#### 4.2. Environment and Package Requirements
Before running the pretrained foundation models, the required software environments and package dependencies should be installed and checked carefully. Since each pretrained model may depend on a different Python or R environment, the corresponding package versions, model-specific requirements, and input data formats should be verified before execution.

#### 4.3. Installation of Local Evaluation Functions
After preparing the dataset and software environments, the custom functions developed for model evaluation should be installed before starting the analysis.

These functions were organized into local packages to provide a reusable, structured analysis framework. They include functions for evaluating embedding quality, classification performance, visualization outputs, and downstream comparison of pretrained and fine-tuned models.

Therefore, before running the main analysis scripts, the local packages containing these evaluation functions must be installed or loaded into the working environment.

#### 4.4. Embedding Generation Using Pretrained Models
After dataset preparation and package installation, pretrained foundation models were used to generate sample-level embedding representations from the TCGA mRNA expression profiles.

Each pretrained model was applied according to its own input requirements. The resulting embeddings were then used as feature representations for downstream classification and performance evaluation tasks.

In this stage, the pretrained models are used without additional task-specific training. The purpose is to assess how well the pretrained transcriptomic representations capture biologically and clinically meaningful differences among TCGA cancer cohorts.

#### 4.5. Model Selection and Fine-Tuning

Following the evaluation of pretrained embedding performance, the model with the most promising performance was selected for fine-tuning. Fine-tuning was performed to adapt the pretrained foundation model more specifically to the TCGA cohort classification task. This step enables the model to update its learned representations with task-specific information, potentially improving downstream classification performance.

There are three fine-tuning model configurations included in this workflow. These models can be executed either sequentially or in parallel, depending on the available computational resources. Running them in parallel may reduce total execution time, whereas sequential execution may be more suitable for systems with limited memory or GPU capacity.

