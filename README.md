# bulkRNAseq-Analysis
This repository was created to analyze the performance of methods used to analyze scRNA-seq data on bulk-RNA-seq data. The relevant analyses were carried out in two stages:
1. The performance of existing methods used to analyze scRNA-seq data on bulk-RNA-seq data was examined.
2. Fine-tuning operations were performed on the model showing the best performance among the applied methods.

### Dataset Used:
We used a dataset created for the Cancer Genome Atlas (TCGA) project, which is available at https://gdc-portal.nci.nih.gov, as our bulk RNA-seq dataset.
The TCGA dataset contains genetic, clinical, and pathological data for over 10,000 patients with 33 different cancer types (also known as the cohort). 

#### Preporcessing of the data
We focused on mRNA expression profiles obtained from patients’ primary tumor samples. Gene expression quantification files were downloaded for 33 TCGA cohorts using the STAR-Counts workflow. For each sample, TPM-based expression values were extracted. Gene identifiers were mapped using a gene conversion table, and cohort-specific mRNA expression matrices were generated. The cohort-level matrices were then merged into a unified TCGA mRNA expression dataset.

The resulting dataset was used as the main bulk RNA-seq input for analyses with pretrained foundation models. Depending on the requirements of each model, additional model-specific preprocessing steps were applied before generating sample-level embeddings. For example, the expression matrix was aligned with the model’s predefined gene vocabulary and converted into AnnData `.h5ad` format. For SCimilarity, the TPM-based mRNA expression matrix was used after being arranged in a compatible input format.

The cohort label was used as the class label in the classification task. 

### scRNA-seq Analysis Methods Used:
1. scMulan (version 1.0) (https://github.com/SuperBianC/scMulan)
2. scGPT (version 0.2.1) (https://github.com/bowang-lab/scGPT)
3. SCimilarity (version=0.4.1) (https://github.com/Genentech/scimilarity)

Files containing the results obtained with these methods are located under embeddings.

### Fine-tunings' Models:
We adapted the encoder layer structure from SCimilarity in our proposed fine-tuning models to ensure compatibility with the TCGA data. We proposed the 
1. No Fine-tuning,
2. Partial Fine-tuning, and
3. All Layers Fine-tuning models.

Files containing the results obtained with these methods are located under finetunings.

The studies were carried out using the Python programming language. For the relevant models to work, the SCimilarity model file (modelv1.1), dataset, and model scripts must be in the same directory.
