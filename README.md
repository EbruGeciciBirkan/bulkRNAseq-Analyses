# bulkRNAseq-Analysis
This repository was created to analyze the performance of methods used to analyze scRNA-seq data on bulk-RNA-seq data. The relevant analyses were carried out in two stages:
1. The performance of existing methods used to analyze scRNA-seq data on bulk-RNA-seq data was examined.
2. Fine-tuning operations were performed on the model showing the best performance among the applied methods.

Dataset Used:
We used a dataset created for the Cancer Genome Atlas (TCGA) project, which is available at https://gdc-portal.nci.nih.gov, as our bulk RNA-seq dataset.
The TCGA dataset contains genetic, clinical, and pathological data for over 10,000 patients with 33 different cancer types (also known as the cohort). 

scRNA-seq Analysis Methods Used:
1. scMulan (https://github.com/SuperBianC/scMulan)
2. scGPT (https://github.com/bowang-lab/scGPT)
3. SCimilarity (https://github.com/Genentech/scimilarity)

Files containing the results obtained with these methods are located under embeddings.

Fine-tunings' Models:
We adapted the encoder layer structure from SCimilarity in our proposed fine-tuning models to ensure compatibility with the TCGA data. We proposed the 
1. No Fine-tuning,
2. Partial Fine-tuning, and
3. All Layers Fine-tuning models.

Files containing the results obtained with these methods are located under finetunings.

The studies were carried out using the Python programming language. For the relevant models to work, the SCimilarity model file (modelv1.1), dataset, and model scripts must be in the same directory.
