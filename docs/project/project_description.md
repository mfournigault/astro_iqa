# ASTRO IQA PROJECT
Astronomical Image Quality Assessment with Machine Learning

## Project Overview

Astro IQA is an end-to-end machine learning project designed to assess the quality of astronomical images automatically. The objective was to classify telescope images into quality categories such as good, bad tracking, very bad tracking, bad seeing, and background-related issues.

This type of solution can be used to support astronomers and astrophotographers in two practical ways: first, by detecting acquisition issues during image capture, and second, by filtering poor-quality frames before stacking. In both cases, the goal is to make astronomical imaging workflows more reliable and more automated.

## Project Context and Challenge

The project was inspired by the research paper Assessment of Astronomical Images Using Combined Machine-learning Models. However, this work was not a simple reproduction of the published approach.

The original dataset was not publicly available, and the models described in the paper did not produce the expected results on the referenced images. Because of that, I rebuilt the workflow from the ground up. This included dataset construction, annotation processing, feature engineering, exploratory analysis, supervised modeling, evaluation, and deployment-oriented preparation.

One of the main challenges of the project was working in a real-world scientific setting with incomplete benchmarks, heterogeneous data sources, and limited access to the exact resources used in the original publication. The project therefore required both research interpretation and practical engineering adaptation.

## Data Engineering and Dataset Construction

A major part of the work involved building a reliable machine learning dataset from raw astronomical data. The source images were FITS files acquired with the MegaCam camera at the Canada-France-Hawaii Telescope, complemented with additional curated images.

Each FITS file contains multiple CCD captures, and each CCD image can include thousands of detected sources. To transform these raw images into structured machine learning inputs, I used source extraction techniques to compute descriptive features for each detected object. These features included background level, ellipticity, elongation, star-likeness, flags, exposure time, and intensity-related measurements.

The extracted information was consolidated into Parquet catalogs, which then became the basis for downstream modeling. I also built the logic required to convert image annotations into structured JSON mappings and merge them with the extracted object catalogs. This allowed each detected source to inherit image-quality labels and be included in a supervised learning workflow.

In addition to data preparation, I handled practical issues such as missing labels, outliers, class imbalance, and problematic value ranges in parts of the catalog. This stage was critical because model quality depended directly on building a stable and reproducible input pipeline.

## Exploratory Analysis with Self-Organizing Maps

Before training the final classifiers, I explored the feature space using a Self-Organizing Map approach. This unsupervised stage was useful to evaluate whether the extracted astronomical features had real discriminative value.

The SOM analysis showed that features such as ISO0, background, ellipticity, elongation, class_star, flags, and exposure time were relevant for separating groups of sources. Cluster visualizations and U-matrix plots confirmed that the feature space was informative and that the data contained meaningful structure.

At the same time, the analysis also revealed an important limitation: clustering alone was not sufficient to reliably distinguish all bad-quality subclasses. This finding helped guide the project toward a stronger supervised learning strategy, rather than relying too heavily on an unsupervised pipeline.

## Deep Learning Pipeline and Model Design

The main supervised modeling work focused on deep learning using structured features rather than raw pixels. I implemented a complete preprocessing pipeline in TensorFlow and Keras, including robust normalization for numerical features and lookup-based encoding for categorical variables.

Robust normalization was especially important because the data came from mixed acquisition sources and contained significant outliers. I also paid close attention to the TensorFlow dataset pipeline itself. One key engineering finding was that shuffling strategy had a major impact on performance: because many extracted objects came from the same image, small shuffle buffers preserved strong correlations and reduced model quality. To address this, I designed the training pipeline with a much larger shuffle buffer, which significantly improved sample mixing and training stability.

I then developed and compared two neural architectures trained on the same feature set and evaluated under the same conditions:

A simple Deep Neural Network
A Deep & Cross Network V2
This comparison made it possible to evaluate not only predictive performance, but also the practical value of explicit feature interaction learning for structured astronomical data.

## Results and Performance

The final results were strong. Both deep learning models achieved excellent performance on training, validation, and test datasets, with best validation accuracy around 99.9 percent in the documented experiments and no clear evidence of overfitting.

The Deep & Cross Network showed a slight advantage over the standard DNN, especially in the stacked configuration. This suggests that explicit modeling of feature interactions is beneficial for this kind of structured scientific data.

Hyperparameter exploration also revealed useful practical insights. In particular, regularization and shuffle buffer size had a much greater impact on model quality than simply increasing network depth. That kind of conclusion is valuable because it improves not only performance, but also model efficiency and training reliability.

## Value of the Project

From a portfolio perspective, this project demonstrates much more than model training alone. It shows the ability to handle a complete applied machine learning workflow in a scientific domain, from raw data to validated models.

More specifically, the project highlights my ability to:

Build a custom dataset from raw domain-specific scientific data
Design data preparation and annotation pipelines for supervised learning
Validate feature relevance through unsupervised exploratory methods
Implement reproducible TensorFlow data pipelines
Develop and compare deep learning architectures for structured data
Translate research ideas into practical and testable engineering solutions
It also reflects the ability to work effectively under imperfect conditions, where the original benchmark data is unavailable and where academic methods need to be adapted to fit real operational constraints.

## Technologies Used

The project was implemented in Python using TensorFlow, Keras, TensorFlow Recommenders, Pandas, and astronomy-specific tooling for FITS processing and source extraction. It follows a structured data science project layout and includes code, notebooks, reports, testing artifacts, and reproducible dataset preparation scripts.

## Conclusion

Astro IQA is a strong example of applied machine learning in a scientific imaging context. It combines data engineering, exploratory analysis, deep learning, and technical documentation into a complete end-to-end project.

Overall, this work demonstrates my ability to translate an academic concept into a practical machine learning prototype, while solving the engineering challenges required to make the approach usable on real data. It is directly relevant to freelance work in AI, machine learning, data science, and data engineering.