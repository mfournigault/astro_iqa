 # Astro IQA: Quality Assessment for astronomical images

This repository contains the code for the Astro IQA project. The project aims to develop a quality assessment tool for astronomical images. 
The aims of the tool is, given an astronomical image, to classify the image between categories: good, bad tracking, very bad tracking, bad seeing, or background issues.
This classification can then be used:
- during image capturing with a telescope to warn the user of potential issues,
- during image stacking to discard bad images of bad quality, and so enable a stacking pipeline completely automated.

The project follows the Microsoft TDSP structure.

## Introduction

This project is based on the publication [Assessment of Astronomical Images Using Combined Machine-learning Models](https://doi.org/10.3847/1538-3881/ab7938).
The paper presents a two-component machine-learning approach for classifying astronomical images based on data quality. It utilizes a clustering algorithm to create representative images that are significantly smaller than the originals, facilitating faster training of the model. A deep neural network then classifies these images, achieving a 97% agreement rate with manual classifications and improving results by about 10% over traditional methods. The method effectively distinguishes usable images from those taken under suboptimal conditions, demonstrating its utility in handling large astronomical data sets. Overall, the approach enhances the efficiency and accuracy of astronomical image assessment .

## Key Contributions of the Paper

The paper titled "Assessment of Astronomical Images Using Combined Machine-learning Models" presents several significant contributions to the field of astronomical image classification. Here are the main contributions outlined in the paper:

- **Two-Component Machine Learning Approach**: The authors introduce a novel two-component machine-learning-based method for classifying astronomical images. This method effectively examines both the sources detected in the images and the pixel values from representative sources, enhancing the classification process .

- **Clustering Algorithm for Data Reduction**: The first component of the approach employs a clustering algorithm that significantly reduces the number of image pixels needed for analysis. The representative images created are approximately 800 times smaller than the original images, which not only preserves useful information but also reduces the time required for training the algorithm .

- **Deep Neural Network for Classification**: The second component utilizes a deep neural network model to classify the representative images. This model is designed to separate "usable" images from those that may present issues for scientific projects, such as images taken under suboptimal conditions .

- **Improved Performance Over Traditional Methods**: The proposed method demonstrates a 97% agreement rate when compared to classifications generated through manual inspection. It also shows an improvement of about 10% over traditional classification methods, providing more comprehensive outcomes .

- **Utilization of Multiple Data Inputs**: The approach leverages two different data sets as inputs to the deep model, which enhances performance compared to using only pixel information from the images. This dual-input strategy allows for a more robust analysis of large and complex astronomical data sets .

- **Effective Use of Self-Organizing Maps (SOM)**: The paper discusses the use of Self-Organizing Maps (SOM) to group sources into classifications, allowing the model to distinguish between different image classes effectively. This method ensures that the model receives specific image subsets, optimizing the input without overwhelming it with redundant information .

These contributions collectively advance the field of astronomical image analysis by integrating machine learning techniques that improve classification accuracy and efficiency.

## Description of the implementation

As the models provided by the authors did not give the expected results on the images referenced in the paper, we decided to implement and train the models from scratch. All the code is implemented in Python (3.10+).
The SOM model is built on the [IntraSOM library](https://github.com/InTRA-USP/IntraSOM).
The deep neural network model is built with Keras.

As no dataset has been shared by the authors, we built a dataset by following the procedure described in the paper and images captured by the MegaCam camera at the Canada-France-Hawaii Telescope (CFHT). 
For details on the dataset, please refer to the data report in the `docs` folder.

## Datasets used in the project

The datasets used in the project are not shared in this repository. They are uploaded to a huggingface dataset. 
To access the datasets, please send a request to the owner of this repository.