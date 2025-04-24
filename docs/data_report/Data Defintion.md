# Building the Datasets for SOM and Deep Learning Models

The datasets used to train the Self-Organizing Map (SOM) and deep learning models are constructed through a systematic process:

- **Source Extraction**: The initial step involves using a source extraction program to gather useful information from astronomical images. This process identifies and measures various sources detected within the images .

- **Clustering with SOM**: The extracted sources, totaling around 3,000,000, are then organized using the SOM algorithm. This algorithm clusters the sources into 20 distinct groups based on their characteristics, allowing for effective dimensionality reduction  .

- **Representative Images**: From each cluster formed by the SOM, one representative source is selected. The original image pixels associated with these sources are combined to create representative images, which are significantly smaller (∼800 times) than the original images. This reduction aids in faster model training  .

- **Input Data for Models**: The representative images serve as Input-1 for the deep learning model, while additional statistical information about the clusters (Input-2) is also provided. This dual-input approach enhances the model's ability to classify images accurately .

- **Training and Validation**: The datasets are further validated by comparing the model's predictions against a robust set of unseen images, ensuring the model's reliability and performance .

This structured approach to dataset construction allows for improved performance and efficiency in classifying astronomical images.


## Raw Data Sources

Raw data sources are images captured by the MegaCam camera at the Canada-France-Hawaii Telescope (CFHT).
The images are distributed in FITS format and can be downloaded at the url http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/ (Using MegaCam as the instrument).
The dataset of raw images includes the images referenced in the paper and additional downloaded from the CADC archive.
Each FITS file contains 36 CCD captures, and each capture has a size of 2048x4612 pixels. So each FITS file contains 36 source images.

As published in the paper, authors trained the SOM model using a subtantial dataset consisting of 3,000,000 sources extracted from the images. 
The sources are objects extracted from the images by using the SExtractor software.
4000 sources are extracted in average on each CCD image, meaning that 144,000 sources can be extracted from each FITS file.
Meaning that a minimum of 21 FITS files are needed to reach the 3,000,000 sources. As in the end we want to classify images into 5 categories, we need to have at least 4 representative FITS files for each category to train the SOM model.

Further thoughts required to determine the number of images needed to train the deep learning model.

## Processed Data

**Sources** are extracted from images by using the software SExtractor.
For each FITS file present in the directory "./data/raw", the software will produce a LDAC file in the format "FITS_1.0".

**The source catalog** used to train the SOM model is built by combining all the LDAC files produced by the SExtractor software. For each object, SExtractor is used to output the following features:
- X and Y coordinates of the object in the image,
- ISO0,
- ELONGATION,
- ELLIPTICITY,
- CLASS_STAR,
- BACKGROUND.
The exposure time of the image is also added to the catalog as it can significantly affect the quality and characteristics of the detected sources.

**Representative images** are created by combining the pixels of the original images associated with the sources selected by the SOM model. For each cluster, a representative object is randomly selected, and the corresponding cutout image extracted off the ccd image. The representative image is a concatenation of the cutout images for each cluster. The representative images are stored in the directory "./data/processed/".

**The hit value** of each cutout compiled in the representative image is also extracted from the SOM model and used as auxiliary input for the deep learning model. The hit value is the number of objects associated to the cluster for the given CCD image.


## Access to the datasets

The datasets used in the project are not shared in this repository. They are uploaded to a huggingface dataset, at the url https://huggingface.co/datasets/selfmaker/astro_iqa. 
Two kinds of datasets are available:
1- Raw data: FITS files containing the raw images and the LDAC files containing the sources extracted from the images.
2- For modeling data: parquet files containing the compilation of the LDAC files, plus the quality annotations, to be read as dataframes.

To access the datasets, please send a request to the owner of this repository.