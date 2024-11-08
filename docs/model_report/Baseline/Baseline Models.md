# Baseline Model Report

As a baseline model, we worked on a SOM clustering based on features extracted from the images with SEXtractor (as in the original paper). The SOM model is built on the [IntraSOM library](https://github.com/InTRA-USP/IntraSOM).

## Analytic Approach
* As in the orginal paper, the target is to classify astronomical images based on data quality. At a first step, we focused on clustering into 5 different groups data extracted from the images, which would allow a remapping in a second step (with a neural network) into the 5 categories: good, bad tracking, very bad tracking, bad seeing, or background issues.
* The inputs are the features extracted from the images with SEXtractor, stored into the ldac files.
* The model built is a Self-Organizing Map (SOM) model used for clustering the data extracted from the images.

## Model Description

The model built is described in the notebooks "evaluating_clusters_classification_power.ipynb", "evaluating_clusters_classification_power2.ipynb" and "som_clustering.ipynb".

* Features used for clustering:
  "OBJECT_ID", "ISO0", "BACKGROUND", "ELLIPTICITY", "ELONGATION", "CLASS_STAR", "FLAGS", "EXPTIME"
* Components of the SOM model:
	"ISO0", "BACKGROUND", "ELLIPTICITY", "ELONGATION", "CLASS_STAR", "FLAGS", "EXPTIME"
* Parameters used for the SOM model:
	mapsize=(15, 15),
	mapshape="planar",
    lattice="hexa",
    normalization="var",
    initialization="random",
    neighborhood="gaussian",
    training="batch",
    name="som_iqa",
    component_names=component_names,
    unit_names=None,
    sample_names=catalog_df["OBJECT_ID"].to_list()


## Results (Model Performance)

* Predicting power of features:
  The plots of U-matrix per component (see the figures in the directory "data/clustering/Plots/Component_plots") show that the features "ISO0", "BACKGROUND", "ELLIPTICITY", "ELONGATION", "CLASS_STAR", "FLAGS", "EXPTIME" have a good predicting power. It confirms the interest of using these features for assessing the image quality.
* Classification power of the clusters:
  As described in the notebook "evaluating_clusters_classification_power2.ipynb", the clusters have a limited power of classification.
  The confusion matrix shows that the clusters are able to separate easily the "good" images from the others, but have difficulties to separate the other categories. The result analysis has shown that the subset of data used for the clustering, corresponding to some of the images described in the paper, are not good representatives enough to separate classes of the "bad" categories.

## Conclusion and Discussions for Next Steps

* Conclusion on Feasibility Assessment of the Machine Learning Task
	The prediction power the features extracted is encouraging on the feasibility of the task. But the classification power of the clusters is not sufficient to classify the images into 5 categories (even if they don't map directly to the 5 target classes).
	The dataset needs to be augmented with more representative images to improve the classification power of the clusters. The difficulty is that the dataset used 
	in the paper is not shared, neither fully described. The image catalog of the CADC is so enormous and not provided with visual and userfriendly tools that could help in choosing new data. It is also not really feasible to build a representative dataset by using the CADC catalog.
	A mean to do it would to import personal images captured with my own telescop and camera.

* Augmenting the dataset
	A first attempt on augmenting the dataset with personal data has been made, resulting in new data available in the directories "data/raw/ngc0896", "data/raw/ngc0869", "data/raw/nrc7000".
	These data have been annotated (annotations are available in the directory "data/for_modeling"), and processsed with SEXtractor (resulting in the ldac files available in the directory "data/raw/ngcXXXX").
	It is important to note that at this date (2024/11/08), the new data imported are mostly representatives of the classes "good", "bad tracking" or "very bad tracking". 
	Only 174 new images have been imported.

* Next Steps
	- Re-train the SOM model with the augmented dataset and evaluate the classification power of the clusters.
	- Import more personal data to augment the dataset and improve the classification power of the clusters.
	- Implement a neural network to classify the clusters into the 5 categories. As in the original paper, authors used only one representative image per fits file (a capture) to feed a CNN, following the same approach with our current dataset would provide very few data to train the CNN.
	An alternative approach would be to use directly the numerical features extracted from the images, to train a neural network. By this way we would have much more data to train the neural network.
	In addition, using a CNN combined with the SOM may be a good idea to reduce the dimension of the dataset fed to the CNN. But if the dataset is not so large, the interest of using a combination of models may be limited. Using a SOM ahead of a CNN, if not for reducing the dataset dimensionnality, means that the features extracted by the CNN are not powerful enough to classify the images. But the features extracted to feed the SOM are similar to some high level features that could be extracted by the CNN. And as we saw it in our analysis, these features show a promising predictive power. So the CNN would not bring much more information than the SOM.
	- Try a gated and selective neural network on the numerical (descriptive) features may be more promising.
	
* What other Features Can Be Generated from the Current Data

* What other Relevant Data Sources Are Available to Help the Modeling