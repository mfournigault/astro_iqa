# List of feature sets

For each catalog file objects_catalog_XXX.parquet.gz, features are 
- "OBJECT_ID", 
- "FITS_ID", 
- "CCD_ID", 
- "ISO0", 
- "BACKGROUND", 
- "ELLIPTICITY", 
- "ELONGATION", 
- "CLASS_STAR", 
- "FLAGS", 
- "EXPTIME", 
- "gt_label1", 
- "gt_label2"

"gt_label1" and "gt_label2" are the ground truth labels for the object, target of classification.
They are extracted from the annotation file map_images_labels_XXX.json.

For the data and file formats, see the data definition report in the documentation folder of the git repository.
