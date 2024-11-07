import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import argparse

def read_catalog(filename, columns):
    """
    Read the catalog from an HDF5 file and return it as a pandas DataFrame.
    """
    # catalog = Table.read(filename, path="som_catalog", format="hdf5")
    # catalog_df = catalog.to_pandas()
    catalog_df = pd.read_parquet(filename , engine='auto')
    catalog_df = catalog_df[columns]
    # catalog_df["OBJECT_ID"] = catalog_df["OBJECT_ID"].apply(lambda x: x.decode('utf-8'))
    # catalog_df["FITS_ID"] = catalog_df["FITS_ID"].apply(lambda x: x.decode('utf-8'))
    return catalog_df


def read_annotations(filename):
    """
    Read the annotations from a JSON file and return it as a dictionary.
    """
    with open(filename, 'r') as f:
        annotations = json.load(f)
    return annotations


def transform_annotations_to_df(annotations, file_type=".fit"):
    """
    Transform the annotations dictionary into a pandas DataFrame.
    """
    annotations_dict = {
        "Image_id": list(annotations["annotations"].keys()),
        "Label": list(annotations["annotations"].values())
    }
    annotations_df = pd.DataFrame(annotations_dict)
    print(annotations_df['Label'])
    try:
        annotations_df[['Label1', 'Label2']] = annotations_df['Label'].str.split(', ', expand=True)
    except Exception:
        # If there is only one label, assign it to Label1 and set Label2 to None
        # annotations_df['Label1'] = annotations_df['Label']
        annotations_df['Label1'] = annotations_df['Label'].apply(lambda x: x)
        annotations_df['Label2'] = None
    print(annotations_df['Label1'])
    annotations_df = annotations_df.drop(columns=['Label'])
    if file_type == ".fits" or file_type == ".fits.gz":
        # Because of file naming convention from the CADC, we need to add 'p' to the Image_id
        annotations_df['Image_id'] = annotations_df['Image_id'].astype(str) + 'p'
    return annotations_df


def add_ground_truth_labels(row, annotations):
    """
    Add ground truth labels to a row of the catalog DataFrame.
    """
    image_id = row["FITS_ID"]
    if image_id in annotations["Image_id"].values:
        print("Image_id: ", image_id)
        label1 = annotations.loc[annotations["Image_id"] == image_id, "Label1"].values[0]
        label2 = annotations.loc[annotations["Image_id"] == image_id, "Label2"].values[0]
        return [label1, label2]
    else:
        print("Image_id not found: ", image_id)
        return [None, None]


def merge_annotations_with_catalog(catalog_df, annotations_df):
    """
    Merge the annotations DataFrame with the catalog DataFrame.
    """
    print("Catalog: ", catalog_df)
    print("Annotations: ", annotations_df)
    catalog_df[["gt_label1", "gt_label2"]] = catalog_df.apply(
        lambda row: add_ground_truth_labels(row, annotations_df), 
        result_type="expand", 
        axis=1
    )
    return catalog_df


def save_to_parquet(df, filename):
    """
    Save the DataFrame to a Parquet file.
    """
    df.to_parquet(filename, compression="gzip", engine="auto")


def process_catalog_and_annotations(catalog_file, annotations_file, output_file, columns):
    """
    Process the catalog and annotations, and save the merged DataFrame to a Parquet file.
    """
    catalog_df = read_catalog(catalog_file, columns)
    annotations = read_annotations(annotations_file)
    annotations_df = transform_annotations_to_df(annotations)
    merged_df = merge_annotations_with_catalog(catalog_df, annotations_df)
    save_to_parquet(merged_df, output_file)


def main():
    parser = argparse.ArgumentParser(description="Process catalog and annotations, and save the merged DataFrame to a Parquet file.")
    parser.add_argument('catalog_file', type=str, help='Path to the catalog HDF5 file')
    parser.add_argument('annotations_file', type=str, help='Path to the annotations JSON file')
    parser.add_argument('output_file', type=str, help='Path to the output Parquet file')

    args = parser.parse_args()
    columns = ["OBJECT_ID", "FITS_ID", "CCD_ID", "ISO0", "BACKGROUND", "ELLIPTICITY", "ELONGATION", "CLASS_STAR", "FLAGS", "EXPTIME"]

    process_catalog_and_annotations(args.catalog_file, args.annotations_file, args.output_file, columns)


# Example usage
if __name__ == "__main__":
    main()