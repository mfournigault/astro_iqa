import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from tensorflow.keras.utils import get_file

import argparse

def read_and_concat_catalogs(catalog_paths: list[str]) -> pd.DataFrame:
    """
    Read multiple Parquet files and concatenate them into a single DataFrame.

    Args:
        catalog_paths (list[str]): List of file paths to the Parquet catalogs.

    Returns:
        pd.DataFrame: Concatenated pandas DataFrame from all Parquet files.
    """
    dataframes = []
    for path in catalog_paths:
        df = pd.read_parquet(path, engine="auto")
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def clean_and_split_catalog(
    catalog: pd.DataFrame,
    label_column: str = "gt_label1",
    drop_columns: list[str] = ["gt_label2"],
    train_fraction: float = 0.8,
    val_fraction: float = 0.5,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean the catalog by dropping NaN from a specific label column, removing unneeded columns,
    and splitting into training, validation, and test sets.

    Args:
        catalog (pd.DataFrame): Source DataFrame.
        label_column (str): Name of the label column to keep.
        drop_columns (list[str]): Columns to drop from the DataFrame.
        train_fraction (float): Fraction of data to keep for training.
        val_fraction (float): Fraction of the remaining data (after train split) for validation.
        random_seed (int): Random seed for reproducibility.

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame): DataFrames for training, validation, and test.
    """
    np.random.seed(random_seed)
    catalog = catalog.dropna(subset=[label_column])
    if drop_columns:
        catalog = catalog.drop(columns=drop_columns)
    
    # Calculate class weights
    class_weights = catalog[label_column].value_counts(normalize=True)
    class_weights = class_weights.to_dict()
    print("Class weights:")
    print(class_weights)
    print("-----------------")

    train_selec = np.random.rand(len(catalog.index)) < train_fraction
    train_df = catalog[train_selec]
    val_test_df = catalog[~train_selec]

    val_selec = np.random.rand(len(val_test_df.index)) < val_fraction
    val_df = val_test_df[val_selec]
    test_df = val_test_df[~val_selec]

    # Convert object columns to string where needed
    for col in ["OBJECT_ID", "FITS_ID"]:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
        if col in val_df.columns:
            val_df[col] = val_df[col].astype(str)
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str)

    return train_df, val_df, test_df, class_weights

def df_to_dataset(
    dataframe: pd.DataFrame,
    label_column: str,
    shuffle: bool = True,
    shuffling_seed: int = 42,
    batch_size: int = 32
) -> tf.data.Dataset:
    """
    Convert a pandas DataFrame into a tf.data.Dataset.

    Args:
        dataframe (pd.DataFrame): Source DataFrame.
        label_column (str): Name of the column containing labels.
        shuffle (bool): Whether to shuffle the dataset.
        shuffling_seed (int): Random seed for shuffling.
        batch_size (int): Number of samples per batch.

    Returns:
        tf.data.Dataset: A tf.data.Dataset containing features and labels.
    """
    df_copy = dataframe.copy()
    labels = df_copy.pop(label_column)
    df_copy = {key: value.to_numpy()[:,tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df_copy), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df_copy)//2, seed=shuffling_seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    return ds

def save_datasets(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    output_dir: str
) -> None:
    """
    Save TensorFlow datasets to the given directory.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        test_dataset (tf.data.Dataset): Test dataset.
        output_dir (str): Directory path where datasets will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.save(os.path.join(output_dir, "training_dataset"))
    val_dataset.save(os.path.join(output_dir, "validation_dataset"))
    test_dataset.save(os.path.join(output_dir, "test_dataset"))


def main() -> None:
    """
    Main function to orchestrate reading, cleaning, splitting, dataset creation, and saving.
    Adjust file paths and parameters as needed.
    """
    parser = argparse.ArgumentParser(description="Data path argument.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/",
        help="Specify the base data path."
    )
    parser.add_argument(
      "--train_fraction",
      type=float,
      default=0.8,
      help="Specify the proportion of training data for the split, between 0 and 1"
    )
    parser.add_argument(
      "--validation_fraction",
      type=float,
      default=0.5,
      help="Specify the proportion of data for the split val/test, between 0 and 1"
    )
    args = parser.parse_args()
    data_path = args.data_path
    train_fraction = args.train_fraction
    validation_fraction = args.validation_fraction

    fm_path = os.path.join(data_path, "for_modeling")
    catalog_paths = [
        os.path.join(fm_path, "objects_catalog_cadc_bronze.parquet.gz"),
        os.path.join(fm_path, "objects_catalog_ngc0869_bronze.parquet.gz"),
        os.path.join(fm_path, "objects_catalog_ngc0896_bronze.parquet.gz"),
        os.path.join(fm_path, "objects_catalog_ngc7000_bronze.parquet.gz")
    ]

    print("Reading and concatening catalogs ...")
    catalog = read_and_concat_catalogs(catalog_paths)
    print("Cleaning and splitting catalog ...")
    train_df, val_df, test_df, class_weights = clean_and_split_catalog(
        catalog=catalog,
        label_column="gt_label1",
        drop_columns=["gt_label2", "OBJECT_ID"],
        train_fraction=train_fraction,
        val_fraction=validation_fraction
    )

    print("Converting datasets ...")
    train_dataset = df_to_dataset(train_df, "gt_label1", True, 42, 128)
    val_dataset = df_to_dataset(val_df, "gt_label1", True, 42, 128)
    test_dataset = df_to_dataset(test_df, "gt_label1", True, 42, 128)

    print("Saving datasets ...")
    save_datasets(train_dataset, val_dataset, test_dataset, fm_path)
    print("Finished.")

if __name__ == "__main__":
    main()

