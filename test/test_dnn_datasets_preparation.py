import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
from dnn_datasets_preparation import (
    read_and_concat_catalogs,
    clean_and_split_catalog,
    df_to_dataset
)

class TestDnnDatasetsPreparation(unittest.TestCase):
    def setUp(self) -> None:
        # Create a small DataFrame for testing
        self.df = pd.DataFrame({
            "OBJECT_ID": ["id1", "id2", "id3", "id4"],
            "gt_label1": [0, 1, 0, 1],
            "num_var": [10.5, 20.1, 5.2, 15.0],
            "str_var": ["A", "B", "C", "A"],
            "gt_label2": [None, None, None, None]
        })

    def test_clean_and_split_catalog(self) -> None:
        train_df, val_df, test_df = clean_and_split_catalog(
            catalog=self.df,
            label_column="gt_label1",
            drop_columns=["gt_label2"],
            random_seed=999
        )
        self.assertFalse(train_df["gt_label1"].isna().any())
        self.assertNotIn("gt_label2", train_df.columns)

    def test_df_to_dataset(self) -> None:
        ds = df_to_dataset(
            dataframe=self.df,
            label_column="gt_label1",
            shuffle=False,
            batch_size=2
        )
        self.assertIsInstance(ds, tf.data.Dataset)
        for features, labels in ds:
            self.assertEqual(labels.shape[0], 2)

if __name__ == "__main__":
    unittest.main()