from pathlib import Path
import unittest

import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data" / "processed" / "metadata.csv"
ROLES_PATH = ROOT / "splits" / "tube_roles.csv"


class TestTubeSplits(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.metadata = pd.read_csv(METADATA_PATH)
        cls.roles = pd.read_csv(ROLES_PATH)

        cls.train = cls.metadata[
            cls.metadata["role"] == "train"
        ].reset_index(drop=True)

        cls.validation = cls.metadata[
            cls.metadata["role"] == "validation"
        ].reset_index(drop=True)

        cls.test = cls.metadata[
            cls.metadata["role"] == "test"
        ].reset_index(drop=True)

    def test_expected_tube_counts(self):
        counts = self.roles.groupby("role")["tube_id"].nunique()

        self.assertEqual(counts["train"], 31)
        self.assertEqual(counts["validation"], 10)
        self.assertEqual(counts["test"], 10)

    def test_no_tube_in_multiple_partitions(self):
        self.assertTrue(
            (self.metadata.groupby("tube_id")["role"].nunique() == 1).all()
        )

        train = set(self.train["tube_id"])
        validation = set(self.validation["tube_id"])
        test = set(self.test["tube_id"])

        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(validation.isdisjoint(test))

    def test_all_concentrations_represented_in_training(self):
        all_levels = set(self.metadata["conc"].unique())
        train_levels = set(self.train["conc"].unique())

        self.assertEqual(all_levels, train_levels)

    def test_validation_and_test_concentration_structure(self):
        val_counts = self.roles[
            self.roles["role"] == "validation"
        ].groupby("conc").size()

        test_counts = self.roles[
            self.roles["role"] == "test"
        ].groupby("conc").size()

        self.assertEqual(len(val_counts), 10)
        self.assertEqual(len(test_counts), 10)

        self.assertTrue((val_counts == 1).all())
        self.assertTrue((test_counts == 1).all())

        self.assertEqual(
            set(val_counts.index).intersection(test_counts.index),
            {0.50, 1.00, 1.50, 2.00},
        )

    def test_outer_grouped_cv_no_tube_leakage(self):
        groups = self.train["tube_id"].to_numpy()
        splitter = GroupKFold(n_splits=5)

        for train_idx, val_idx in splitter.split(
            self.train, groups=groups
        ):
            train_tubes = set(groups[train_idx])
            val_tubes = set(groups[val_idx])

            self.assertTrue(train_tubes.isdisjoint(val_tubes))

    def test_inner_grouped_cv_no_tube_leakage(self):
        groups = self.train["tube_id"].to_numpy()
        outer = GroupKFold(n_splits=5)

        for outer_train_idx, _ in outer.split(
            self.train, groups=groups
        ):
            outer_train = self.train.iloc[
                outer_train_idx
            ].reset_index(drop=True)

            inner_groups = outer_train["tube_id"].to_numpy()
            inner = GroupKFold(n_splits=4)

            for train_idx, val_idx in inner.split(
                outer_train, groups=inner_groups
            ):
                train_tubes = set(inner_groups[train_idx])
                val_tubes = set(inner_groups[val_idx])

                self.assertTrue(train_tubes.isdisjoint(val_tubes))

    def test_leave_one_concentration_out_no_tube_leakage(self):
        # Development data only; the test partition is excluded.
        dev = self.metadata[
            self.metadata["role"].isin(["train", "validation"])
        ].reset_index(drop=True)

        for concentration in sorted(dev["conc"].unique()):
            train = dev[dev["conc"] != concentration]
            val = dev[dev["conc"] == concentration]

            train_tubes = set(train["tube_id"])
            val_tubes = set(val["tube_id"])

            self.assertTrue(train_tubes.isdisjoint(val_tubes))

            # All held-out rows must have the requested concentration.
            self.assertTrue(
                (val["conc"] == concentration).all()
            )
    def test_loco_folds_have_no_concentration_or_tube_leakage(self):
        loco_path = ROOT / "splits" / "loco_folds.csv"
        loco = pd.read_csv(loco_path)

        self.assertEqual(loco["fold_id"].nunique(), 27)

        test_tubes = set(
            self.roles.loc[
                self.roles["role"] == "test", "tube_id"
            ]
        )

        for fold_id, fold in loco.groupby("fold_id"):

            train = fold[fold["fold_role"] == "train"]
            val = fold[fold["fold_role"] == "validation"]

            train_tubes = set(train["tube_id"])
            val_tubes = set(val["tube_id"])

            # No tube can occur on both sides.
            self.assertTrue(train_tubes.isdisjoint(val_tubes))

            # The held-out concentration is absent from training.
            held_conc = fold["held_out_conc"].iloc[0]

            self.assertTrue(
                (train["conc"] != held_conc).all()
            )

            # Every validation tube must belong to the held-out level.
            self.assertTrue(
                (val["conc"] == held_conc).all()
            )

            # No test tube is allowed in a LOCO fold.
            self.assertTrue(
                set(fold["tube_id"]).isdisjoint(test_tubes)
            )

            # All 41 development tubes must occur in each fold.
            self.assertEqual(fold["tube_id"].nunique(), 41)


if __name__ == "__main__":
    unittest.main(verbosity=2)