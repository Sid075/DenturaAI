import unittest
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

class TestModelEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        self.class_names = ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Mouth_Ulcer", "Tooth_Discoloration"]
        # Known synthetic truth and prediction arrays for 6 classes
        self.y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5]) # 13 samples
        self.y_pred = np.array([0, 0, 2, 1, 0, 2, 2, 2, 0, 3, 3, 4, 5])

    def test_accuracy_calculation(self):
        """Test accuracy calculation against manual ground truth."""
        # Correct predictions:
        # 0==0 (yes), 0==0 (yes), 0==2 (no), 1==1 (yes), 1==0 (no),
        # 2==2 (yes), 2==2 (yes), 2==2 (yes), 2==0 (no),
        # 3==3 (yes), 3==3 (yes), 4==4 (yes), 5==5 (yes)
        # Total correct: 10 / 13
        acc = accuracy_score(self.y_true, self.y_pred)
        expected = 10.0 / 13.0
        self.assertAlmostEqual(acc, expected, places=4)

    def test_confusion_matrix_orientation(self):
        """
        Verify confusion matrix orientation:
        Rows = True Labels, Columns = Predicted Labels
        """
        cm = confusion_matrix(self.y_true, self.y_pred, labels=list(range(6)))
        self.assertEqual(cm.shape, (6, 6))

        # Class 0 (Calculus): 3 true samples. 2 predicted as 0, 1 predicted as 2 (Gingivitis).
        self.assertEqual(cm[0, 0], 2) # True=0, Pred=0
        self.assertEqual(cm[0, 2], 1) # True=0, Pred=2 (Row 0, Col 2)
        self.assertEqual(np.sum(cm[0, :]), 3) # Row sum = True support

        # Class 1 (Caries): 2 true samples. 1 predicted as 1, 1 predicted as 0 (Calculus).
        self.assertEqual(cm[1, 1], 1)
        self.assertEqual(cm[1, 0], 1)
        self.assertEqual(np.sum(cm[1, :]), 2)

    def test_confusion_matrix_row_normalization(self):
        """Verify row-normalization sums to 1.0 (or 0 for unrepresented classes)."""
        cm = confusion_matrix(self.y_true, self.y_pred, labels=list(range(6)))
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        # For every row with support > 0, sum should equal 1.0
        for i in range(6):
            if np.sum(cm[i, :]) > 0:
                self.assertAlmostEqual(float(np.sum(cm_norm[i, :])), 1.0, places=4)

    def test_class_order_consistency(self):
        """Ensure class order remains fixed and consistent."""
        labels = list(range(len(self.class_names)))
        prec, rec, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=labels, average=None, zero_division=0
        )
        self.assertEqual(len(prec), len(self.class_names))
        self.assertEqual(len(rec), len(self.class_names))
        self.assertEqual(len(f1), len(self.class_names))
        self.assertEqual(len(support), len(self.class_names))

        # Check support matches ground truth counts
        # Calculus: 3, Caries: 2, Gingivitis: 4, Hypodontia: 2, Mouth_Ulcer: 1, Tooth_Discoloration: 1
        expected_support = [3, 2, 4, 2, 1, 1]
        self.assertListEqual(support.tolist(), expected_support)

    def test_missing_classes_handling(self):
        """Verify zero-division and missing class stability."""
        # Array with only classes 0 and 1, missing 2, 3, 4, 5
        y_true_partial = np.array([0, 0, 1])
        y_pred_partial = np.array([0, 1, 1])
        labels = list(range(6))

        prec, rec, f1, support = precision_recall_fscore_support(
            y_true_partial, y_pred_partial, labels=labels, average=None, zero_division=0
        )
        self.assertEqual(len(prec), 6)
        # Missing classes (index 2, 3, 4, 5) must have 0 support and 0 metric values, not error
        for i in [2, 3, 4, 5]:
            self.assertEqual(support[i], 0)
            self.assertEqual(prec[i], 0.0)
            self.assertEqual(rec[i], 0.0)
            self.assertEqual(f1[i], 0.0)

    def test_empty_dataset_handling(self):
        """Verify behavior on empty test inputs."""
        y_true_empty = np.array([])
        y_pred_empty = np.array([])
        self.assertEqual(len(y_true_empty), 0)

    def test_balanced_accuracy(self):
        """Verify balanced accuracy correctly computes macro average of recall per class."""
        bal_acc = balanced_accuracy_score(self.y_true, self.y_pred)
        _, rec, _, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=list(range(6)), average=None, zero_division=0
        )
        expected_bal_acc = float(np.mean(rec))
        self.assertAlmostEqual(bal_acc, expected_bal_acc, places=4)


if __name__ == '__main__':
    unittest.main()
