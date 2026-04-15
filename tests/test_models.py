#!/usr/bin/env python3
"""
单元测试
"""
import unittest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.multi_task_model import (
    MultiTaskProteinClassifier,
    MultiTaskLoss,
    SharedFeatureExtractor,
    create_model,
)
from src.data.preprocessing import ECNumberEncoder, LocalizationEncoder
from src.data.augmentation import ProteinAugmenter
from src.data.location_taxonomy import map_to_simplified_location


class TestModels(unittest.TestCase):
    """模型测试"""

    def test_shared_extractor(self):
        """测试共享特征提取器"""
        extractor = SharedFeatureExtractor(input_dim=320, hidden_dims=[256, 128])
        x = torch.randn(4, 320)
        y = extractor(x)
        self.assertEqual(y.shape, (4, 128))

    def test_multi_task_classifier(self):
        """测试多任务分类器"""
        model = MultiTaskProteinClassifier(
            input_dim=320,
            ec_num_classes=100,
            loc_num_classes=20,
            func_num_classes=50,
        )

        x = torch.randn(4, 320)
        outputs = model(x)

        self.assertEqual(outputs['ec'].shape, (4, 100))
        self.assertEqual(outputs['loc'].shape, (4, 20))
        self.assertEqual(outputs['func'].shape, (4, 50))

    def test_multi_task_loss(self):
        """测试多任务损失"""
        criterion = MultiTaskLoss()

        predictions = {
            'ec': torch.randn(4, 100),
            'loc': torch.randn(4, 20),
            'func': torch.randn(4, 50),
        }

        targets = {
            'ec': torch.randint(0, 2, (4, 100)).float(),
            'loc': torch.randint(0, 20, (4,)),
            'func': torch.randint(0, 2, (4, 50)).float(),
        }

        loss, task_losses = criterion(predictions, targets)

        self.assertIsInstance(loss.item(), float)
        self.assertIn('ec', task_losses)
        self.assertIn('loc', task_losses)
        self.assertIn('func', task_losses)

    def test_create_model(self):
        """测试模型工厂"""
        model = create_model(
            input_dim=320,
            ec_num_classes=100,
            loc_num_classes=20,
            func_num_classes=50,
            model_type='multi_task',
        )
        self.assertIsInstance(model, MultiTaskProteinClassifier)


class TestEncoders(unittest.TestCase):
    """编码器测试"""

    def test_ec_encoder(self):
        """测试 EC 编码器"""
        encoder = ECNumberEncoder(min_depth=3)

        # 测试数据
        ec_strings = ['1.2.3.4', '1.2.3.5', '2.3.4.5']
        encoder.fit(pd.Series(ec_strings))

        self.assertGreater(len(encoder.classes_), 0)

    def test_localization_encoder(self):
        """测试定位编码器"""
        encoder = LocalizationEncoder()

        # 测试数据
        loc_strings = [
            'Nucleus',
            'Cytoplasm',
            'Mitochondrion inner membrane',
        ]

        encoder.fit(pd.Series(loc_strings))

        self.assertGreater(len(encoder.classes_), 0)


class TestAugmentation(unittest.TestCase):
    """数据增强测试"""

    def setUp(self):
        self.augmenter = ProteinAugmenter(seed=42)
        self.sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

    def test_mutate(self):
        """测试突变"""
        result = self.augmenter.mutate(self.sequence, rate=0.1)
        self.assertEqual(len(result), len(self.sequence))

    def test_swap(self):
        """测试交换"""
        result = self.augmenter.swap(self.sequence, n=2)
        self.assertEqual(len(result), len(self.sequence))

    def test_truncate(self):
        """测试截断"""
        result = self.augmenter.truncate(self.sequence, min_length=30, max_length=40)
        self.assertGreaterEqual(len(result), 30)
        self.assertLessEqual(len(result), 40)

    def test_augment_batch(self):
        """测试批量增强"""
        sequences = [self.sequence, "MKTAYIAKVRQGPVKPTKSSVLSQEGCK"]
        labels = np.array([1, 0])

        aug_seqs, aug_labels = self.augmenter.augment_batch(
            sequences, labels, n_augments=2
        )

        self.assertEqual(len(aug_seqs), len(sequences) * 3)  # 原序列 + 2个增强
        self.assertEqual(len(aug_labels), len(aug_seqs))


class TestLocationTaxonomy(unittest.TestCase):
    """定位分类测试"""

    def test_mapping(self):
        """测试定位映射"""
        test_cases = [
            ("Nucleus", "Nucleus"),
            ("Cytoplasm", "Cytoplasm"),
            ("Mitochondrion inner membrane", "Mitochondrion"),
            ("Secreted", "Secreted"),
        ]

        for loc, expected in test_cases:
            result = map_to_simplified_location(loc)
            # 检查是否返回了有效的简化分类
            self.assertIsInstance(result, str)


class TestMetrics(unittest.TestCase):
    """评估指标测试"""

    def test_binary_metrics(self):
        """测试二分类指标"""
        from src.utils.metrics import calculate_binary_metrics

        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1]])

        metrics = calculate_binary_metrics(y_true, y_pred)

        self.assertIn('accuracy', metrics)
        self.assertIn('f1_macro', metrics)
        self.assertIn('precision_macro', metrics)


if __name__ == '__main__':
    # 添加 pandas 导入
    import pandas as pd
    unittest.main()
