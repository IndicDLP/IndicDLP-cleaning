import pytest
from utils.metrics import MetricsEvaluator
from PIL import Image
import tempfile
import os

@pytest.fixture
def dummy_dataset(tmp_path):
    # Create dummy image
    image_path = tmp_path / "images"
    image_path.mkdir()
    img = Image.new("RGB", (100, 200))
    img.save(image_path / "doc1.jpg")

    # Create dummy ground truth
    gt_path = tmp_path / "gts"
    gt_path.mkdir()
    (gt_path / "doc1.txt").write_text("0 0.5 0.5 0.2 0.4\n")

    # Create dummy prediction
    pred_path = tmp_path / "preds"
    pred_path.mkdir()
    (pred_path / "doc1.txt").write_text("0 0.5 0.5 0.2 0.4 0.9\n")

    return gt_path, pred_path, image_path

def test_evaluate_image(dummy_dataset):
    gt_dir, pred_dir, image_dir = dummy_dataset
    evaluator = MetricsEvaluator(gt_dir, pred_dir, image_dir)
    ap07, ap08 = evaluator.evaluate_image("doc1")
    assert 0.0 <= ap07 <= 1.0
    assert 0.0 <= ap08 <= 1.0

def test_evaluate_all(dummy_dataset, tmp_path):
    gt_dir, pred_dir, image_dir = dummy_dataset
    save_path = tmp_path / "result.csv"
    evaluator = MetricsEvaluator(gt_dir, pred_dir, image_dir)
    results = evaluator.evaluate_all(save_path)
    assert len(results) == 1
    assert save_path.exists()