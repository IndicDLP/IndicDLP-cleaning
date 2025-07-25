import pytest
from utils.preprocess import Preprocessor
from pathlib import Path
from PIL import Image
import tempfile

def test_yolo_to_xyxy():
    box = [0.5, 0.5, 0.2, 0.4]
    img_w, img_h = 100, 200
    result = Preprocessor.yolo_to_xyxy(box, img_w, img_h)
    expected = [40.0, 60.0, 60.0, 140.0]
    assert result == expected

def test_load_yolo_file_no_conf(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("0 0.5 0.5 0.2 0.4\n1 0.4 0.4 0.1 0.2\n")
    boxes, classes, scores = Preprocessor.load_yolo_file(f)
    assert boxes == [[0.5, 0.5, 0.2, 0.4], [0.4, 0.4, 0.1, 0.2]]
    assert classes == [0, 1]
    assert scores is None

def test_load_yolo_file_with_conf(tmp_path):
    f = tmp_path / "pred.txt"
    f.write_text("0 0.5 0.5 0.2 0.4 0.9\n1 0.4 0.4 0.1 0.2 0.8\n")
    boxes, classes, scores = Preprocessor.load_yolo_file(f, has_conf=True)
    assert boxes == [[0.5, 0.5, 0.2, 0.4], [0.4, 0.4, 0.1, 0.2]]
    assert classes == [0, 1]
    assert scores == [0.9, 0.8]
    
def test_get_image_size(tmp_path):
    img_path = tmp_path / "image.jpg"
    img = Image.new("RGB", (128, 256))
    img.save(img_path)
    w, h = Preprocessor.get_image_size(img_path)
    assert w == 128 and h == 256