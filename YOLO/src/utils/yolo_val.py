from ultralytics import YOLO

# Load model
model = YOLO("/home/sahithi_kukkala/indicDLP/yolov10_scratch_indic/YOLOv10x-1024imgsz-AugScratch5/weights/best.pt")

# Run validation
results = model.val(
    data="/home/sahithi_kukkala/indicDLP/data/indic_data/data.yaml",
    imgsz=1024,
    batch=16,
    device="0"
)

# Extract mAP@75
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
print(metrics.box.map75) # map75
print(metrics.box.map50)
print(metrics.box.map) # map75
 # map75
