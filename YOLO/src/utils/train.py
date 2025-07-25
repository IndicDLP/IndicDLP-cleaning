
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

# Model setup
model = YOLO("yolov10x.yaml")  # This will automatically initialize the model config

# W&B setup
wandb.login(key="77139d553fbf0bcbbc88e3d429cbaccf08d2e1e6")

# Descriptive run name and grouping
run = wandb.init(
    project='yolov10_scratch_indic',
    name='YOLOv10x-1024imgsz-AugScratch',
    group='YOLOv10x-indic-experiments',  # Logical group for related runs
    notes='Training from scratch with custom aug: shear, erasing, hsv_s, no mosaic',
    tags=['scratch', 'yolov10x', 'indic', '1024'],
    config={
        "model": "yolov10x",
        "imgsz": 1024,
        "batch": 100,
        "device": "4,5,6,7",
        "data": "/home/sahithi_kukkala/indicDLP/data/indic_data/data.yaml",
        "patience": 5
    }
)

# Add W&B callback
add_wandb_callback(model, enable_model_checkpointing=True)

# Training
results = model.train(
    data=wandb.config.data,
    project=wandb.run.project,
    name=wandb.run.name,
    device=wandb.config.device,
    batch=wandb.config.batch,
    imgsz=wandb.config.imgsz,
    patience=wandb.config.patience,
)

wandb.finish()