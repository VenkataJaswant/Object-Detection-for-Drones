from darknet2pytorch import Darknet
import torch

# Path to the YOLOv4-Tiny .cfg and .weights files
cfg_path = "yolov4-tiny.cfg"
weights_path = "yolov4-tiny.weights"

# Load the Darknet model
model = Darknet(cfg_path)
model.load_weights(weights_path)

# Convert the model to PyTorch format
model.eval()  # Set the model to evaluation mode

# Save the model in PyTorch format
output_path = "yolov4-tiny.pt"
torch.save(model.state_dict(), output_path)

print(f"Model successfully converted and saved to {output_path}")