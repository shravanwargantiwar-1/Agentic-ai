"""Helper script for YOLO -> ONNX -> OpenVINO flow."""

print('Run: yolo export model=yolo11n.pt format=onnx')
print('Run: mo --input_model yolo11n.onnx')
