import openvino as ov

# Create an OpenVINO Core object
core = ov.Core()

# Get the list of available devices
devices = core.available_devices

print("✅ Available OpenVINO devices:")
for device in devices:
    try:
        full_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"  - {device}: {full_name}")
    except Exception:
        print(f"  - {device}")
