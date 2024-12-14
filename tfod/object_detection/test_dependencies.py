import importlib
import subprocess

# List of dependencies to check
dependencies = {
    "tensorflow": "tensorflow",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "matplotlib": "matplotlib",
    "pillow": "PIL",
    "protobuf": "google.protobuf",
    "tensorflow-hub": "tensorflow_hub",
    "tensorflow-addons": "tensorflow_addons"
}

print("Checking required dependencies...\n")

for dep_name, import_name in dependencies.items():
    try:
        # Dynamically import the dependency
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "Version not found")
        print(f"[OK] {dep_name}: {version}")
    except ImportError:
        print(f"[MISSING] {dep_name}: Not installed. Installing...")
        try:
            subprocess.check_call(["pip", "install", dep_name])
            print(f"[INSTALLED] {dep_name} installed successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to install {dep_name}: {str(e)}")

print("\nDependency check complete.")
