import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    # Check and install torch and torchvision
    try:
        import torch
        import torchvision
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
    except ImportError:
        print("PyTorch or Torchvision not installed. Installing...")
        install("torch==1.8.1+cpu")
        install("torchvision==0.9.1+cpu")
        import torch
        import torchvision
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)

    # Check and install pandas
    try:
        import pandas as pd
        print("Pandas version:", pd.__version__)
    except ImportError:
        print("Pandas not installed. Installing...")
        install("pandas")
        import pandas as pd
        print("Pandas version:", pd.__version__)

    # Check and install Pillow
    try:
        from PIL import Image
        print("Pillow version:", Image.__version__)
    except ImportError:
        print("Pillow not installed. Installing...")
        install("pillow")
        from PIL import Image
        print("Pillow version:", Image.__version__)

if __name__ == "__main__":
    main()
