import sys
from src import setup
from src import predict

def execute():
    setup.setup_environment()

    if len(sys.argv) != 2:
        print("Usage: python execute.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict.main(image_path)

if __name__ == "__main__":
    execute()