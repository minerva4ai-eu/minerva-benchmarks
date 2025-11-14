import os, sys, mii
PORT=2951

tensor_parallelism = 1

if len(sys.argv) > 1:
    try:
        tensor_parallelism = int(sys.argv[1])  # Get the first argument after the script name
    except ValueError:
        print("Invalid tensor_parallelism value. Using default tensor_parallelism = 1.") # This should exit instead.

print(f"Using tensor_parallelism Value: {tensor_parallelism}")

# Get environment variables
MODEL_PATH = os.getenv("MODEL_PATH", None)
MODEL_NAME = os.getenv("MODEL", None)
PORT = os.getenv("PORT", None) # Override PORT from environment variable set on the top of the script

print(f"DeepSpeed-MII - MODEL_PATH: {MODEL_PATH}")
print(f"DeepSpeed-MII - MODEL_NAME: {MODEL_NAME}")
print(f"DeepSpeed-MII - PORT: {PORT}")


if MODEL_PATH is None:
    print("'MODEL_PATH not found.")
else:
    
    # Serve the model
    client = mii.serve(
        MODEL_PATH,
        deployment_name=MODEL_NAME,
        enable_restful_api=True,
        restful_api_port=PORT,
        tensor_parallel=tensor_parallelism,
        max_length=2048,
    )
