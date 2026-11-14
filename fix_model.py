
import json
import h5py
import shutil

# Original model
original_model = "simple_rnn_imdb.h5"

# Create a compatibility copy
fixed_model = "simple_rnn_imdb_compatible.h5"

shutil.copy2(original_model, fixed_model)

# Open the copied H5 model
with h5py.File(fixed_model, "r+") as f:

    # Read the saved model configuration
    config = f.attrs["model_config"]

    if isinstance(config, bytes):
        config = config.decode("utf-8")

    config = json.loads(config)

    # Remove the unsupported time_major argument
    def fix_config(obj):

        if isinstance(obj, dict):

            obj.pop("time_major", None)

            for value in obj.values():
                fix_config(value)

        elif isinstance(obj, list):

            for item in obj:
                fix_config(item)

    fix_config(config)

    # Save the corrected configuration
    del f.attrs["model_config"]
    f.attrs["model_config"] = json.dumps(config)

print("Model configuration fixed successfully!")
