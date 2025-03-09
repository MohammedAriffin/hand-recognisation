import splitfolders
import os

# Configure paths
input_dir = r"train_data\asl_dataset"
output_dir = r"dataset"
class_subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

# Verify directory structure
assert len(class_subdirs) > 0, "No class subdirectories found!"

# Stratified split with balanced classes
splitfolders.ratio(
    input_dir,
    output=output_dir,
    seed=42,
    ratio=(0.8, 0.1, 0.1),  # Train/Val/Test
    group_prefix=None,  # Preserve alphabetical groupings
    move=False  # Copy instead of move original files
)
