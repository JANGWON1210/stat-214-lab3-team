#!/bin/bash

# Check if environment exists
if conda info --envs | grep -q "^stat214"; then
    echo "Environment 'stat214' already exists."
else
    echo "Creating conda environment 'stat214'..."
    conda env create -f environment.yaml
fi

# Activate environment
echo "Activating environment..."
conda activate stat214

# Run notebooks sequentially
echo "Executing notebooks..."

jupyter nbconvert --to notebook --execute --inplace 3.2_part_1.ipynb
jupyter nbconvert --to notebook --execute --inplace 3.2_part_2_colab.ipynb
jupyter nbconvert --to notebook --execute --inplace 3.2_part_2_regression.ipynb

echo "All notebooks executed successfully."