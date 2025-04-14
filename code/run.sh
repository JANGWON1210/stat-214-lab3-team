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

# Run notebooks
echo "Executing notebooks..."
jupyter nbconvert --to notebook --execute --inplace part_1.ipynb
jupyter nbconvert --to notebook --execute --inplace part_2.ipynb

echo "All notebooks executed."
