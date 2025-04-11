try:
    import cupy as np
    import cupy
    gpu_available = True
    print("Using GPU via CuPy on device:", cupy.cuda.get_device_name(cupy.cuda.current_device()))
except ImportError:
    import numpy as np
    gpu_available = False
    print("Using CPU with NumPy (CuPy not available)")

import joblib
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Append the ridge_utils folder to the module search path so we can import from it
sys.path.append("./ridge_utils")
from ridge import bootstrap_ridge  # Use the bootstrap function defined in ridge.py

# ------------------------------------------------------------------------------
# Define a run_ridge() wrapper function for GPU-accelerated ridge regression.
# ------------------------------------------------------------------------------
def run_ridge(X_train, Y_train, X_test, Y_test, alphas=None, cv=15, chunklen=10, nchunks=5):
    if alphas is None:
        alphas = np.logspace(1, 3, 10)
    
    # Define a simple z-score function
    def zs(v):
        std = v.std(0)
        std[std == 0] = 1
        return (v - v.mean(0)) / std

    print("Z-scoring training data...")
    X_train_z = zs(X_train)
    Y_train_z = zs(Y_train)
    print("Z-scoring test data...")
    X_test_z  = zs(X_test)
    Y_test_z  = zs(Y_test)
    
    print("Starting bootstrap ridge regression with cv =", cv)
    wt, corrs, valphas, _, _ = bootstrap_ridge(
        X_train_z, Y_train_z, X_test_z, Y_test_z,
        alphas=np.array(alphas),
        nboots=cv,
        chunklen=chunklen,
        nchunks=nchunks,
        use_corr=True,
        normalpha=False,
        return_wt=True
    )
    print("Bootstrap ridge regression finished.")
    
    print("Computing test predictions and voxel-wise correlations...")
    Y_pred = np.dot(X_test_z, wt)
    voxel_ccs = np.array([
        np.corrcoef(Y_test_z[:, i], Y_pred[:, i])[0, 1] if np.std(Y_pred[:, i]) > 0 else 0
        for i in range(Y_test_z.shape[1])
    ])
    metrics = {
        "mean_cc": np.mean(voxel_ccs),
        "median_cc": np.median(voxel_ccs),
        "top1_cc": np.percentile(voxel_ccs, 99),
        "top5_cc": np.percentile(voxel_ccs, 95),
        "voxel_ccs": voxel_ccs
    }
    
    print("run_ridge() finished. Metrics computed.")
    return wt, metrics

# ------------------------------------------------------------------------------
# Load precomputed embedding results (from Part 1)
# ------------------------------------------------------------------------------
def load_embeddings(data_dir):
    X_bow = joblib.load(data_dir / "X_lagged_BoW.joblib")
    with open(data_dir / "X_lagged_W2V.pkl", "rb") as f:
        X_w2v = pickle.load(f)
    with open(data_dir / "X_lagged_GloVe.pkl", "rb") as f:
        X_glove = pickle.load(f)
    return {"BoW": X_bow, "Word2Vec": X_w2v, "GloVe": X_glove}

# ------------------------------------------------------------------------------
# Load fMRI data from a subject's directory
# ------------------------------------------------------------------------------
def load_fmri(subject_dir):
    fmri_data = {}
    for file in Path(subject_dir).glob("*.npy"):
        story = file.stem
        fmri_data[story] = np.load(file)
    return fmri_data

# ------------------------------------------------------------------------------
# Split stories into training and testing sets (70% / 30%)
# ------------------------------------------------------------------------------
def split_stories(story_list, train_ratio=0.7, random_state=42):
    np.random.seed(random_state)
    stories = sorted(story_list)
    np.random.shuffle(stories)
    split_idx = int(len(stories) * train_ratio)
    return stories[:split_idx], stories[split_idx:]

# ------------------------------------------------------------------------------
# Prepare data by concatenating embedding (X) and fMRI (Y) data across stories
# ------------------------------------------------------------------------------
def prepare_data(stories, embedding_dict, fmri_dict):
    X_list, Y_list = [], []
    for story in stories:
        if story in embedding_dict and story in fmri_dict:
            X_list.append(embedding_dict[story])
            Y_list.append(fmri_dict[story])
        else:
            print(f"Warning: Story {story} is missing in embeddings or fMRI data.")
    if len(X_list) == 0 or len(Y_list) == 0:
        raise ValueError("No overlapping stories found!")
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    return X, Y

# ------------------------------------------------------------------------------
# Compute voxel-wise correlation coefficient (CC)
# ------------------------------------------------------------------------------
def compute_voxel_cc(Y_true, Y_pred):
    n_voxels = Y_true.shape[1]
    voxel_ccs = []
    for i in range(n_voxels):
        true_voxel = Y_true[:, i]
        pred_voxel = Y_pred[:, i]
        if np.std(true_voxel) == 0 or np.std(pred_voxel) == 0:
            cc = 0.0
        else:
            cc = np.corrcoef(true_voxel, pred_voxel)[0, 1]
        voxel_ccs.append(cc)
    return np.array(voxel_ccs)

# ------------------------------------------------------------------------------
# Train and evaluate GPU-accelerated ridge regression model
# ------------------------------------------------------------------------------
def train_and_evaluate_gpu(X_train, Y_train, X_test, Y_test, subject, emb_name):
    alphas = np.logspace(1, 3, 10)
    model, metrics = run_ridge(X_train, Y_train, X_test, Y_test, alphas=alphas, cv=5)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    model_filename = results_dir / f"ridge_{subject}_{emb_name}.pkl"
    with open(model_filename, "wb") as f:
        # Convert model to CPU (if GPU array) before saving:
        try:
            pickle.dump(model.get(), f)
        except AttributeError:
            pickle.dump(model, f)
    print(f"Saved model for {subject} - {emb_name} to {model_filename}")
    print(f"[{subject} - {emb_name}] Evaluation Metrics: {metrics}")
    return metrics

# ------------------------------------------------------------------------------
# Plot the distribution of voxel-wise correlation coefficients (CC)
# ------------------------------------------------------------------------------
def plot_cc_distribution(voxel_ccs, subject, emb_name):
    plt.figure()
    plt.hist(voxel_ccs, bins=50)
    plt.xlabel("Correlation Coefficient (CC)")
    plt.ylabel("Number of Voxels")
    plt.title(f"CC Distribution for {subject} ({emb_name})")
    plot_filename = f"results/cc_distribution_{subject}_{emb_name}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved distribution plot to {plot_filename}")

# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------
def main():
    data_dir = Path("../data")
    subject2_dir = Path("../../tmp_ondemand_ocean_mth240012p_symlink/shared/data/subject2")
    subject3_dir = Path("../../tmp_ondemand_ocean_mth240012p_symlink/shared/data/subject3")
    
    embeddings = load_embeddings(data_dir)
    fmri_subject2 = load_fmri(subject2_dir)
    fmri_subject3 = load_fmri(subject3_dir)
    
    # IMPORTANT: For common stories, we use the keys from the chosen embedding.
    # If you later wish to process other embeddings, adjust accordingly.
    stories_subject2 = set(embeddings["BoW"].keys()) & set(fmri_subject2.keys())
    stories_subject3 = set(embeddings["BoW"].keys()) & set(fmri_subject3.keys())
    common_stories = sorted(list(stories_subject2 & stories_subject3))
    print(f"Total common stories across embeddings and both subjects: {len(common_stories)}")
    
    train_stories, test_stories = split_stories(common_stories, train_ratio=0.7, random_state=42)
    print(f"Training stories: {len(train_stories)}; Testing stories: {len(test_stories)}")
    
    subjects = {"subject2": fmri_subject2, "subject3": fmri_subject3}
    results = {}
    
    # For each subject and each selected embedding type.
    # To test other embeddings, change the list below, e.g., ["BoW", "Word2Vec", "GloVe"]
    for subject, fmri_data in subjects.items():
        results[subject] = {}
        for emb_name in ["BoW"]:
            emb_dict = embeddings[emb_name]
            print(f"Processing {subject} with {emb_name} embeddings...")
            X_train, Y_train = prepare_data(train_stories, emb_dict, fmri_data)
            X_test, Y_test = prepare_data(test_stories, emb_dict, fmri_data)
            metrics = train_and_evaluate_gpu(X_train, Y_train, X_test, Y_test, subject, emb_name)
            results[subject][emb_name] = metrics
    
    best_emb = None
    best_mean_cc = -np.inf
    for emb_name in embeddings.keys():
        if emb_name in results["subject2"] and emb_name in results["subject3"]:
            avg_mean_cc = np.mean([results[subj][emb_name]["mean_cc"] for subj in subjects])
            print(f"Average mean CC for {emb_name}: {avg_mean_cc}")
            if avg_mean_cc > best_mean_cc:
                best_mean_cc = avg_mean_cc
                best_emb = emb_name
    print(f"Best performing embedding overall: {best_emb} with average mean CC {best_mean_cc}")
    
    for subject in subjects.keys():
        voxel_ccs = results[subject][best_emb]["voxel_ccs"]
        plot_cc_distribution(voxel_ccs, subject, best_emb)
        
    # Optionally, save the complete results dictionary for later inspection.
    results_filename = Path("results") / "complete_results.json"
    import json
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete results to {results_filename}")

if __name__ == "__main__":
    main()