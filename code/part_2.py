import joblib
import pickle
from pathlib import Path

def load_embeddings(data_dir):
    """Load all three types of embedding data into a dictionary."""
    X_bow = joblib.load(data_dir / "X_lagged_BoW.joblib")
    with open(data_dir / "X_lagged_W2V.pkl", "rb") as f:
        X_w2v = pickle.load(f)
    with open(data_dir / "X_lagged_GloVe.pkl", "rb") as f:
        X_glove = pickle.load(f)
    return {"BoW": X_bow, "Word2Vec": X_w2v, "GloVe": X_glove}

def check_timepoint_consistency(embeddings):
    """Check if timepoint lengths match across BoW, Word2Vec, and GloVe for each story."""
    print("Checking story timepoint consistency across embeddings...\n")
    story_set = set(embeddings["BoW"]) & set(embeddings["Word2Vec"]) & set(embeddings["GloVe"])
    mismatches = []

    for story in sorted(story_set):
        l_bow = embeddings["BoW"][story].shape[0]
        l_w2v = embeddings["Word2Vec"][story].shape[0]
        l_glove = embeddings["GloVe"][story].shape[0]

        if not (l_bow == l_w2v == l_glove):
            mismatches.append((story, l_bow, l_w2v, l_glove))

    if mismatches:
        print("Found mismatches in the following stories:")
        for story, l_b, l_w, l_g in mismatches:
            print(f"- {story}: BoW={l_b}, Word2Vec={l_w}, GloVe={l_g}")
    else:
        print("All stories have consistent timepoint lengths across embeddings.")

# === Run check ===
DATA_DIR = Path("../data")
embeddings = load_embeddings(DATA_DIR)
check_timepoint_consistency(embeddings)


import os
import numpy as np
import pickle
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

# -------------------------------
# Part1 결과 로딩 (이미 생성되었다고 가정)
# -------------------------------
def load_embeddings(data_dir):
    """Load all three types of embedding data into a dictionary.
    data_dir: Path to the directory with part1 embedding result files."""
    X_bow = joblib.load(data_dir / "X_lagged_BoW.joblib")
    with open(data_dir / "X_lagged_W2V.pkl", "rb") as f:
        X_w2v = pickle.load(f)
    with open(data_dir / "X_lagged_GloVe.pkl", "rb") as f:
        X_glove = pickle.load(f)
    return {"BoW": X_bow, "Word2Vec": X_w2v, "GloVe": X_glove}

# -------------------------------
# fMRI 데이터 로딩
# -------------------------------
def load_fmri(subject_dir):
    """
    subject_dir: Path to subject folder containing .npy files for each story.
    Returns a dictionary mapping story names to fMRI response matrices.
    """
    fmri_data = {}
    for file in subject_dir.glob("*.npy"):
        story = file.stem  # 파일 이름에서 확장자 제거
        fmri_data[story] = np.load(file)
    return fmri_data

# -------------------------------
# 스토리 분할 (70% training, 30% testing)
# -------------------------------
def split_stories(story_list, train_ratio=0.7, random_state=42):
    """
    story_list: list of story identifiers
    Returns: train_stories, test_stories (둘 다 리스트)
    """
    np.random.seed(random_state)
    stories = sorted(story_list)
    np.random.shuffle(stories)
    split_idx = int(len(stories) * train_ratio)
    return stories[:split_idx], stories[split_idx:]

# -------------------------------
# Data Preparation: X (embedding)와 Y (fMRI) 데이터 연결
# -------------------------------
def prepare_data(stories, embedding_dict, fmri_dict):
    """
    stories: list of story names to include.
    embedding_dict: dictionary for one embedding method (e.g., embeddings["BoW"])
    fmri_dict: subject의 fMRI 데이터 dictionary.
    Returns: X, Y where data for each story are concatenated along time axis.
    """
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

# -------------------------------
# Voxel 단위 상관계수 (CC) 계산 함수
# -------------------------------
def compute_voxel_cc(Y_true, Y_pred):
    """
    Y_true, Y_pred: shape (N_time, V_voxels)
    Returns: 1D numpy array of CC (길이 V)
    """
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

# -------------------------------
# Ridge regression 모델 학습 및 평가 함수
# -------------------------------
def train_and_evaluate(X_train, Y_train, X_test, Y_test, subject, emb_name):
    """
    주어진 학습/테스트 데이터를 바탕으로 Ridge 회귀 모형 학습 및 평가
    - 내부 교차 검증으로 최적 alpha 선택 (RidgeCV 이용)
    - 모델 저장 (.pkl 파일, results 폴더에 저장)
    - 테스트 set에 대해 voxel별 CC를 계산하고 요약 통계량을 반환
    """
    # 하이퍼파라미터: alpha 후보 값 (필요에 따라 조정)
    alphas = np.logspace(1, 3, 10)
    ridge = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)
    ridge.fit(X_train, Y_train)

    # 모델 저장 (ridge_utils의 코드 활용할 수 있음  참고)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    model_filename = results_dir / f"ridge_{subject}_{emb_name}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(ridge, f)
    print(f"Saved model for {subject} - {emb_name} to {model_filename}")

    # 테스트 데이터에 대한 예측 및 평가
    Y_pred = ridge.predict(X_test)
    voxel_ccs = compute_voxel_cc(Y_test, Y_pred)
    mean_cc = np.mean(voxel_ccs)
    median_cc = np.median(voxel_ccs)
    top1_cc = np.percentile(voxel_ccs, 99)  # 상위 1% (99번째 백분위)
    top5_cc = np.percentile(voxel_ccs, 95)  # 상위 5% (95번째 백분위)
    metrics = {
        "mean_cc": mean_cc,
        "median_cc": median_cc,
        "top1_cc": top1_cc,
        "top5_cc": top5_cc,
        "voxel_ccs": voxel_ccs
    }
    print(f"[{subject} - {emb_name}] Evaluation Metrics: {metrics}")
    return metrics

# -------------------------------
# CC 분포 플롯 생성 함수
# -------------------------------
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

# -------------------------------
# 메인 파이프라인
# -------------------------------
def main():
    # 경로 설정
    data_dir = Path("../data")  # 파트1의 embedding 결과 파일들이 위치
    subject2_dir = Path("../../tmp_ondemand_ocean_mth240012p_symlink/shared/data/subject2")
    subject3_dir = Path("../../tmp_ondemand_ocean_mth240012p_symlink/shared/data/subject3")
    
    # 파트1 결과 (embedding)을 로딩
    embeddings = load_embeddings(data_dir)
    # embeddings는 {"BoW": {story: embedding_matrix, ...}, "Word2Vec": ..., "GloVe": ...} 형태
    
    # fMRI 데이터 로딩 (각 subject별로)
    fmri_subject2 = load_fmri(subject2_dir)
    fmri_subject3 = load_fmri(subject3_dir)
    
    # 두 subject와 embeddings에 모두 포함된 story들을 선택 (동일 split 적용)
    stories_subject2 = set(embeddings["BoW"].keys()) & set(fmri_subject2.keys())
    stories_subject3 = set(embeddings["BoW"].keys()) & set(fmri_subject3.keys())
    common_stories = sorted(list(stories_subject2 & stories_subject3))
    print(f"Total common stories across embeddings and both subjects: {len(common_stories)}")
    
    # 70% training / 30% test split (동일한 난수 seed 사용)
    train_stories, test_stories = split_stories(common_stories, train_ratio=0.7, random_state=42)
    print(f"Training stories: {len(train_stories)}; Testing stories: {len(test_stories)}")
    
    subjects = {"subject2": fmri_subject2, "subject3": fmri_subject3}
    results = {}
    
    # 각 subject 및 각 embedding 방법에 대해 모델 학습 및 평가 실시
    for subject, fmri_data in subjects.items():
        results[subject] = {}
        # for emb_name, emb_dict in embeddings.items():
        for emb_name in ["BoW"]:
            emb_dict = embeddings[emb_name]
            print(f"Processing {subject} with {emb_name} embeddings...")
            X_train, Y_train = prepare_data(train_stories, emb_dict, fmri_data)
            X_test, Y_test = prepare_data(test_stories, emb_dict, fmri_data)
            metrics = train_and_evaluate(X_train, Y_train, X_test, Y_test, subject, emb_name)
            results[subject][emb_name] = metrics

    # 각 embedding에 대해 두 subject의 mean_cc 평균을 계산하여 최고 성능 embedding 선택
    best_emb = None
    best_mean_cc = -np.inf
    for emb_name in embeddings.keys():
        avg_mean_cc = np.mean([results[subj][emb_name]["mean_cc"] for subj in subjects])
        print(f"Average mean CC for {emb_name}: {avg_mean_cc}")
        if avg_mean_cc > best_mean_cc:
            best_mean_cc = avg_mean_cc
            best_emb = emb_name
    print(f"Best performing embedding overall: {best_emb} with average mean CC {best_mean_cc}")
    
    # 최고 성능 embedding에 대해 각 subject 별 CC 분포 플롯 생성 (추가 분석)
    for subject in subjects.keys():
        voxel_ccs = results[subject][best_emb]["voxel_ccs"]
        plot_cc_distribution(voxel_ccs, subject, best_emb)
        
if __name__ == "__main__":
    main()

