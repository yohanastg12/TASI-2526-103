# final2.py — Hyperparameter Optimization XGBoost Multi-GPU Lengkap (~350–400 baris)
import os, sys, time, json, warnings, functools, pickle
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
print = functools.partial(print, flush=True)

# ─── CONFIG ─────────────────────────────────────────
INPUT_PATH = "/home/jovyan/work/TASI-103/code/2803/data/dataset_final.csv"
OUTPUT_ROOT = "/home/jovyan/work/TASI-103/code/2803/output/"

TARGET_COLUMNS = [
    "argument_clarity(ground_truth)",
    "justifying_persuasiveness(ground_truth)",
    "organizational_structure(ground_truth)",
    "coherence(ground_truth)",
    "essay_length(ground_truth)",
    "grammatical_accuracy(ground_truth)",
    "grammatical_diversity(ground_truth)",
    "lexical_accuracy(ground_truth)",
    "lexical_diversity(ground_truth)",
    "punctuation_accuracy(ground_truth)",
]
DROP_COLUMNS = ["graph", "Question", "Essay", "image_number", "Type"]
TRAIT_LABELS = [t.replace("(ground_truth)","") for t in TARGET_COLUMNS]

N_FOLDS = 5
N_TRIALS = 100
RANDOM_STATE = 42

# ─── HELPERS ─────────────────────────────────────
def timestamp(): return datetime.now().strftime("%H:%M:%S")
def banner(title,width=80):
    print(f"\n{'='*width}\n  [{timestamp()}] {title}\n{'='*width}")
def sub_banner(title): print(f"\n  --- [{timestamp()}] {title} ---")
def progress_bar(current,total,width=40,prefix="",suffix=""):
    pct=current/total*100
    filled=int(width*current//total)
    bar="█"*filled + "░"*(width-filled)
    print(f"\r  {prefix} |{bar}| {pct:5.1f}% ({current}/{total}) {suffix}",end="")
    if current==total: print()

def get_gpu_ids():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES","0")
    ids = list(range(len(visible.split(','))))
    return ids[:4]  # pakai maksimal 4 GPU

# ─── LOAD DATA ─────────────────────────────────────
def load_data():
    banner("STEP 0 — LOADING DATA")
    t0=time.time()
    df=pd.read_csv(INPUT_PATH)
    present_targets=[c for c in TARGET_COLUMNS if c in df.columns]
    cols_to_drop=[c for c in DROP_COLUMNS+present_targets if c in df.columns]
    X=df.drop(columns=cols_to_drop).select_dtypes(include=[np.number]).fillna(0)
    Y=df[present_targets].fillna(df[present_targets].median())
    labels=[t.replace("(ground_truth)","") for t in present_targets]

    # Save summary CSV
    csv_dir=os.path.join(OUTPUT_ROOT,"csv")
    os.makedirs(csv_dir,exist_ok=True)
    pd.DataFrame({
        "feature":X.columns,
        "mean":X.mean().values,
        "std":X.std().values,
        "min":X.min().values,
        "max":X.max().values,
        "non_zero_pct":(X!=0).mean().values*100
    }).to_csv(os.path.join(csv_dir,"data_feature_summary.csv"),index=False)
    pd.DataFrame({
        "trait":labels,
        "mean":[Y.iloc[:,i].mean() for i in range(len(labels))],
        "std":[Y.iloc[:,i].std() for i in range(len(labels))],
        "min":[Y.iloc[:,i].min() for i in range(len(labels))],
        "max":[Y.iloc[:,i].max() for i in range(len(labels))]
    }).to_csv(os.path.join(csv_dir,"data_target_summary.csv"),index=False)
    print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features in {time.time()-t0:.1f}s")
    return X,Y,labels

# ─── OPTUNA OBJECTIVE ─────────────────────────────
def create_objective(X_vals,y_vals,n_folds,gpu_id,trait):
    def objective(trial):
        params={
            "learning_rate":trial.suggest_float("learning_rate",0.005,0.2,log=True),
            "subsample":trial.suggest_float("subsample",0.8,1.0),
            "max_leaves":trial.suggest_int("max_leaves",10,200),
            "max_depth":trial.suggest_int("max_depth",5,30),
            "gamma":trial.suggest_float("gamma",0.0,0.02),
            "colsample_bytree":trial.suggest_float("colsample_bytree",0.8,1.0),
            "min_child_weight":trial.suggest_int("min_child_weight",0,10)
        }
        kf=KFold(n_splits=n_folds,shuffle=True,random_state=RANDOM_STATE)
        rmse_list=[]
        for idx,(tr_idx,va_idx) in enumerate(kf.split(X_vals)):
            X_tr,X_va=X_vals[tr_idx],X_vals[va_idx]
            y_tr,y_va=y_vals[tr_idx],y_vals[va_idx]
            model=XGBRegressor(
                n_estimators=500,
                tree_method="hist",
                importance_type="gain",
                device=f"cuda:{gpu_id}",
                verbosity=0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                **params
            )
            model.fit(X_tr,y_tr,eval_set=[(X_va,y_va)],verbose=False)
            y_pred=model.predict(X_va)
            rmse_list.append(float(np.sqrt(mean_squared_error(y_va,y_pred))))
        return np.mean(rmse_list)
    return objective

# ─── RUN HPO PER TRAIT ─────────────────────────────
def run_optuna_for_trait(args):
    X_vals,X_cols,y_vals,trait,gpu_id,n_trials,n_folds,output_dir=args
    csv_dir=os.path.join(output_dir,"csv")
    study_dir=os.path.join(output_dir,"studies")
    os.makedirs(csv_dir,exist_ok=True)
    os.makedirs(study_dir,exist_ok=True)
    study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(create_objective(X_vals,y_vals,n_folds,gpu_id,trait),n_trials=n_trials,show_progress_bar=False)
    # Save trial history & pickle
    with open(os.path.join(study_dir,f"optuna_study_{trait}.pkl"),"wb") as f: pickle.dump(study,f)
    study.trials_dataframe().to_csv(os.path.join(csv_dir,f"optuna_trials_{trait}.csv"),index=False)
    best=study.best_params.copy()
    best.update({"trait":trait,"best_rmse_cv":study.best_value,"gpu_id":gpu_id,"n_trials":n_trials,"n_folds":n_folds})
    # Save progress CSV per trait
    pd.DataFrame([best]).to_csv(os.path.join(csv_dir,f"progress_{trait}.csv"),index=False)
    # Save RMSE chart per trait
    fig,ax=plt.subplots(figsize=(8,4))
    vals=[t.value for t in study.trials if t.value is not None]
    best_so_far=np.minimum.accumulate(vals)
    ax.plot(vals,alpha=0.3,label="Trial RMSE")
    ax.plot(best_so_far,label="Best so far",color="red",linewidth=2)
    ax.set_xlabel("Trial");ax.set_ylabel("CV RMSE");ax.set_title(f"{trait} RMSE History")
    ax.legend();plt.tight_layout();plt.savefig(os.path.join(csv_dir,f"rmse_history_{trait}.png"),dpi=150)
    plt.close()
    return best

# ─── MAIN ─────────────────────────────────────────────
def main():
    T0=time.time()
    gpu_ids=get_gpu_ids()
    X,Y,labels=load_data()
    X_vals=X.values
    tasks=[(X_vals,list(X.columns),Y.iloc[:,i].values,labels[i],gpu_ids[i%len(gpu_ids)],N_TRIALS,N_FOLDS,OUTPUT_ROOT) for i in range(len(labels))]
    all_results=[]
    banner("STEP 1 — OPTUNA HPO Multi-GPU")
    completed=0
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures={pool.submit(run_optuna_for_trait,t):t[3] for t in tasks}
        for fut in as_completed(futures):
            res=fut.result()
            all_results.append(res)
            completed+=1
            progress_bar(completed,len(labels),prefix="STEP1 Progress",suffix=f"✓ {res['trait']} RMSE={res['best_rmse_cv']:.4f}")
    # STEP2 — Extract best hyperparameters, save CSV & JSON
    banner("STEP 2 — EXTRACT BEST HYPERPARAMETERS")
    hp_df=pd.DataFrame(all_results)
    csv_dir=os.path.join(OUTPUT_ROOT,"csv"); os.makedirs(csv_dir,exist_ok=True)
    hp_df.to_csv(os.path.join(csv_dir,"best_hyperparameters.csv"),index=False)
    hp_json={row['trait']:{k:v for k,v in row.items() if k!="trait"} for idx,row in hp_df.iterrows()}
    with open(os.path.join(csv_dir,"best_hyperparameters.json"),"w") as f: json.dump(hp_json,f,indent=2)
    # RMSE Bar chart per trait
    viz_dir=os.path.join(OUTPUT_ROOT,"viz"); os.makedirs(viz_dir,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,6))
    ax.barh([t[:20] for t in hp_df['trait']],hp_df['best_rmse_cv'],color="#3b82f6")
    ax.set_xlabel("Best CV RMSE");ax.set_title("Optuna Best CV RMSE per Trait")
    plt.tight_layout();plt.savefig(os.path.join(viz_dir,"best_rmse_per_trait.png"),dpi=150);plt.close()
    print(f"STEP1-2 DONE, elapsed {time.time()-T0:.1f}s")

if __name__=="__main__":
    main()