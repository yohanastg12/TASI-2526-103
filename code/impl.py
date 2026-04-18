# impl2_safe.py — Retrain Best XGBoost Models & Extract Feature Importance Multi-GPU (XGBoost ≥3.1)
import os, json, time, warnings, functools, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, f1_score
from sklearn.preprocessing import KBinsDiscretizer
from xgboost import XGBRegressor
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# ─── CONFIGURATION ─────────────────────────────
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
DROP_COLUMNS = ["graph","Question","Essay","image_number","Type"]
TRAIT_LABELS = [t.replace("(ground_truth)","") for t in TARGET_COLUMNS]

N_FOLDS_EVAL = 5
N_BINS_F1 = 5
TOP_K = 30
RANDOM_STATE = 42

# ─── HELPERS ─────────────────────────────────
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
def normalize_series(s):
    mn,mx=s.min(),s.max()
    return s if (mx-mn)<1e-12 else (s-mn)/(mx-mn)
def discretize_for_f1(y_true,y_pred,n_bins=N_BINS_F1):
    kbd=KBinsDiscretizer(n_bins=n_bins,encode="ordinal",strategy="quantile")
    y_all=np.concatenate([y_true,y_pred]).reshape(-1,1)
    kbd.fit(y_all)
    yt_bin=kbd.transform(y_true.reshape(-1,1)).ravel().astype(int)
    yp_bin=kbd.transform(y_pred.reshape(-1,1)).ravel().astype(int)
    return f1_score(yt_bin,yp_bin,average="macro",zero_division=0), f1_score(yt_bin,yp_bin,average="weighted",zero_division=0)
def get_gpu_ids():
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","0")
    return list(range(len(visible.split(","))))[:4]

# ─── LOAD DATA ─────────────────────────────
def load_data():
    banner("STEP 0 — LOADING DATA & HYPERPARAMETERS")
    df=pd.read_csv(INPUT_PATH)
    present_targets=[c for c in TARGET_COLUMNS if c in df.columns]
    cols_to_drop=[c for c in DROP_COLUMNS+present_targets if c in df.columns]
    X=df.drop(columns=cols_to_drop).select_dtypes(include=[np.number]).fillna(0)
    Y=df[present_targets].fillna(df[present_targets].median())
    labels=[t.replace("(ground_truth)","") for t in present_targets]
    hp_path=os.path.join(OUTPUT_ROOT,"csv","best_hyperparameters.json")
    with open(hp_path,"r") as f: best_params=json.load(f)
    # Pastikan gpu_id dihapus dari dict untuk XGBoost ≥3.1
    for k in best_params.keys():
        if "gpu_id" in best_params[k]: del best_params[k]["gpu_id"]
    return X,Y,labels,best_params

# ─── WORKER: RETRAIN + FEATURE IMPORTANCE ─────────
def retrain_trait(args_tuple):
    X_vals,X_cols,y_vals,trait,params_dict,gpu_id,n_folds,n_bins_f1,top_k,output_root=args_tuple
    csv_dir=os.path.join(output_root,"csv")
    model_dir=os.path.join(output_root,"models")
    viz_dir=os.path.join(output_root,"viz")
    for d in [csv_dir,model_dir,viz_dir]: os.makedirs(d,exist_ok=True)

    # Clean params → hapus key gpu_id
    xgb_hp={k:v for k,v in params_dict.items() if k!="gpu_id"}
    base_params=dict(
        **xgb_hp,
        n_estimators=500,
        tree_method="hist",
        importance_type="gain",
        device=f"cuda:{gpu_id}",
        verbosity=0,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    # K-Fold CV
    kf=KFold(n_splits=n_folds,shuffle=True,random_state=RANDOM_STATE)
    oof_preds=np.zeros(len(y_vals))
    fold_metrics=[]
    for fold_i,(tr_idx,va_idx) in enumerate(kf.split(X_vals)):
        model_cv=XGBRegressor(**base_params)
        model_cv.fit(X_vals[tr_idx],y_vals[tr_idx])
        preds=model_cv.predict(X_vals[va_idx])
        oof_preds[va_idx]=preds
        fold_metrics.append({
            "fold":fold_i+1,
            "r2":r2_score(y_vals[va_idx],preds),
            "mae":mean_absolute_error(y_vals[va_idx],preds),
            "rmse":float(np.sqrt(mean_squared_error(y_vals[va_idx],preds)))
        })

    # OOF metrics
    oof_r2=r2_score(y_vals,oof_preds)
    oof_mae=mean_absolute_error(y_vals,oof_preds)
    oof_rmse=float(np.sqrt(mean_squared_error(y_vals,oof_preds)))
    f1_m,f1_w=discretize_for_f1(y_vals,oof_preds,n_bins_f1)
    pd.DataFrame(fold_metrics).to_csv(os.path.join(csv_dir,f"cv_folds_{trait}.csv"),index=False)

    # Full retrain
    model_full=XGBRegressor(**base_params)
    model_full.fit(X_vals,y_vals)
    y_pred_full=model_full.predict(X_vals)
    train_r2=r2_score(y_vals,y_pred_full)
    train_mae=mean_absolute_error(y_vals,y_pred_full)
    train_rmse=float(np.sqrt(mean_squared_error(y_vals,y_pred_full)))
    model_full.save_model(os.path.join(model_dir,f"best_model_{trait}.json"))

    # Feature importance
    imp_df=pd.DataFrame({"feature":X_cols,"importance":model_full.feature_importances_}).sort_values("importance",ascending=False)
    imp_df["importance_norm"]=normalize_series(imp_df["importance"])
    imp_df["rank"]=range(1,len(imp_df)+1)
    imp_df.to_csv(os.path.join(csv_dir,f"importance_best_{trait}.csv"),index=False)
    imp_df.head(top_k).to_csv(os.path.join(csv_dir,f"importance_top{top_k}_{trait}.csv"),index=False)

    # Bar chart Top-K
    fig,ax=plt.subplots(figsize=(8,4))
    ax.barh(imp_df.head(top_k)["feature"],imp_df.head(top_k)["importance_norm"],color="#f59e0b")
    ax.set_xlabel("Normalized Importance");ax.set_title(f"{trait} Top-{top_k} Features")
    plt.tight_layout();plt.savefig(os.path.join(viz_dir,f"bar_top{top_k}_{trait}.png"),dpi=150);plt.close()

    return {
        "trait":trait,
        "oof_preds":oof_preds,
        "y_pred_full":y_pred_full,
        "cv_r2":oof_r2,
        "cv_mae":oof_mae,
        "cv_rmse":oof_rmse,
        "cv_f1_macro":f1_m,
        "cv_f1_weighted":f1_w,
        "train_r2":train_r2,
        "train_mae":train_mae,
        "train_rmse":train_rmse,
        "top_features":imp_df.head(top_k)["feature"].tolist()
    }

# ─── MAIN ─────────────────────────────
def main():
    T0=time.time()
    X,Y,labels,best_params=load_data()
    X_vals=X.values; X_cols=list(X.columns)
    gpu_ids=get_gpu_ids()
    tasks=[(X_vals,X_cols,Y.iloc[:,i].values,labels[i],best_params[labels[i]],gpu_ids[i%len(gpu_ids)],N_FOLDS_EVAL,N_BINS_F1,TOP_K,OUTPUT_ROOT) for i in range(len(labels))]

    results=[]; completed=0
    banner("STEP 3 — RETRAIN BEST MODELS Multi-GPU")
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures={pool.submit(retrain_trait,t):t[3] for t in tasks}
        for fut in as_completed(futures):
            res=fut.result()
            results.append(res)
            completed+=1
            progress_bar(completed,len(labels),prefix="STEP3-4 Progress",suffix=f"✓ {res['trait']} CV_R²={res['cv_r2']:.4f} Train_R²={res['train_r2']:.4f}")

    # Save metrics
    csv_dir=os.path.join(OUTPUT_ROOT,"csv"); os.makedirs(csv_dir,exist_ok=True)
    metrics_rows=[{k:v for k,v in r.items() if k not in ["oof_preds","y_pred_full","top_features"]} for r in results]
    metrics_df=pd.DataFrame(metrics_rows).set_index("trait")
    metrics_df.to_csv(os.path.join(csv_dir,"metrics_best_xgb.csv"))

    # Save OOF & full train predictions
    pd.DataFrame({r["trait"]:r["oof_preds"] for r in results},index=X.index).to_csv(os.path.join(csv_dir,"predictions_oof.csv"))
    pd.DataFrame({r["trait"]:r["y_pred_full"] for r in results},index=X.index).to_csv(os.path.join(csv_dir,"predictions_best_xgb_train.csv"))

    # Top-K features summary
    top_summary=[]
    for r in results:
        for rank,f in enumerate(r["top_features"],1):
            top_summary.append({"trait":r["trait"],"rank":rank,"feature":f})
    pd.DataFrame(top_summary).to_csv(os.path.join(csv_dir,"top_features_summary.csv"),index=False)

    # Heatmap Top15
    viz_dir=os.path.join(OUTPUT_ROOT,"viz"); os.makedirs(viz_dir,exist_ok=True)
    all_feats=set(); imp_data={}
    for r in results:
        idf=pd.read_csv(os.path.join(csv_dir,f"importance_best_{r['trait']}.csv")).head(15)
        all_feats.update(idf["feature"].tolist())
        imp_data[r["trait"]]=dict(zip(idf["feature"],idf["importance_norm"]))
    feat_list=sorted(all_feats)
    heat_df=pd.DataFrame(0.0,index=feat_list,columns=[r["trait"] for r in results])
    for trait,fd in imp_data.items():
        for feat in feat_list: heat_df.loc[feat,trait]=fd.get(feat,0.0)
    fig,ax=plt.subplots(figsize=(14,max(8,len(feat_list)*0.25)))
    im=ax.imshow(heat_df.values,aspect="auto",cmap="YlOrRd")
    ax.set_xticks(range(len(heat_df.columns)));ax.set_xticklabels([c[:20] for c in heat_df.columns],rotation=45,ha="right",fontsize=7)
    ax.set_yticks(range(len(heat_df.index)));ax.set_yticklabels(heat_df.index,fontsize=6)
    ax.set_title("Feature Importance Heatmap Top15 per Trait")
    plt.colorbar(im,ax=ax,shrink=0.6);plt.tight_layout();plt.savefig(os.path.join(viz_dir,"feature_importance_heatmap.png"),dpi=150);plt.close()

    # Scatter OOF vs Actual
    fig,axes=plt.subplots(2,5,figsize=(20,8));axes=axes.flatten()
    for idx,r in enumerate(results):
        ax=axes[idx]
        ax.scatter(Y.iloc[:,idx].values,r["oof_preds"],alpha=0.3,s=10,color="#6366f1")
        lims=[min(Y.iloc[:,idx].min(),r["oof_preds"].min()),max(Y.iloc[:,idx].max(),r["oof_preds"].max())]
        ax.plot(lims,lims,'r--',linewidth=1)
        ax.set_title(f"{r['trait'][:18]}\nR²={r['cv_r2']:.3f}",fontsize=8)
        ax.set_xlabel("Actual",fontsize=7);ax.set_ylabel("Predicted",fontsize=7);ax.tick_params(labelsize=6)
    plt.suptitle("OOF Predictions vs Ground Truth",fontsize=12)
    plt.tight_layout();plt.savefig(os.path.join(viz_dir,"scatter_oof_all_traits.png"),dpi=150);plt.close()

    banner("STEP 3-4 COMPLETE")
    print(f"Total elapsed: {time.time()-T0:.1f}s")

if __name__=="__main__":
    main()