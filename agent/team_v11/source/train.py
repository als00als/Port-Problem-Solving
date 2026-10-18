"""LightGBM 랭커 학습 · vessel 단위 CV · 제출 폴더 생성.

  CV       : .venv_competition_py311\\Scripts\\python.exe agent\\team_v11\\source\\train.py --cv --no-dict
  로컬 모델: .venv_competition_py311\\Scripts\\python.exe agent\\team_v11\\source\\train.py --out team_v11_artifacts\\v11\\src --no-dict --callsign --name-lookup

학습 쿼리의 사전(토큰 사전, 국적·선종 확률)은 같은 선박이 빠진 inner fold로 만든다(OOF).
그래야 학습 피처가 '처음 보는 선박'에서의 피처 분포와 같아진다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import shutil
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import match_lib as ml  # noqa: E402

SEED = 42
RANKER_PARAMS = dict(
    objective="lambdarank",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=8,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


def load():
    train = pd.read_csv(ROOT / "data" / "open" / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "data" / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
    return train, catalog


NO_DICT = False  # --no-dict: 한글 토큰 사전·전체 이름 사전을 비워 사전 의존도 확인
OOF_TAG = "latest"


def resources(train: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    res = ml.build_resources(train, catalog)
    if NO_DICT:
        res["token_dict"], res["whole_dict"] = {}, {}
    return res


def training_matrix(train: pd.DataFrame, catalog: pd.DataFrame, index: ml.CatalogIndex, inner: int = 5):
    xs, ys, groups = [], [], []
    hits = 0
    splitter = GroupKFold(n_splits=inner)
    for fit_idx, prep_idx in splitter.split(train, groups=train["target_vessel_id"]):
        res = resources(train.iloc[fit_idx], catalog)
        part = train.iloc[prep_idx]
        rows = part.to_dict("records")
        queries = [ml.prepare_query(r, res) for r in rows]
        truth = [index.id_to_index[r["target_vessel_id"]] for r in rows]
        # 정답이 후보에 없으면 학습용으로만 추가
        hit_flags: list[bool] = []
        cands, feats = ml.candidates_and_features(queries, index, res, force=truth, hits=hit_flags)
        hits += sum(hit_flags)
        for cand, t in zip(cands, truth):
            ys.append((cand == t).astype(np.int8))
            groups.append(len(cand))
        xs.append(feats)
    return np.vstack(xs), np.concatenate(ys), np.asarray(groups), hits / len(train)


# 단조 제약: 유사도·국적·선종 확률은 높을수록(+), 톤수 차이·번호 충돌은 클수록(-) 점수가 낮아지도록 강제
MONOTONE_UP = {
    "s_compact", "s_skel_rom", "s_skel_trans", "s_token_set", "s_token_sort", "s_partial", "s_jw", "s_rom_compact",
    "exact", "first_eq", "num_equal", "gt_exact", "flag_prob", "type_prob",
    "s_var_ratio", "s_var_jw", "s_var_partial", "s_learn_ratio", "s_learn_top1", "s_learn_jw",
    "s_vocab_ratio", "s_vocab_jw", "s_alpha_ratio",
}
MONOTONE_DOWN = {"num_conflict", "gap_gross", "gap_intl", "gap_best", "num_q_only", "num_c_only"}
USE_MONOTONE = True


def ranker_params(seed: int = SEED) -> dict:
    params = dict(RANKER_PARAMS, random_state=seed)
    if USE_MONOTONE:
        params["monotone_constraints"] = [
            1 if f in MONOTONE_UP else -1 if f in MONOTONE_DOWN else 0 for f in ml.FEATURE_NAMES
        ]
    return params


def fit(train, catalog, index):
    x, y, groups, recall = training_matrix(train, catalog, index)
    model = lgb.LGBMRanker(**ranker_params())
    model.fit(x, y, group=groups, feature_name=list(ml.FEATURE_NAMES))
    return model, recall


# ---------------------------------------------------------------- 앙상블
CAT_TASK_TYPE = "GPU"
MEMBERS = {"lgb42": ("lgb", 42), "lgb7": ("lgb", 7), "lgb2024": ("lgb", 2024), "cat": ("cat", 42)}
COMBOS = {
    "lgb42": ["lgb42"],
    "lgb3": ["lgb42", "lgb7", "lgb2024"],
    "cat": ["cat"],
    "lgb42+cat": ["lgb42", "cat"],
    "lgb3+cat": ["lgb42", "lgb7", "lgb2024", "cat"],
}


def train_member(kind: str, seed: int, x, y, groups) -> tuple[str, object]:
    if kind == "lgb":
        model = lgb.LGBMRanker(**ranker_params(seed)).fit(x, y, group=groups, feature_name=list(ml.FEATURE_NAMES))
        return "lgb", model.booster_.model_to_string()
    from catboost import CatBoostRanker

    group_id = np.repeat(np.arange(len(groups)), groups)
    # 학습만 로컬 GPU로 (추론은 평가 서버 CPU에서 그대로 동작)
    model = CatBoostRanker(
        loss_function="YetiRank", iterations=600, learning_rate=0.08, depth=6,
        random_seed=seed, thread_count=8, verbose=0, allow_writing_files=False,
        task_type=CAT_TASK_TYPE,
    )
    model.fit(x, y, group_id=group_id)
    return "cat", model


def fit_members(train, catalog, index, names: list[str]) -> tuple[dict, float]:
    x, y, groups, recall = training_matrix(train, catalog, index)
    members = {}
    for name in names:
        t0 = time.time()
        members[name] = train_member(*MEMBERS[name], x, y, groups)
        print(f"    {name} trained ({time.time()-t0:.0f}s)", flush=True)
    return members, recall


def run_cv_ensemble(train, catalog, index) -> None:
    """같은 fold·같은 피처로 단일 모델과 앙상블 조합을 함께 비교."""
    hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {name: np.empty(len(train), dtype=object) for name in COMBOS}
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, hangul, train["target_vessel_id"])):
        t0 = time.time()
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        members, _ = fit_members(tr, catalog, index, list(MEMBERS))
        res = resources(tr, catalog)
        queries = [ml.prepare_query(r, res) for r in va.to_dict("records")]
        cands, feats = ml.candidates_and_features(queries, index, res)
        sizes = [len(c) for c in cands]
        z = {name: ml.RankEnsemble([m]).predict_grouped(feats, sizes) for name, m in members.items()}
        line = []
        for combo, names in COMBOS.items():
            total = sum(z[n] for n in names)
            preds, offset = [], 0
            for cand in cands:
                preds.append(index.ids[cand[int(np.argmax(total[offset : offset + len(cand)]))]])
                offset += len(cand)
            oof[combo][va_idx] = preds
            line.append(f"{combo}={score(va, preds)['weighted']:.4f}")
        print(f"fold {fold}: " + " ".join(line) + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n===== OOF (vessel GroupKFold, 5 folds) =====")
    for combo in COMBOS:
        s = score(train, list(oof[combo]))
        print(f"{combo:10s} weighted={s['weighted']:.4f} hangul={s['hangul']:.4f} non_hangul={s['non_hangul']:.4f}")
    out = ROOT / "team_v11_artifacts" / "cv"
    for combo in COMBOS:
        correct = oof[combo] == train["target_vessel_id"].to_numpy()
        train.assign(pred_vessel_id=oof[combo], correct=correct, is_hangul=hangul).to_csv(
            out / f"{OOF_TAG}_{combo.replace('+', '_')}_oof.csv", index=False, encoding="utf-8-sig")


def score(truth: pd.DataFrame, pred: list[str]) -> dict:
    hangul = truth["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    correct = truth["target_vessel_id"].to_numpy() == np.asarray(pred)
    h, nh = correct[hangul].mean(), correct[~hangul].mean()
    return {"weighted": 0.7 * h + 0.3 * nh, "hangul": h, "non_hangul": nh}


def run_cv(train, catalog, index) -> None:
    hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.empty(len(train), dtype=object)
    recall_hits = np.zeros(len(train), dtype=bool)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, hangul, train["target_vessel_id"])):
        t0 = time.time()
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        model, train_recall = fit(tr, catalog, index)
        res = resources(tr, catalog)
        cands: list = []
        pred = ml.predict(model, va, index, res, candidates_out=cands)
        oof[va_idx] = pred
        for j, (cand, tid) in enumerate(zip(cands, va["target_vessel_id"])):
            recall_hits[va_idx[j]] = index.id_to_index[tid] in set(cand.tolist())
        s = score(va, pred)
        print(f"fold {fold}: weighted={s['weighted']:.4f} hangul={s['hangul']:.4f} "
              f"non_hangul={s['non_hangul']:.4f} (train recall {train_recall:.3f}, {time.time()-t0:.0f}s)", flush=True)
    s = score(train, list(oof))
    print("\n===== OOF (vessel GroupKFold, 5 folds) =====")
    print(f"weighted   : {s['weighted']:.4f}")
    print(f"hangul     : {s['hangul']:.4f}")
    print(f"non_hangul : {s['non_hangul']:.4f}")
    print(f"candidate recall  hangul={recall_hits[hangul].mean():.4f}  non_hangul={recall_hits[~hangul].mean():.4f}")
    out = ROOT / "team_v11_artifacts" / "cv"
    out.mkdir(parents=True, exist_ok=True)
    correct = oof == train["target_vessel_id"].to_numpy()
    train.assign(pred_vessel_id=oof, correct=correct, candidate_hit=recall_hits, is_hangul=hangul).to_csv(
        out / f"{OOF_TAG}_oof.csv", index=False, encoding="utf-8-sig")


def run_rowcheck(train, catalog, index) -> None:
    """행 단위 랜덤 분할 1 fold: test에 train과 같은 선박이 섞인 상황에서 조회 규칙 효과 확인."""
    from sklearn.model_selection import KFold

    tr_idx, va_idx = next(KFold(n_splits=5, shuffle=True, random_state=SEED).split(train))
    tr, va = train.iloc[tr_idx], train.iloc[va_idx]
    model, _ = fit(tr, catalog, index)
    res = resources(tr, catalog)
    seen = va["target_vessel_id"].isin(set(tr["target_vessel_id"])).to_numpy()
    print(f"valid {len(va)}행 중 train에 있는 선박 {seen.mean():.3f}")
    for name, kw in {
        "model only": {},
        "+ name lookup": {"use_name_lookup": True},
        "+ callsign": {"use_callsign": True},
        "+ both": {"use_name_lookup": True, "use_callsign": True},
    }.items():
        pred = np.asarray(ml.predict(model, va, index, res, **kw))
        s = score(va, list(pred))
        correct = pred == va["target_vessel_id"].to_numpy()
        print(f"{name:15s} weighted={s['weighted']:.4f} hangul={s['hangul']:.4f} non_hangul={s['non_hangul']:.4f}"
              f" | seen acc={correct[seen].mean():.4f} unseen acc={correct[~seen].mean():.4f}")


def build_submission(train, catalog, index, out_dir: Path, use_callsign: bool, use_name_lookup: bool,
                     ensemble: str | None = None) -> None:
    if ensemble:
        members, recall = fit_members(train, catalog, index, COMBOS[ensemble])
        model = None
    else:
        model, recall = fit(train, catalog, index)
    res = resources(train, catalog)
    res["use_callsign"] = use_callsign
    res["use_name_lookup"] = use_name_lookup
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "model").mkdir(parents=True)
    if ensemble:
        with open(out_dir / "model" / "ensemble.pkl", "wb") as f:
            pickle.dump(ml.RankEnsemble([members[n] for n in COMBOS[ensemble]]), f)
    else:
        # LightGBM C++ 저장은 한글 경로를 못 열어서 문자열로 받아 Python으로 쓴다
        (out_dir / "model" / "lgb_ranker.txt").write_text(model.booster_.model_to_string(), encoding="utf-8")
    with open(out_dir / "model" / "resources.pkl", "wb") as f:
        pickle.dump(res, f)
    shutil.copy(SOURCE / "match_lib.py", out_dir / "model" / "match_lib.py")
    shutil.copy(SOURCE / "script.py", out_dir / "script.py")
    (out_dir / "requirements.txt").write_text(
        "# 평가 서버 기본 설치 패키지(pandas, numpy, lightgbm, rapidfuzz, Unidecode)만 사용\n", encoding="utf-8")
    print(f"saved {out_dir} (train candidate recall {recall:.3f}, "
          f"use_callsign={use_callsign}, use_name_lookup={use_name_lookup})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--rowcheck", action="store_true", help="행 단위 랜덤 분할로 조회 규칙 효과 확인")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--callsign", action="store_true", help="train call_sign 조회로 예측 덮어쓰기")
    parser.add_argument("--name-lookup", action="store_true", help="train 선박명 조회로 예측 덮어쓰기")
    parser.add_argument("--no-dict", action="store_true", help="한글 토큰 사전 없이 실행")
    parser.add_argument("--tag", default="latest", help="CV 결과 파일 이름 (outputs/cv/<tag>_oof.csv)")
    parser.add_argument("--ensemble-cv", action="store_true", help="단일·앙상블 조합 CV 비교")
    parser.add_argument("--ensemble", choices=list(COMBOS), help="제출 폴더를 이 앙상블 조합으로 생성")
    args = parser.parse_args()
    global NO_DICT, OOF_TAG
    NO_DICT, OOF_TAG = args.no_dict, args.tag
    train, catalog = load()
    index = ml.CatalogIndex(catalog)
    if args.cv:
        run_cv(train, catalog, index)
    if args.rowcheck:
        run_rowcheck(train, catalog, index)
    if args.ensemble_cv:
        run_cv_ensemble(train, catalog, index)
    if args.out:
        build_submission(train, catalog, index, args.out.resolve(), args.callsign, args.name_lookup, args.ensemble)


if __name__ == "__main__":
    main()
