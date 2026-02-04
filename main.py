import numpy as np
import argparse
import yaml
import os
from sentence_transformers import SentenceTransformer, util
from scipy.stats import bootstrap, ttest_rel

def evaluate(candidates, references, model):
    if len(candidates) != len(references):
        raise ValueError(f"文の数が一致しません。Candidates: {len(candidates)}, References: {len(references)}")

    all_sentences = candidates + references
    embeddings = model.encode(all_sentences, convert_to_tensor=True, show_progress_bar=False)

    embeddings_pred = embeddings[:len(candidates)]
    embeddings_ref = embeddings[len(candidates):]

    cosine_scores = util.pairwise_cos_sim(embeddings_pred, embeddings_ref)
    return np.array(cosine_scores.cpu().tolist())

def calculate_statistics(scores_base, scores_target, n_resamples=5000):
    diffs = scores_target - scores_base
    
    # ブートストラップ法による信頼区間の計算
    res = bootstrap((diffs,), np.mean, 
                    n_resamples=n_resamples, 
                    confidence_level=0.95, 
                    method='percentile')
    
    ci_low = res.confidence_interval.low
    ci_high = res.confidence_interval.high
    
    # 信頼区間が0をまたいでいないかチェック
    is_significant_ci = (ci_low > 0) or (ci_high < 0)

    # 対応のあるt検定によるp値の計算
    # H0: 平均差 = 0
    t_stat, p_value = ttest_rel(scores_target, scores_base)
    
    return {
        "mean": np.mean(scores_target),
        "mean_diff": np.mean(diffs),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "is_significant_ci": is_significant_ci,
        "p_value": p_value
    }

def load_and_clean(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    return [s.strip().replace(" ", "").replace("　", "") for s in lines]

def run_evaluation(base_path, target_paths, ref_path, model_name):
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    lines_ref = load_and_clean(ref_path)
    lines_base = load_and_clean(base_path)
    
    print("Calculating Baseline scores...")
    scores_base = evaluate(lines_base, lines_ref, model)
    base_mean = np.mean(scores_base)

    results = []
    for name, path in target_paths.items():
        print(f"Evaluating {name}...")
        lines_target = load_and_clean(path)
        scores_target = evaluate(lines_target, lines_ref, model)
        
        stat = calculate_statistics(scores_base, scores_target)
        stat['name'] = name
        results.append(stat)

    # 結果表示
    print(f"\nBASE: {base_path}")
    print("="*85)
    print(f"{'Method':<20} | {'Mean':<6} | {'Diff':<7} | {'95% CI (Bootstrap)':<20} | {'p-value (t-test)'}")
    print("-" * 85)
    print(f"{'BASE':<20} | {base_mean:.4f} | {'-':<7} | {'-':<20} | -")
    
    for r in results:
        # CIによる有意判定マーク（視覚補助）
        sig_mark = "*" if r['is_significant_ci'] else ""
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]{sig_mark}"
        
        # p値のフォーマット
        p_val_str = f"{r['p_value']:.4e}" if r['p_value'] < 0.0001 else f"{r['p_value']:.4f}"
        
        print(f"{r['name']:<20} | {r['mean']:.4f} | {r['mean_diff']:+.4f} | {ci_str:<20} | {p_val_str}")
    print("="*85)
    print("Note: '*' in CI indicates the interval excludes zero (Significant at alpha=0.05).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentencBERTのcos類似度を有意差検定付きで比較")
    parser.add_argument("--preset", type=str, help="設定ファイル内のプリセット名")
    parser.add_argument("--base", type=str, help="基準(Baseline)のファイルパス")
    parser.add_argument("--targets", nargs='+', help="比較対象ファイルパス（例: name1:path1 name2:path2）")
    parser.add_argument("--ref", type=str, help="参照訳(Ground Truth)のファイルパス")
    
    args = parser.parse_args()

    # パラメータの決定（Config or CLI）
    base_file = None
    target_files = {}
    ref_file = None
    model_name = 'sonoisa/sentence-bert-base-ja-mean-tokens-v2'

    if args.preset:
        with open(f"{os.path.dirname(__file__)}/presets.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            preset = config['presets'].get(args.preset)
            if not preset:
                print(f"Error: Preset '{args.preset}' not found in {os.path.dirname(__file__)}/presets.yaml")
                exit(1)
            base_file = preset['base']
            target_files = preset['targets']
            ref_file = preset['ref']
            model_name = preset.get('model', model_name)

    # CLI引数が指定されている場合は上書き
    if args.base: base_file = args.base
    if args.ref: ref_file = args.ref
    if args.targets:
        # "name:path" 形式をパース
        target_files = {t.split(':')[0]: t.split(':')[1] for t in args.targets}

    if not (base_file and target_files and ref_file):
        print("Error: 必要な入力ファイル（base, targets, ref）が指定されていません。")
        parser.print_help()
        exit(1)

    run_evaluation(base_file, target_files, ref_file, model_name)

