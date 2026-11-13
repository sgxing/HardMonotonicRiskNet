import os
# ---- ★ 可复现性 (Mac M5 / Apple Silicon + Metal 后端) ----
#os.environ['PYTHONHASHSEED'] = '0'                       # Python dict/set 遍历顺序确定
#os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'   # 关闭 XLA 自动调优 (Metal 也适用)
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from datetime import datetime
import time
import pandas as pd
import glob
from typing import Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ_PATH = os.path.join(CURRENT_DIR, "lending_club_train_val_test.npz")
DEFAULT_RAW_CSV_PATH = "/Users/mingxi/.cache/kagglehub/datasets/wordsforthewise/lending-club/versions/3/accepted_2007_to_2018q4.csv/accepted_2007_to_2018q4.csv"

def print_mob_crosstab(name, y_mob6, y_mob2):
    """
    打印 mob6 与 mob2 标签的交叉分布 (2×2 列联表) 及业务转化率
    • 行: mob2 状态 (0: 2月内未逾期, 1: 2月内已逾期)
    • 列: mob6 状态 (0: 6月内未逾期, 1: 6月内已逾期)
    """
    n = len(y_mob6)
    y6 = np.asarray(y_mob6, dtype=np.int64)
    y2 = np.asarray(y_mob2, dtype=np.int64)

    m00 = int(((y6 == 0) & (y2 == 0)).sum())   # mob6=0, mob2=0 (始终未逾期)
    m01 = int(((y6 == 0) & (y2 == 1)).sum())   # mob6=0, mob2=1 (反转异常，业务定义上应为 0)
    m10 = int(((y6 == 1) & (y2 == 0)).sum())   # mob6=1, mob2=0 (增量逾期: 2月正常但6月逾期)
    m11 = int(((y6 == 1) & (y2 == 1)).sum())   # mob6=1, mob2=1 (早期逾期并持续至6月)

    print(f"\n{'='*22} {name} (N={n:,}) — MOB2 × MOB6 交叉分布 {'='*22}")
    print(f"{'mob2 \\ mob6':<16}{'mob6=0':>12}{'mob6=1':>12}{'合计':>12}")
    print("-" * 54)
    print(f"{'mob2=0 (早期正常)':<16}{m00:>12,}{m10:>12,}{m00+m10:>12,}")
    print(f"{'mob2=1 (早期逾期)':<16}{m01:>12,}{m11:>12,}{m01+m11:>12,}")
    print(f"{'合计':<16}{m00+m01:>12,}{m10+m11:>12,}{n:>12,}")

    mob6_pos = m10 + m11
    mob2_pos = m01 + m11
    p_m11_m6 = (m11 / mob6_pos * 100) if mob6_pos > 0 else 0.0
    p_m11_m2 = (m11 / mob2_pos * 100) if mob2_pos > 0 else 0.0
    p_delta  = (m10 / (m00 + m10) * 100) if (m00 + m10) > 0 else 0.0

    print(f"\n  • 增量逾期转化率 P(mob6=1 | mob2=0) : {p_delta:6.2f}%  ({m10:,}/{m00+m10:,})  [ESMM 增量塔拟合核心]")
    print(f"  • mob2 恶化维持率 P(mob6=1 | mob2=1) : {p_m11_m2:6.2f}%  ({m11:,}/{mob2_pos:,})  [单调性检验: mob2=1 导致 mob6=1]")
    print(f"  • mob6 归因来自早期逾期占比           : {p_m11_m6:6.2f}%  ({m11:,}/{mob6_pos:,})")
    if m01 > 0:
        print(f"  • ⚠️ 异常反转样本 (mob2=1 但 mob6=0) : {m01:,}  (请核查数据标签定义)")
    else:
        print(f"  • 严格单调性检验通过 (mob2=1 但 mob6=0 为 0)")

    ct_df = pd.DataFrame(
        [[m00, m10, m00 + m10], [m01, m11, m01 + m11], [m00 + m01, m10 + m11, n]],
        index=['mob2=0', 'mob2=1', '合计'],
        columns=['mob6=0', 'mob6=1', '合计']
    )
    return ct_df

def build_lending_club_high_dim_features(csv_path):
    # 1. 严格剔除贷后前瞻泄露变量（催收、核销金额、收回本息、贷后纾困与债务重组等）
    leakage_cols = [
        'collection_recovery_fee', 'recoveries', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'total_pymnt', 'total_pymnt_inv', 'last_pymnt_amnt',
        'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'debt_settlement_flag',
        'debt_settlement_flag_date', 'settlement_status', 'settlement_date',
        'settlement_amount', 'settlement_percentage', 'settlement_term',
        'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
        'deferral_term', 'hardship_amount', 'hardship_start_date', 'hardship_end_date',
        'payment_plan_start_date', 'hardship_length', 'hardship_dpd', 'hardship_loan_status',
        'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount',
        'hardship_last_payment_amount', 'disbursement_method', 'pymnt_plan',
        'last_credit_pull_d', 'last_fico_range_low', 'last_fico_range_high', 'verification_status_joint', 'sec_app_earliest_cr_line',
        'int_rate','sub_grade','grade'
    ]
    
    # 2. 读取数据并保留已结案订单
    df = pd.read_csv(csv_path, low_memory=False)
    df['issue_dt'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df = df[(df['issue_dt'] <= '2018-06-01') & (df['issue_dt'] >= '2016-06-01')].copy().reset_index(drop=True)
    # 3. 构造 MOB2 和 MOB6 跨期标签与时序排序字段 issue_dt
    last_pymnt_dt = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y')
    pay_mths = ((last_pymnt_dt.dt.year - df['issue_dt'].dt.year) * 12 + 
                (last_pymnt_dt.dt.month - df['issue_dt'].dt.month)).fillna(0)
    
    bad_statuses = [
        'Charged Off',
        'Default',
        'Late (31-120 days)',
        'Does not meet the credit policy. Status:Charged Off'
    ]
    is_bad = df['loan_status'].isin(bad_statuses)
    df['y_mob2'] = ((is_bad) & (pay_mths <= 1)).astype(int)
    df['y_mob6'] = ((is_bad) & (pay_mths <= 5)).astype(int)
    
    # 按照 issue_dt 时间先后顺序严格升序排序
    df = df.sort_values('issue_dt').reset_index(drop=True)
    
    # 剔除日期与泄露列
    df.drop(columns=leakage_cols + ['loan_status', 'last_pymnt_d'], inplace=True, errors='ignore')
    
    # 4. 衍生核心金融压力与比率交互特征（约 20+ 维）
    earliest_dt = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
    df['cr_history_mths'] = ((df['issue_dt'].dt.year - earliest_dt.dt.year) * 12 + 
                             (df['issue_dt'].dt.month - earliest_dt.dt.month)).fillna(0)
    
    df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2.0
    df['pti'] = df['installment'] / (df['annual_inc'] / 12.0 + 1.0)
    df['lti'] = df['loan_amnt'] / (df['annual_inc'] + 1.0)
    df['avail_revol_credit'] = np.maximum(0, df['total_rev_hi_lim'].fillna(0) - df['revol_bal'].fillna(0))
    df['delinq_density'] = df['delinq_2yrs'] / (df['open_acc'] + 1.0)
    df['inq_density'] = df['inq_last_6mths'] / (df['total_acc'] + 1.0)
    df['revol_util_ratio'] = df['revol_bal'] / (df['total_rev_hi_lim'] + 1.0)
    df['tot_cur_bal_to_inc'] = df['tot_cur_bal'] / (df['annual_inc'] + 1.0)
    
    # 7. 移除非特征字段与高基数文本 (emp_title 等)
    drop_meta = ['id', 'member_id', 'url', 'desc', 'title', 'zip_code', 'issue_d', 'earliest_cr_line', 'emp_title','policy_code']
    df.drop(columns=drop_meta, inplace=True, errors='ignore')
    
    feature_count = len([c for c in df.columns if c not in ['y_mob2', 'y_mob6', 'issue_dt']])
    feature_list = [c for c in df.columns if c not in ['y_mob2', 'y_mob6', 'issue_dt']]
    df[feature_list] = df[feature_list].fillna(-1)
    print(f"数据清洗与高维特征衍生完成！当前总特征维度: {feature_count} 维")
    return df

def compute_woe_maps(df, y, cols, smoothing=0.5, min_count=20):
    maps = {}
    iv_dict = {}
    N_total = len(df)
    total_bad = int(y.sum())
    total_good = int(N_total - total_bad)
    
    for col in cols:
        ser = df[col].fillna('__MISSING__').astype(str)
        cat_counts = ser.value_counts()
        low_cats = set(cat_counts[cat_counts < min_count].index)
        if low_cats:
            ser = ser.where(~ser.isin(low_cats), '<UNK>')
        
        df_temp = pd.DataFrame({'cat': ser, 'y': y})
        stats = df_temp.groupby('cat')['y'].agg(['sum', 'count'])
        bad = stats['sum']
        good = stats['count'] - bad
        n_cats = len(stats)
        
        bad_smooth = bad + smoothing
        good_smooth = good + smoothing
        global_bad_smooth = total_bad + smoothing * n_cats
        global_good_smooth = total_good + smoothing * n_cats
        
        p_bad = bad_smooth / global_bad_smooth
        p_good = good_smooth / global_good_smooth
        
        woe = np.log(p_bad / p_good)
        iv = np.sum((p_bad - p_good) * woe)
        
        wmap = woe.to_dict()
        if '<UNK>' not in wmap:
            wmap['<UNK>'] = 0.0
        maps[col] = wmap
        iv_dict[col] = float(iv)
    return maps, iv_dict

def get_woe_matrix(df, woe_maps, cols):
    out = pd.DataFrame(index=df.index)
    for col in cols:
        wmap = woe_maps[col]
        unk_woe = wmap.get('<UNK>', 0.0)
        ser = df[col].fillna('__MISSING__').astype(str)
        out[f'{col}_woe'] = ser.map(wmap).fillna(unk_woe).astype(np.float32)
    return out

def compute_continuous_iv_dict(df, y, cont_cols, n_bins=10, smoothing=0.5):
    """
    在训练集上严格计算连续数值特征的 IV 值 (10等频分箱，缺失值作为独立分箱 '__MISSING__')
    """
    cont_iv_dict = {}
    total_bad = int(y.sum())
    total_good = int(len(df) - total_bad)
    
    for col in cont_cols:
        ser = df[col]
        is_missing = ser.isna() | (ser == -1)
        valid_vals = ser[~is_missing]
        
        if len(valid_vals) == 0:
            cont_iv_dict[col] = 0.0
            continue
            
        try:
            bins = pd.qcut(valid_vals, q=n_bins, duplicates='drop')
            cat_ser = pd.Series(index=ser.index, dtype='object')
            cat_ser[~is_missing] = bins.astype(str)
        except Exception:
            bins = pd.cut(valid_vals, bins=n_bins, duplicates='drop')
            cat_ser = pd.Series(index=ser.index, dtype='object')
            cat_ser[~is_missing] = bins.astype(str)
            
        if is_missing.any():
            cat_ser[is_missing] = '__MISSING__'
            
        df_temp = pd.DataFrame({'bin': cat_ser, 'y': y})
        stats = df_temp.groupby('bin', observed=True)['y'].agg(['sum', 'count'])
        bad = stats['sum']
        good = stats['count'] - bad
        n_cats = len(stats)
        
        p_bad = (bad + smoothing) / (total_bad + smoothing * n_cats)
        p_good = (good + smoothing) / (total_good + smoothing * n_cats)
        
        woe = np.log(p_bad / p_good)
        iv = float(np.sum((p_bad - p_good) * woe))
        cont_iv_dict[col] = iv
        
    return cont_iv_dict

def get_iv_level(iv):
    if iv >= 0.5:
        return "极高区分度 (IV >= 0.5)"
    elif iv >= 0.3:
        return "强预测能力 (0.3 <= IV < 0.5)"
    elif iv >= 0.1:
        return "中等预测能力 (0.1 <= IV < 0.3)"
    elif iv >= 0.02:
        return "较弱预测能力 (0.02 <= IV < 0.1)"
    else:
        return "微弱/无显著贡献 (IV < 0.02)"

def process_lending_club_dataset(
    csv_path: str = DEFAULT_RAW_CSV_PATH,
    npz_save_path: Optional[str] = DEFAULT_NPZ_PATH
) -> Dict[str, Union[np.ndarray, list]]:
    """
    执行 Lending Club 全量特征工程流水线并返回训练集、验证集、测试集。
    若指定 npz_save_path，则自动将处理结果持久化为高速压缩 NPZ 文件。
    """
    print(f"📖 正在从 CSV 读取数据: {csv_path}")
    df_featured = build_lending_club_high_dim_features(csv_path)

    # 按照 issue_dt 时间先后顺序，按照 0.70 : 0.15 : 0.15 切分为训练集、验证集、测试集
    N = len(df_featured)
    n_train = int(N * 0.70)
    n_valid = int(N * 0.15)

    df_train = df_featured.iloc[:n_train].copy().reset_index(drop=True)
    df_valid = df_featured.iloc[n_train:n_train + n_valid].copy().reset_index(drop=True)
    df_test = df_featured.iloc[n_train + n_valid:].copy().reset_index(drop=True)

    print("\n" + "="*25 + " 时序切分完成 (0.70 : 0.15 : 0.15) " + "="*25)
    print(f"总样本数: {N:,} 行")
    print(f"训练集 (Train) : {len(df_train):,} 行 ({len(df_train)/N*100:.1f}%) | 时间跨度: {df_train['issue_dt'].min().strftime('%Y-%m')} 至 {df_train['issue_dt'].max().strftime('%Y-%m')} | MOB2违约率: {df_train['y_mob2'].mean():.4f} | MOB6违约率: {df_train['y_mob6'].mean():.4f}")
    print(f"验证集 (Valid) : {len(df_valid):,} 行 ({len(df_valid)/N*100:.1f}%) | 时间跨度: {df_valid['issue_dt'].min().strftime('%Y-%m')} 至 {df_valid['issue_dt'].max().strftime('%Y-%m')} | MOB2违约率: {df_valid['y_mob2'].mean():.4f} | MOB6违约率: {df_valid['y_mob6'].mean():.4f}")
    print(f"测试集 (Test)  : {len(df_test):,} 行 ({len(df_test)/N*100:.1f}%) | 时间跨度: {df_test['issue_dt'].min().strftime('%Y-%m')} 至 {df_test['issue_dt'].max().strftime('%Y-%m')} | MOB2违约率: {df_test['y_mob2'].mean():.4f} | MOB6违约率: {df_test['y_mob6'].mean():.4f}")
    print("=" * 75)

    # 调用交叉列联表打印三集单调性与违约分布
    print_mob_crosstab("训练集 (Train)", df_train['y_mob6'], df_train['y_mob2'])
    print_mob_crosstab("验证集 (Valid)", df_valid['y_mob6'], df_valid['y_mob2'])
    print_mob_crosstab("测试集 (Test/OOT)", df_test['y_mob6'], df_test['y_mob2'])

    # ── 6.1 安全过滤：剔除高基数非结构化文本 (emp_title) 及贷后前瞻列 ──
    unwanted_cols = [
        'policy_code', 'emp_title', 'zip_code', 'title', 'desc', 'issue_d', 'earliest_cr_line',
        'debt_settlement_flag_date', 'settlement_status', 'settlement_date',
        'settlement_amount', 'settlement_percentage', 'settlement_term',
        'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
        'deferral_term', 'hardship_amount', 'hardship_start_date', 'hardship_end_date',
        'payment_plan_start_date', 'hardship_length', 'hardship_dpd', 'hardship_loan_status',
        'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount',
        'hardship_last_payment_amount', 'disbursement_method', 'pymnt_plan', 'last_credit_pull_d', 'last_fico_range_low', 'last_fico_range_high',
        'term','application_type','initial_list_status'
    ]
    drop_existing = [c for c in unwanted_cols if c in df_train.columns]
    if drop_existing:
        print(f"已过滤高基数文本与贷后前瞻列 ({len(drop_existing)} 个): {drop_existing}")
        df_train.drop(columns=drop_existing, inplace=True, errors='ignore')
        df_valid.drop(columns=drop_existing, inplace=True, errors='ignore')
        df_test.drop(columns=drop_existing, inplace=True, errors='ignore')

    # ── 6.2 零方差常数列自动过滤 ──
    candidate_features = [c for c in df_train.columns if c not in ['issue_dt', 'y_mob2', 'y_mob6']]
    zero_var_cols = [c for c in candidate_features if df_train[c].nunique(dropna=True) <= 1]
    if zero_var_cols:
        print(f"已自动过滤零方差常数列 ({len(zero_var_cols)} 个): {zero_var_cols}")
        df_train.drop(columns=zero_var_cols, inplace=True, errors='ignore')
        df_valid.drop(columns=zero_var_cols, inplace=True, errors='ignore')
        df_test.drop(columns=zero_var_cols, inplace=True, errors='ignore')

    # ── 6.4 特征三分类精准分流 ──
    current_features = [c for c in df_train.columns if c not in ['issue_dt', 'y_mob2', 'y_mob6']]
    cat_cols = ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'addr_state']
    cat_cols = [c for c in cat_cols if c in current_features]
    continuous_cols = [c for c in current_features if c not in cat_cols]

    print(f"\n特征精准分流:")
    print(f"  • 多分类变量 (计算缺失独立分箱 WOE)   : {len(cat_cols)} 维 -> {cat_cols}")
    print(f"  • 连续数值特征 (进行 Z-score 标准化)    : {len(continuous_cols)} 维")

    # ── 6.5 第一路：多分类变量 WOE 计算 ──
    print("\n正在计算多分类变量 WOE (缺失值作为独立分箱)...")
    woe_maps, iv_dict = compute_woe_maps(df_train, df_train['y_mob6'], cat_cols, smoothing=0.5, min_count=20)
    X_train_woe = get_woe_matrix(df_train, woe_maps, cat_cols)
    X_valid_woe = get_woe_matrix(df_valid, woe_maps, cat_cols)
    X_test_woe  = get_woe_matrix(df_test,  woe_maps, cat_cols)

    # ── 6.7 第二路：连续数值特征严格标准化 (StandardScaler，基于训练集拟合) ──
    df_train_cont_imp = df_train[continuous_cols]
    df_valid_cont_imp = df_valid[continuous_cols]
    df_test_cont_imp  = df_test[continuous_cols]

    scaler = StandardScaler()
    scaler.fit(df_train_cont_imp)

    X_train_cont_scaled = pd.DataFrame(scaler.transform(df_train_cont_imp), columns=continuous_cols, index=df_train.index).astype(np.float32)
    X_valid_cont_scaled = pd.DataFrame(scaler.transform(df_valid_cont_imp), columns=continuous_cols, index=df_valid.index).astype(np.float32)
    X_test_cont_scaled  = pd.DataFrame(scaler.transform(df_test_cont_imp),  columns=continuous_cols, index=df_test.index).astype(np.float32)

    # ── 6.8 最终特征横向拼接 (Continuous Scaled + WOE) ──
    X_train = pd.concat([X_train_cont_scaled, X_train_woe], axis=1)
    X_valid = pd.concat([X_valid_cont_scaled, X_valid_woe], axis=1)
    X_test  = pd.concat([X_test_cont_scaled,  X_test_woe],  axis=1)

    final_feature_names = X_train.columns.tolist()

    y_train_mob2 = df_train['y_mob2'].values.astype(np.int64)
    y_train_mob6 = df_train['y_mob6'].values.astype(np.int64)
    y_valid_mob2 = df_valid['y_mob2'].values.astype(np.int64)
    y_valid_mob6 = df_valid['y_mob6'].values.astype(np.int64)
    y_test_mob2  = df_test['y_mob2'].values.astype(np.int64)
    y_test_mob6  = df_test['y_mob6'].values.astype(np.int64)

    # ── 6.9 数据集质量与规格验证报告 ──
    print("\n" + "=" * 75)
    print("🚀 最终建模数据集构建完成！特征结构校验通过:")
    print("=" * 75)
    print(f"X_train 维度: {X_train.shape} | 缺失值数: {X_train.isna().sum().sum()}")
    print(f"X_valid 维度: {X_valid.shape} | 缺失值数: {X_valid.isna().sum().sum()}")
    print(f"X_test  维度: {X_test.shape}  | 缺失值数: {X_test.isna().sum().sum()}")
    print("-" * 75)
    print(f"y_train: MOB2 逾期数={y_train_mob2.sum():,} ({y_train_mob2.mean():.4f}), MOB6 逾期数={y_train_mob6.sum():,} ({y_train_mob6.mean():.4f})")
    print(f"y_valid: MOB2 逾期数={y_valid_mob2.sum():,} ({y_valid_mob2.mean():.4f}), MOB6 逾期数={y_valid_mob6.sum():,} ({y_valid_mob6.mean():.4f})")
    print(f"y_test : MOB2 逾期数={y_test_mob2.sum():,} ({y_test_mob2.mean():.4f}), MOB6 逾期数={y_test_mob6.sum():,} ({y_test_mob6.mean():.4f})")
    print("=" * 75)

    # ── 6.10 计算并打印信息价值 (IV) ──
    print("\n正在计算连续变量信息价值 (IV)...")
    cont_iv_dict = compute_continuous_iv_dict(df_train, df_train['y_mob6'], continuous_cols, n_bins=10)

    # 打印连续变量 Top 20 IV 榜
    print("\n" + "=" * 75)
    print("📈 连续数值特征信息价值 (Top 20, 基准: MOB6):")
    sorted_cont_iv = sorted(cont_iv_dict.items(), key=lambda x: x[1], reverse=True)
    for col, iv in sorted_cont_iv[:20]:
        print(f"  {col:<26}{iv:<12.4f}{get_iv_level(iv)}")
    print("=" * 75)

    # ── 转为标准 float32 NumPy ndarray ──
    X_train_arr = np.asarray(X_train.values, dtype=np.float32)
    X_valid_arr = np.asarray(X_valid.values, dtype=np.float32)
    X_test_arr  = np.asarray(X_test.values,  dtype=np.float32)

    # ── 6.11 持久化导出为高速 NPZ 格式 ──
    if npz_save_path:
        np.savez_compressed(
            npz_save_path,
            X_train=X_train_arr, y_train_mob2=y_train_mob2, y_train_mob6=y_train_mob6,
            X_valid=X_valid_arr, y_valid_mob2=y_valid_mob2, y_valid_mob6=y_valid_mob6,
            X_test=X_test_arr,   y_test_mob2=y_test_mob2,   y_test_mob6=y_test_mob6,
            feature_names=np.array(final_feature_names)
        )
        print(f"✅ 数据集已成功持久化保存至: {npz_save_path}")

    return {
        "X_train": X_train_arr,
        "y_train_mob2": y_train_mob2,
        "y_train_mob6": y_train_mob6,
        "X_valid": X_valid_arr,
        "y_valid_mob2": y_valid_mob2,
        "y_valid_mob6": y_valid_mob6,
        "X_test": X_test_arr,
        "y_test_mob2": y_test_mob2,
        "y_test_mob6": y_test_mob6,
        "feature_names": final_feature_names,
    }

def load_data(
    npz_path: str = DEFAULT_NPZ_PATH,
    csv_path: str = DEFAULT_RAW_CSV_PATH,
    force_recompute: bool = False
) -> Dict[str, Union[np.ndarray, list]]:
    """
    加载模型训练集、验证集和测试集 (方式二: 缓存优先)。

    1. 若 npz_path 文件存在且 force_recompute 为 False，直接毫秒级读取 NPZ 缓存。
    2. 若不存在或 force_recompute 为 True，调用特征工程流水线生成数据并自动保存缓存。
    """
    if os.path.exists(npz_path) and not force_recompute:
        print(f"📦 发现本地数据集缓存: {npz_path}，正在直接加载...")
        with np.load(npz_path, allow_pickle=True) as data:
            return {
                "X_train": data["X_train"].astype(np.float32),
                "y_train_mob2": data["y_train_mob2"].astype(np.int64),
                "y_train_mob6": data["y_train_mob6"].astype(np.int64),
                "X_valid": data["X_valid"].astype(np.float32),
                "y_valid_mob2": data["y_valid_mob2"].astype(np.int64),
                "y_valid_mob6": data["y_valid_mob6"].astype(np.int64),
                "X_test": data["X_test"].astype(np.float32),
                "y_test_mob2": data["y_test_mob2"].astype(np.int64),
                "y_test_mob6": data["y_test_mob6"].astype(np.int64),
                "feature_names": data["feature_names"].tolist(),
            }

    print(f"🚀 未找到本地缓存 {npz_path} 或设置了 force_recompute=True，启动全量数据读取与特征工程...")
    return process_lending_club_dataset(csv_path=csv_path, npz_save_path=npz_path)

if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    dataset = load_data()
    print("\n✅ 数据集加载/处理完成，概览:")
    for k, v in dataset.items():
        if isinstance(v, np.ndarray):
            print(f"  • {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  • {k}: 共 {len(v)} 个特征")
