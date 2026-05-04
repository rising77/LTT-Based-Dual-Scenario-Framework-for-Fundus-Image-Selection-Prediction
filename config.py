"""
configs/config.py
=================

中央配置文件：把所有路径、超参与 LTT 实验设置集中放在一处，
方便复现与上传 GitHub。其他模块只 `from configs import config as C` 引用。
"""
from pathlib import Path

# ============================ 1. 路径 ============================
ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_ROOT   = ROOT_DIR / "data" / "aptos2019"        # 放 train.csv 和 train_images/
OUTPUT_DIR  = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_BEST   = OUTPUT_DIR / "efficientnet_b3_best.pth"
CKPT_LAST   = OUTPUT_DIR / "efficientnet_b3_last.pth"
UNC_CAL_NPZ  = OUTPUT_DIR / "uncertainties_cal.npz"
UNC_TEST_NPZ = OUTPUT_DIR / "uncertainties_test.npz"

# ============================ 2. 数据切分 ============================
# APTOS-2019 共 3662 张，按 6:2:2 分层切分 (train / cal / test)
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2     # 同时充当 LTT 校准集
TEST_RATIO  = 0.2
NUM_CLASSES = 5
IMG_SIZE    = 300     # EfficientNet-B3 推荐分辨率

# ============================ 3. 训练超参 ============================
SEED              = 42
BATCH_SIZE        = 32
NUM_WORKERS       = 4
MODEL_NAME        = "efficientnet_b3"
STAGE1_EPOCHS     = 10
STAGE2_EPOCHS     = 30
HEAD_LR           = 1e-4
BACKBONE_LR_RATIO = 0.1     # backbone lr = head lr * 0.1
WEIGHT_DECAY      = 1e-5
FOCAL_GAMMA       = 2.0

# ============================ 4. LTT 风险控制 ============================
DELTA            = 0.10                 # 显著性水平 δ
ALPHA_LIST       = [0.10, 0.125, 0.15, 0.175]   # 论文报告的四个风险预算
COVERAGE_LEVELS  = [0.50, 0.60, 0.70]   # 反向视角的目标覆盖率
N_REPEATS        = 100                  # 重抽样次数
GRID_NUM         = 99                   # λ 网格大小（0.01,...,0.99）
CAL_SPLIT_RATIO  = 0.10                 # 校准集内 9:1 拆分；10% 用来定 FST 顺序
