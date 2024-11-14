from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG
from mmdet.models.backbones.vmamba import MM_VSSM

# 檢查模型是否已註冊到 mmdet
model_in_mmdet = MODELS_MMDET.get("MM_VSSM")
print("MM_VSSM in mmdet registry:", model_in_mmdet)

# 檢查模型是否已註冊到 mmseg
model_in_mmseg = MODELS_MMSEG.get("MM_VSSM")
print("MM_VSSM in mmseg registry:", model_in_mmseg)
