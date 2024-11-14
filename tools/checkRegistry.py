from mmdet.registry import MODELS
# 在配置文件的頂部添加這一行
import mmdet.models.backbones.vmamba

if 'MM_VSSM' in MODELS.module_dict:
    print("MM_VSSM successfully registered!")
else:
    print("MM_VSSM not registered.")
