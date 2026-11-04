"""Light registration remains sufficient for inference and leaves training opt-in."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('light', ['0','1'])
def test_inference_registration_and_import_scope(light):
    code='''
import json,sys
from engine.core import YAMLConfig,GLOBAL_CONFIG
cfg=YAMLConfig('configs/dfine/dfine_hgnetv2_s_supership.yml')
cfg.yaml_cfg['HGNetv2']['pretrained']=False
model=cfg.model;post=cfg.postprocessor
assert type(model).__name__=='RTv4'
assert type(model.backbone).__name__=='HGNetv2'
assert type(model.decoder).__name__=='DFINETransformer'
assert post.num_top_queries==300
print(json.dumps({'criterion':'RTv4Criterion' in GLOBAL_CONFIG,'timm':'engine.backbone.timm_model' in sys.modules,'tensorboard':'torch.utils.tensorboard' in sys.modules}))
'''
    env=os.environ.copy();env['INFERENCE_IMPORTS']=light
    r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    d=json.loads(r.stdout.splitlines()[-1]);assert d['criterion']==(light=='0');assert d['timm']==(light=='0');assert d['tensorboard']==(light=='0')
