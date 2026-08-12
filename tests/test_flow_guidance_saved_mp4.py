from pathlib import Path
import json
import pytest
from sstw.flow_guidance_saved_mp4 import evaluate_group,frozen_trajectories,load_g1_config

ROOT=Path(__file__).resolve().parents[1]

def test_config_and_trajectories_are_frozen():
    config=load_g1_config(ROOT/"configs/g1_flow_guidance_saved_mp4.json"); trajectories=frozen_trajectories(config)
    assert tuple(trajectories)==("A","B") and all(len(v)==13 for v in trajectories.values())
    assert trajectories["A"] != trajectories["B"]

def test_affine_mixed_channel_is_admitted_but_rank_collapse_is_not():
    np=pytest.importorskip("numpy"); config=load_g1_config(ROOT/"configs/g1_flow_guidance_saved_mp4.json"); t=frozen_trajectories(config)
    matrix=np.array([[1.5,-.75],[-.8,1.1]])*.01; off=np.zeros((13,2)); obs={"OFF_R1":off,"OFF_R2":off.copy()}
    for name in ("A","B"): obs[name]=np.asarray(t[name])@matrix.T
    result=evaluate_group(obs,t,{"A":.005,"B":.005},config["criteria"])
    assert result["passed"] and result["condition"]<10
    obs["B"]=np.asarray(t["B"])@np.array([[.01,0],[.01,0]]).T
    assert not evaluate_group(obs,t,{"A":.005,"B":.005},config["criteria"])["passed"]

def test_json_and_sources_compile():
    json.loads((ROOT/"configs/g1_flow_guidance_saved_mp4.json").read_text())
    compile((ROOT/"src/sstw/flow_guidance_saved_mp4.py").read_text(),"module","exec")
    compile((ROOT/"experiments/run_g1_flow_guidance_saved_mp4.py").read_text(),"runner","exec")
