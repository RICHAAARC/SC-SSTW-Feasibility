import json
from pathlib import Path
from sstw.structured_initial_noise import conditions,evaluate,latent_basis,load_noise_config,trajectories
ROOT=Path(__file__).resolve().parents[1]
def test_config_and_trajectory():
 c=load_noise_config(ROOT/"configs/g0_structured_initial_noise.json");t=trajectories(c);assert tuple(t)==("A","B") and all(len(x)==13 for x in t.values())
def test_full_rank_affine_response_passes():
 import numpy as np
 c=load_noise_config(ROOT/"configs/g0_structured_initial_noise.json");t=trajectories(c);off=np.zeros((13,2));m=np.array([[.01,.003],[-.002,.012]]);o={"OFF_R1":off,"OFF_R2":off.copy(),"A":np.asarray(t["A"])@m.T,"B":np.asarray(t["B"])@m.T};assert evaluate(o,t,{"A":.01,"B":.01},c["criteria"])["passed"]
def test_zero_signal_is_json_safe_scientific_no_go():
 import numpy as np
 c=load_noise_config(ROOT/"configs/g0_structured_initial_noise.json");t=trajectories(c);off=np.zeros((13,2));result=evaluate({name:off.copy() for name in ("OFF_R1","OFF_R2","A","B")},t,{"A":0.0,"B":0.0},c["criteria"])
 assert not result["passed"] and result["condition"] is None and result["fit_relative_residual"] is None
 json.dumps(result,allow_nan=False)
def test_exact_tensor_basis_and_condition_identity():
 import pytest
 torch=pytest.importorskip("torch")
 c=load_noise_config(ROOT/"configs/g0_structured_initial_noise.json");t=trajectories(c)
 base=torch.Generator().manual_seed(4);base=torch.randn((1,16,13,40,64),generator=base)
 basis=latent_basis(torch,base,t["A"])
 assert tuple(basis.shape)==tuple(base.shape)
 assert torch.equal(basis[:,1],-basis[:,0]) and torch.equal(basis[:,3],-basis[:,2])
 assert torch.count_nonzero(basis[:,4:])==0
 values=conditions(torch,base,t,.03)
 assert torch.equal(values["OFF_R1"],values["OFF_R2"])
 for name in ("A","B"):
  ratio=(values[name]-base).square().mean().sqrt()/base.square().mean().sqrt()
  assert float(ratio)==pytest.approx(.03,abs=1e-6)
def test_sources_compile():
 json.loads((ROOT/"configs/g0_structured_initial_noise.json").read_text());compile((ROOT/"src/sstw/structured_initial_noise.py").read_text(),"m","exec");compile((ROOT/"experiments/run_g0_structured_initial_noise.py").read_text(),"r","exec")
 assert "pipe.scheduler=pipe.scheduler.__class__.from_config(pipe.scheduler.config)" in (ROOT/"src/sstw/structured_initial_noise.py").read_text()
