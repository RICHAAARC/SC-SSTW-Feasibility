#!/usr/bin/env python3
"""CPU-only A1/A2 bounded development gate; never reads fresh held-out cases."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sc_sstw_feasibility.aisb import BurstTemplate, affine_burst_residual
from sc_sstw_feasibility.calibration import calibrate_from_pilot_pairs, equalize_observations
from sc_sstw_feasibility.learned_observation import TEMPORAL_POINTS, CALIBRATION_INDICES, PER_VIDEO_HELD_OUT_INDICES

FIT=(41001,41002,41003,41004); DEV=(41005,41006); STARTS=tuple(range(8))

def canonical(v): return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def load_features(run,i):
    x=np.asarray(json.loads((run/'artifacts'/'datasets'/str(i)/'features.json').read_text())['features'],dtype=np.float64)
    if x.shape!=(13,30) or not np.isfinite(x).all(): raise ValueError(f'invalid features {i}')
    return x
def transform(x,candidate,cfg):
    n=cfg['normalization']; med=np.median(x,axis=0); mad=np.median(np.abs(x-med),axis=0)
    z=np.clip((x-med)/np.maximum(n['mad_scale']*mad,n['mad_floor']),n['clip_min'],n['clip_max'])
    if candidate=='A1': return z
    y=np.empty_like(z); y[0]=z[0]-z[1]; y[-1]=z[-1]-z[-2]; y[1:-1]=z[1:-1]-(z[:-2]+z[2:])/2
    return y
def fit(xs,ridge):
    x=np.concatenate(xs); y=np.tile(np.asarray(TEMPORAL_POINTS),(len(xs),1)); d=np.column_stack((x,np.ones(len(x))))
    p=ridge*np.eye(d.shape[1]); p[-1,-1]=0
    return np.linalg.solve(d.T@d+p,d.T@y)
def observe(x,b): return np.column_stack((x,np.ones(len(x))))@b
def second(x): return float(np.linalg.svd(x,compute_uv=False)[1])
def metrics(q,start):
    template=BurstTemplate('burst_alpha',tuple(TEMPORAL_POINTS[:6])); window=q[start:start+6]
    residual=float(affine_burst_residual(window.tolist(),template)); pairs=[(TEMPORAL_POINTS[i],q[start+i].tolist()) for i in CALIBRATION_INDICES]
    c=calibrate_from_pilot_pairs(pairs); s=np.linalg.svd(np.asarray(c.matrix),compute_uv=False)
    eq=np.asarray(equalize_observations(q[[start+i for i in PER_VIDEO_HELD_OUT_INDICES]].tolist(),c)); target=np.asarray([TEMPORAL_POINTS[i] for i in PER_VIDEO_HELD_OUT_INDICES])
    centered=(q-q.mean(0))/math.sqrt(13)
    return {'residual':residual,'global_s2':second(centered),'affine_s2':float(s[1]),'condition':float(s[0]/s[1]) if s[1] else 1e300,'held_out_mse':float(np.mean((eq-target)**2))}
def passes(m,t): return m['residual']<=t['max_residual'] and m['global_s2']>=t['min_global_s2'] and m['affine_s2']>=t['min_affine_s2'] and m['condition']<=t['max_condition'] and m['held_out_mse']<=t['max_held_out_mse']
def candidate_result(raw,candidate,cfg):
    tx={i:transform(raw[i],candidate,cfg) for i in FIT+DEV}; loo=[]
    for held in FIT:
        b=fit([tx[i] for i in FIT if i!=held],cfg['readout']['ridge']); loo.append(metrics(observe(tx[held],b),0))
    t={'max_residual':max(x['residual'] for x in loo),'min_global_s2':min(x['global_s2'] for x in loo),'min_affine_s2':min(x['affine_s2'] for x in loo),'max_condition':max(x['condition'] for x in loo),'max_held_out_mse':max(x['held_out_mse'] for x in loo)}
    b=fit([tx[i] for i in FIT],cfg['readout']['ridge']); cases={}
    for i in FIT+DEV:
        q=observe(tx[i],b); windows=[]
        for start in STARTS:
            m=metrics(q,start); windows.append({'start':start,'correct':start==0,**m,'final_accepted':passes(m,t),'rejection_reasons':[k for k,v in [('residual',m['residual']<=t['max_residual']),('global_s2',m['global_s2']>=t['min_global_s2']),('affine_s2',m['affine_s2']>=t['min_affine_s2']),('condition',m['condition']<=t['max_condition']),('held_out_mse',m['held_out_mse']<=t['max_held_out_mse'])] if not v]})
        cases[str(i)]={'observation':q.tolist(),'windows':windows}
    gate=all(cases[str(i)]['windows'][0]['final_accepted'] and not any(w['final_accepted'] for w in cases[str(i)]['windows'][1:]) for i in DEV)
    return {'candidate':candidate,'thresholds_train_loo_only':t,'loo_correct_window_metrics':loo,'cases':cases,'development_gate_pass':gate}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    cfg=json.loads(a.config.read_text()); raw={i:load_features(a.run,i) for i in FIT+DEV}; results=[candidate_result(raw,c,cfg) for c in ('A1','A2')]
    selected='A1' if results[0]['development_gate_pass'] else ('A2' if results[1]['development_gate_pass'] else None)
    out={'protocol_id':cfg['protocol_id'],'config_sha256':hashlib.sha256(canonical(cfg)).hexdigest(),'old_l1_status':'Contradicted','fresh_held_out_read':False,'gpu_executed':False,'candidates':results,'selected':selected,'decision':'GPU_READY' if selected else 'STOP_NOT_GPU_READY','l2_admission':False}
    a.output.mkdir(parents=True,exist_ok=False); (a.output/'development_gate.json').write_bytes(canonical(out)+b'\n'); (a.output/'config.json').write_bytes(canonical(cfg)+b'\n')
    (a.output/'command.txt').write_text(f'python experiments/run_learned_observation_l1_v2_development.py --run {a.run} --config {a.config} --output {a.output}\n',encoding='utf-8')
    lines=[]
    for f in sorted(a.output.iterdir()): lines.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}')
    (a.output/'checksums.sha256').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(out['decision'])
if __name__=='__main__': main()
