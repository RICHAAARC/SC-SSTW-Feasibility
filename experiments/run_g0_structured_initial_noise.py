#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from sstw.structured_initial_noise import run
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);a=p.parse_args()
 try:r=run(ROOT,a.output)
 except Exception as e:
  chain=[];x=e
  while x is not None and len(chain)<8:chain.append({"type":type(x).__name__,"message":str(x)});x=x.__cause__
  print(json.dumps({"status":"INSTRUMENTATION_INSUFFICIENT","diagnostic_class":"DIAGNOSTIC_ONLY","exception_chain":chain},separators=(",",":")));return 2
 print(json.dumps({"status":r["status"],"diagnostic_class":"DIAGNOSTIC_ONLY",**r["execution"]},separators=(",",":")));return 0 if r["status"].endswith("FEASIBLE") and not r["status"].endswith("NOT_FEASIBLE") else 3
if __name__=="__main__":raise SystemExit(main())
