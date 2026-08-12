#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from sstw.flow_guidance_saved_mp4 import G1InstrumentationError,g1_exit_code,run_g1_once

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",required=True,type=Path); parser.add_argument("--reuse-first-group-archive",type=Path); args=parser.parse_args()
    try: result=run_g1_once(ROOT,args.output,reuse_first_group_archive=args.reuse_first_group_archive)
    except Exception as exc:
        chain=[]; current=exc
        while current is not None and len(chain)<8:
            chain.append({"type":type(current).__name__,"message":str(current)}); current=current.__cause__
        print(json.dumps({"status":"INSTRUMENTATION_INSUFFICIENT","diagnostic_class":"DIAGNOSTIC_ONLY","reason":type(exc).__name__,"message":str(exc),"exception_chain":chain,"formal_result":False},sort_keys=True,separators=(",",":"))); return 2
    print(json.dumps({"status":result["status"],"diagnostic_class":"DIAGNOSTIC_ONLY","output":str(args.output.resolve()),**result["execution"]},sort_keys=True,separators=(",",":"))); return g1_exit_code(result["status"])
if __name__=="__main__": raise SystemExit(main())
