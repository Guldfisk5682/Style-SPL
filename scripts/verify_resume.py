"""Compare a split 20+30 step run to uninterrupted 50 steps, including optimizer."""
import argparse
import json
from pathlib import Path

import torch


def compare(a,b,path="root"):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0,msg=path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:compare(a[k],b[k],path+"/"+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+"/"+str(i))
    else:
        assert a==b,(path,a,b)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("uninterrupted",type=Path);parser.add_argument("resumed",type=Path)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    a=torch.load(args.uninterrupted,map_location="cpu",weights_only=True)
    b=torch.load(args.resumed,map_location="cpu",weights_only=True)
    keys=["prompt","optimizer","scheduler","step","target_feature_count","valley","running_means","running_count","best_instant","rng","source_stream","target_stream"]
    for key in keys:compare(a[key],b[key],key)
    report={"exact_resume_verified":True,"compared_fields":keys,"steps":a["step"]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
