"""Verify separate source/target gradient paths through the real frozen CLIP."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from clip_custom import clip
from model import PromptGenerator, Custom_Clip
from style import UnlabelledImages
from runtime import fix_random_seed, gradient_check
from spl import soft_cross_entropy_loss


def main():
    p=argparse.ArgumentParser();p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--data_root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();fix_random_seed(1)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,preprocess=clip.load("RN50",device=device);model.float().eval().requires_grad_(False)
    bank=torch.load(args.bank,map_location="cpu",weights_only=True)
    source_names=bank["metadata"]["source_domain_order"];target=bank["metadata"]["target_domain"]
    classnames=sorted(d.name for d in (args.data_root/source_names[0]).iterdir() if d.is_dir())
    config=SimpleNamespace(device=device,M1=16,M2=16,style_spl_enabled=1,seed=1)
    fix_random_seed(1);prompt=PromptGenerator(classnames,model,source_names,target,config,style_bank=bank)
    rng_after=torch.get_rng_state().clone();cuda_after=torch.cuda.get_rng_state_all()
    config.style_spl_enabled=0
    fix_random_seed(1);b0=PromptGenerator(classnames,model,source_names,target,config)
    torch.testing.assert_close(prompt.ctx_cls,b0.ctx_cls,rtol=0,atol=0)
    torch.testing.assert_close(torch.get_rng_state(),rng_after,rtol=0,atol=0)
    for a,b in zip(cuda_after,torch.cuda.get_rng_state_all()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not any(n.startswith(("ctx_source","ctx_target")) for n,p in prompt.named_parameters())
    encoder=Custom_Clip(model).eval();scale=encoder.logit_scale.exp()
    images=UnlabelledImages(args.data_root/target,preprocess)
    batch=torch.stack([images[0],images[1]]).to(device)
    with torch.no_grad():
        feature=encoder.forward_img(batch)
        _,base=model(batch,clip.tokenize([f"A photo of a {name}" for name in classnames]).to(device))
        teacher=(scale*feature@base.t()).softmax(-1).detach()
    report={"class_initialization_matches_b0":True,"training_rng_matches_b0":True,
            "no_old_learnable_domain_prompts":True,"fixed_bank_buffers":True,"gradient_paths":{}}
    bank_before={n:b.clone() for n,b in prompt.style_bank.named_buffers()}
    for branch in ("source","target"):
        prompt.zero_grad(set_to_none=True)
        tokens=prompt.forward_source(0) if branch=="source" else prompt()[1]
        assert tokens.shape==(65,77,512)
        text=encoder.forward_txt(tokens,prompt.tokenized_prompts)
        logits=scale*(feature@text.t())
        loss=F.cross_entropy(logits,torch.tensor([0,1],device=device)) if branch=="source" else soft_cross_entropy_loss(logits,teacher)
        loss.backward()
        norms=gradient_check(prompt,{branch:loss})
        assert all(value>0 for value in norms.values())
        assert all(parameter.grad is None for parameter in model.parameters())
        report["gradient_paths"][branch]=norms
    for n,b in prompt.style_bank.named_buffers():
        assert not b.requires_grad
        torch.testing.assert_close(b,bank_before[n],rtol=0,atol=0)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
