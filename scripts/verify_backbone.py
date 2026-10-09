"""Real CLIP acceptance against the archived repaired B0, on GPU0."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from clip_custom import clip
from model import PromptGenerator, Custom_Clip
from style import RN50StyleExtractor, UnlabelledImages
from runtime import fix_random_seed, preserve_rng
from spl import calc_distance, soft_cross_entropy_loss


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser();p.add_argument("--baseline_root",type=Path,required=True)
    p.add_argument("--data_root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();fix_random_seed(1)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,preprocess=clip.load("RN50",device=device);model.float().eval().requires_grad_(False)
    reference=load_module("b0_reference_model",args.baseline_root/"model.py")
    sys.path.append(str(args.baseline_root.resolve()))
    # Import the actual author's scalar helpers as an independent algebra oracle.
    original=load_module("b0_reference_main",args.baseline_root/"main.py")
    classnames=sorted(d.name for d in (args.data_root/"art").iterdir() if d.is_dir())
    config=SimpleNamespace(device=device,M1=16,M2=16,style_spl_enabled=0,seed=1)
    sources=["clipart","product","real_world"]
    fix_random_seed(1);before=torch.get_rng_state().clone()
    old=reference.PromptGenerator(classnames,model,sources,"art",config)
    after_old=torch.get_rng_state().clone()
    torch.set_rng_state(before)
    # CUDA initialization RNG also needs the same reset for exact B0 comparison.
    fix_random_seed(1)
    new=PromptGenerator(classnames,model,sources,"art",config)
    torch.testing.assert_close(torch.get_rng_state(),after_old,rtol=0,atol=0)
    for (n,a),(m,b) in zip(old.named_parameters(),new.named_parameters()):
        assert n==m
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(old(),new()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for i in range(3):torch.testing.assert_close(old.forward_source(i),new.forward_source(i),rtol=0,atol=0)
    dataset=UnlabelledImages(args.data_root/"art",preprocess)
    images=torch.stack([dataset[i] for i in range(2)]).to(device)
    buffers={n:b.clone() for n,b in model.visual.named_buffers()}
    with torch.no_grad():
        ordinary=model.encode_image(images)
        with RN50StyleExtractor(model.visual) as extractor:
            stats,hooked=extractor(images)
        assert [extractor.feature_shapes[i] for i in range(4)] == [[2,256,56,56],[2,512,28,28],[2,1024,14,14],[2,2048,7,7]]
        torch.testing.assert_close(ordinary,hooked,rtol=0,atol=0)
        torch.testing.assert_close(ordinary,model.encode_image(images),rtol=0,atol=0)
        for name,b in model.visual.named_buffers():torch.testing.assert_close(b,buffers[name],rtol=0,atol=0)
        assert all(not getattr(model.visual,f"layer{i+1}")._forward_hooks for i in range(4))
    encoder=Custom_Clip(model).eval();scale=encoder.logit_scale.exp()
    labels=torch.tensor([0,1],device=device)
    losses=[]
    for prompt in (old,new):
        pooled,target=prompt()
        image,pooled_text,raw=encoder(images,pooled,prompt.tokenized_prompts)
        target_text=encoder.forward_txt(target,prompt.tokenized_prompts)
        domain_texts=[encoder.forward_txt(prompt.forward_source(i),prompt.tokenized_prompts) for i in range(3)]
        with torch.no_grad():_,base=model(images,clip.tokenize([f"A photo of a {name}" for name in classnames]).to(device))
        # Both two-branch warmup and three-branch teacher, using the original helpers.
        centroids=raw[:1].repeat(65,1)
        distance=calc_distance(raw,centroids)
        torch.testing.assert_close(distance,original.calc_distance(raw,centroids),rtol=0,atol=0)
        weights=torch.softmax(torch.stack([distance+i for i in range(3)])*-10,dim=0).detach()
        teacher_logits=image@base.t()+image@pooled_text.t()
        for i in range(3):teacher_logits+=weights[i]*(image@domain_texts[i].t())
        teacher=(scale*teacher_logits/3).softmax(-1).detach()
        prediction=scale*(image@target_text.t())
        soft=soft_cross_entropy_loss(prediction,teacher)
        torch.testing.assert_close(soft,original.soft_cross_entropy_loss(prediction,teacher),rtol=0,atol=0)
        source=F.cross_entropy(scale*(image@pooled_text.t()),labels)
        source_avg=sum(F.cross_entropy(scale*(image@t.t()),labels) for t in domain_texts)/3
        loss=source+source_avg+.5*soft
        losses.append(loss)
    torch.testing.assert_close(losses[0],losses[1],rtol=0,atol=0)
    report={"b0_prompt_parameters_exact":True,"b0_prompt_outputs_exact":True,
            "b0_complete_spl_loss_exact":True,"distance_and_soft_ce_author_helpers_exact":True,
            "visual_embeddings_unchanged":True,"bn_buffers_unchanged":True,"hooks_removed":True,
            "stage_statistics_shapes":[list(mean.shape) for mean,std in stats],"loss":losses[0].item()}
    report["feature_map_shapes"] = [extractor.feature_shapes[i] for i in range(4)]
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
