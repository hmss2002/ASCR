import argparse,json,re,sys
from datetime import datetime
from pathlib import Path

def slug(text):
    value=re.sub(r'[^a-zA-Z0-9]+','-',text.strip().lower()).strip('-')
    return (value or 'prompt')[:80]

def read_prompts(path, limit, offset):
    prompts=[line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    prompts=prompts[max(0,offset):]
    return prompts if limit is None else prompts[:max(0,limit)]

def main():
    parser=argparse.ArgumentParser(description='Generate text-to-image samples with BAGEL.')
    parser.add_argument('--bagel-root',default='.deps/Bagel')
    parser.add_argument('--model-path',default='models/BAGEL-7B-MoT')
    parser.add_argument('--prompt-file',default='configs/benchmarks/prompts/t2i_compbench_hard64.txt')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--limit',type=int,default=None)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--mode',type=int,default=1,choices=[1,2,3])
    parser.add_argument('--image-ratio',default='1:1')
    parser.add_argument('--cfg-text-scale',type=float,default=4.0)
    parser.add_argument('--cfg-interval',type=float,default=0.4)
    parser.add_argument('--timestep-shift',type=float,default=3.0)
    parser.add_argument('--num-timesteps',type=int,default=50)
    parser.add_argument('--cfg-renorm-min',type=float,default=0.0)
    parser.add_argument('--cfg-renorm-type',default='global')
    parser.add_argument('--max-think-tokens',type=int,default=1024)
    parser.add_argument('--show-thinking',action='store_true')
    parser.add_argument('--do-sample',action='store_true')
    parser.add_argument('--text-temperature',type=float,default=0.3)
    parser.add_argument('--skip-existing',action='store_true',help='Skip prompts whose output image already exists')
    args=parser.parse_args()
    out=Path(args.output_dir); images=out/'images'; records=out/'records'
    images.mkdir(parents=True,exist_ok=True); records.mkdir(parents=True,exist_ok=True)
    prompts=read_prompts(args.prompt_file,args.limit,args.offset)
    if not prompts: raise ValueError('No prompts selected')
    import types
    class DummyGradioObject:
        def __init__(self,*args,**kwargs): pass
        def __call__(self,*args,**kwargs): return self
        def __getattr__(self,name): return self
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def queue(self,*args,**kwargs): return self
        def launch(self,*args,**kwargs): return self
    gradio_stub=types.ModuleType('gradio')
    gradio_stub.__file__='<dummy-gradio>'
    gradio_stub.__path__=[]
    gradio_stub.__spec__=None
    gradio_stub.__getattr__=lambda name: DummyGradioObject()
    gradio_stub.update=lambda **kwargs: kwargs
    sys.modules['gradio']=gradio_stub
    sys.path.insert(0,str(Path(args.bagel_root).resolve()))
    sys.argv=['app.py','--model_path',args.model_path,'--mode',str(args.mode)]
    print(json.dumps({'event':'bagel_load_start','model_path':args.model_path,'prompts':len(prompts)}),flush=True)
    import app
    print(json.dumps({'event':'bagel_load_done'}),flush=True)
    started=datetime.utcnow().isoformat(timespec='seconds')+'Z'
    results=[]
    for local_index,prompt in enumerate(prompts):
        index=args.offset+local_index; seed=args.seed+index if args.seed>=0 else args.seed
        stem=f'prompt_{index:03d}-{slug(prompt)}'
        image_path=images/f'{stem}.png'; record_path=records/f'{stem}.json'
        if args.skip_existing and image_path.exists():
            print(json.dumps({'event':'bagel_prompt_skip','index':index,'image':str(image_path)}),flush=True)
            if record_path.exists():
                results.append(json.loads(record_path.read_text(encoding='utf-8')))
            continue
        print(json.dumps({'event':'bagel_prompt_start','index':index,'prompt':prompt}),flush=True)
        image, thought=app.text_to_image(prompt,show_thinking=args.show_thinking,cfg_text_scale=args.cfg_text_scale,cfg_interval=args.cfg_interval,timestep_shift=args.timestep_shift,num_timesteps=args.num_timesteps,cfg_renorm_min=args.cfg_renorm_min,cfg_renorm_type=args.cfg_renorm_type,max_think_token_n=args.max_think_tokens,do_sample=args.do_sample,text_temperature=args.text_temperature,seed=seed,image_ratio=args.image_ratio)
        if image is None: raise RuntimeError('BAGEL returned no image')
        image.save(image_path)
        record={'index':index,'prompt':prompt,'bagel_image':str(image_path),'image':str(image_path),'thought_text':thought,'seed':seed,'model':'BAGEL-7B-MoT','model_path':args.model_path,'record_path':str(record_path),'generation':{'mode':args.mode,'image_ratio':args.image_ratio,'cfg_text_scale':args.cfg_text_scale,'cfg_interval':args.cfg_interval,'timestep_shift':args.timestep_shift,'num_timesteps':args.num_timesteps,'cfg_renorm_min':args.cfg_renorm_min,'cfg_renorm_type':args.cfg_renorm_type,'show_thinking':args.show_thinking,'do_sample':args.do_sample,'text_temperature':args.text_temperature}}
        record_path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        results.append(record)
        print(json.dumps({'event':'bagel_prompt_done','index':index,'image':str(image_path)}),flush=True)
    suite={'protocol':'bagel_t2i_generation_suite_v1','model':'BAGEL-7B-MoT','model_path':args.model_path,'prompt_file':args.prompt_file,'offset':args.offset,'limit':args.limit,'started_at':started,'finished_at':datetime.utcnow().isoformat(timespec='seconds')+'Z','results':results}
    suite_path=out/'suite.json'; suite_path.write_text(json.dumps(suite,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'event':'bagel_suite_done','suite':str(suite_path),'count':len(results)}),flush=True)

if __name__=='__main__': main()
