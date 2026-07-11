# probe_mat_s3.py — 物料样本3 beam锁定/扇宽探针(2026-07-09 标定)
# 锚位:riders_fk_lora_ep1(7题)=35/13;exp_seed_ep3(8题)=55/18(平台日志 29/13 与 54/18,扇宽逐位复现,锁定+6系统偏移)
# 用法:CUDA_VISIBLE_DEVICES=N python scripts/eval/probe_mat_s3.py <merged_model_path>
# 判读:扇宽≥17 且锁定≥50 ⇒ 8题签名;扇宽≤13 ⇒ 7题及以下。单题探针,只作上传前预读,不作收益证据。
import re, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1) 从日志重建样本3的精确prompt(去掉日志换行伪影)
txt = open('logs/eval/riders_fk_lora_ep1_20260706.log', errors='ignore').read()
s = txt.find('Task [4/8]'); e = txt.find('Task [5/8]', s); sec = txt[s:e]
blocks = re.split(r'Sample ID: (\d+)', sec)
body = blocks[blocks.index('3')+1]
inp = body[body.find('Input:')+6: body.find('Output[0]:')]
sysm = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end', inp, re.S).group(1).replace('\n','')
um = re.search(r'<\|im_start\|>user\n(.*?)/no_think', inp, re.S).group(1)
head, _, desc = um.partition('：\n\n')
desc = '\n\n'.join(p.replace('\n','') for p in desc.split('\n\n'))
prompt = f"<|im_start|>system\n{sysm}<|im_end|>\n<|im_start|>user\n请解析以下视频内容并输出对应的视频token：\n\n{desc}/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|video_begin|>"
print('desc长度:', len(desc), 'desc头:', desc[:40])

model_path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
ids = tok(prompt, return_tensors='pt').input_ids.cuda()
with torch.no_grad():
    out = model.generate(ids, num_beams=64, num_return_sequences=64, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
seqs = [tok.decode(o[ids.shape[1]:], skip_special_tokens=False) for o in out]
lock = sum(1 for q in seqs if q.startswith('<s_a_2391>'))
fan = len(set(re.findall(r'<s_a_2391><s_b_(\d+)>', ''.join(seqs))))
top_as = {}
for q in seqs:
    m = re.match(r'<s_a_(\d+)>', q)
    if m: top_as[m.group(1)] = top_as.get(m.group(1), 0) + 1
print(f'MODEL={model_path}')
print(f'锁定a2391={lock}/64  a2391下s_b扇宽={fan}  top_a分布={sorted(top_as.items(), key=lambda x:-x[1])[:4]}')
