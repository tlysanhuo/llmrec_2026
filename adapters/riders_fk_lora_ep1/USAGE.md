# riders_fk_lora_ep1 LoRA adapter(线上总分 0.9177,2026-07-06)

- 底座:OneReason-0.8B-pretrain-competition;LoRA r32/α32/dropout0.05,lr 2e-4,1 epoch,seq32768,模板 qwen3_nothink。
- md5:adapter_model.safetensors = 0c294240875fc8ef66ad7eb01f09ceb3;adapter_config.json = d21dce60254b57fc990a0c3298c3d9e8。
- 平台只收合并后的 **bf16 全参**。合并方法(LLaMA-Factory):
  参照 configs/riders_fk_lora_ep1_merge.yaml,把 adapter_name_or_path 改成本目录路径,llamafactory-cli export 即得 1.6GB 合并权重(与我方线上 0.9177 提交逐字节等价,model.safetensors md5=c2046b60...)。
- 面板(域序=video/prod/ad/live):mat 0.2146 / action 0.0655 / topic 0.0427 / 0.0768 / 0.1258 / 0.1386 / 0.1098 / world 0.1439。
- 数据配方与毒物清单见仓库根 README。
