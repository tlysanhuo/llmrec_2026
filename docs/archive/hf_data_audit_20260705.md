# 官方17GB HF数据·质量缺陷账本(2026-07-05,全量精确扫描,零采样)

## 总判决
群里三大抱怨:"sid重复"证伪(多对一=聚类码设计;真脏方向=同(pid,domain)多SID/重复行/越界 全部为0);"两表大小不一"证伪(caption⊂sid严格子集,是覆盖缺口非错位);真缺陷集中在 Pid2Caption。

## 缺陷账本(行动优先级)
1. **caption模板重复→desc→SID歧义**:全表13.46%行caption非唯一(goods 25.84%!"闪电购女装1..28"单文最高1666行;ad 12.68%)→ 物料造样必须按域内caption唯一过滤(fresh_mat/P1/P2已有全局caption去重,基本规避)
2. **空caption 13,766行,100%在video/ad域**(占ad域0.97%)→ ad物料造样黑名单
3. UserProfile死列:ec_item_id_list三件套99.81%null、outer_loop_deep_target 98.5%null → 用ec_colossus_rs_*(29.79%null)替代
4. Pid2Tag:goods域覆盖0%(16.1M);live 56%/ad 36%/video 26%;格式零异常
5. caption覆盖缺口=物料训练规模上限:video 55.3%/goods 60.7%/ad 69.2%/live 99.7%
6. video事件pid悬空155,025(0.94%);ec序列硬截断500;live caption为字符串化list(须literal_eval)
7. UserProfile时间戳437M个全量零异常;42/50万用户ts乱序

## 关键硬数字
- Pid2Sid 35,914,095行=每(pid,domain)恰一条;分域:live 107,289/goods 16.09M/video-ad 2.06M/video-video 17.66M
- **各域实际码字预算**:live s_a仅802(s_b 1905/s_c 2918)/goods s_a 4279/video-ad s_a 2971/video-video s_a 5777(max 8189)——"子空间曝光"的逐域预算表
- General 152,005条(98.6%带CoT);MC矿=贫矿:可机械提金标仅245条(中文42)——贝壳的331应含自答
- domain字段真实取值含斜杠:live/goods/video\/ad/video\/video

## 可变现清单
1. ★物料造样消歧过滤器(缺陷1+2+7打包)——服务物料/ad两头
2. **world_zh扩容+1,871条**(中文多轮1179+长回答≤8000的692,改build_world_zh.py参数级)
3. 官方原生MC增补~245条(/tmp/mc_candidates.jsonl已备,合规最优,混入world_mc_clean)
4. 逐域码字预算表指导物料SID覆盖配比
