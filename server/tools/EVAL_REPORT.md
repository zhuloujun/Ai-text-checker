# 检测效果评估报告

模型：Qwen/Qwen2.5-0.5B + Qwen/Qwen2.5-0.5B-Instruct + yuchuantian/AIGC_detector_zhv3
数据：NLPCC 2025 Task 1。校准集 392 人写 / 396 AI。

**校准前**（旧经验参数，阈值 0.5）在测试集上：AI 检出率 76.6%，人写误判率 7.1%，AUROC 0.9412

**校准后**：阈值 0.5，特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, lp_burstiness, style_cv, style_phrases；交叉验证 AUROC 0.9993

## NLPCC 测试集（训练时未见，含 DeepSeek-V3）
- AUROC 0.975；AI 检出率 90.9%；人写误判率 3.3%（197 AI / 183 人写）
- 各特征 AUROC（>0.5 越大越像 AI，<0.5 越小越像 AI）：fastdetect 0.8535，binoculars 0.1447，logit_classifier 0.9733，fastdetect_norm 0.8539，lrr 0.8526，log_rank 0.169，entropy 0.2171，top10 0.8663，lp_burstiness 0.8566，style_cv 0.2225，style_phrases 0.514

## CSL 学术摘要保留集
- AUROC 1.0；AI 检出率 100.0%；人写误判率 0.0%（180 AI / 60 人写）
- 各模型检出率：glm 100.0%，gpt4o 100.0%，qwen 100.0%
- 各特征 AUROC（>0.5 越大越像 AI，<0.5 越小越像 AI）：fastdetect 0.9662，binoculars 0.0361，logit_classifier 0.9997，fastdetect_norm 0.9641，lrr 0.9775，log_rank 0.0315，entropy 0.1517，top10 0.9758，lp_burstiness 0.5981，style_cv 0.215，style_phrases 0.8453
