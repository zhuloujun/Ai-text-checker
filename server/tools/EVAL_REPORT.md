# 检测效果评估报告

每种文体各自校准：先识别段落是现代汉语、文言还是英文，再用对应的分类器和阈值判断。
语言模型：Qwen/Qwen2.5-0.5B + Qwen/Qwen2.5-0.5B-Instruct；中文分类器 yuchuantian/AIGC_detector_zhv3；英文分类器 desklib/ai-text-detector-v1.01。

“检出率”= AI 文本被判为 AI 的比例；“误判率”= 人写文本被误判为 AI 的比例；AUROC 1 为完美区分，0.5 为随机。

## 现代汉语
- 数据：NLPCC 2025 Task 1（CSL 学术摘要 / 新闻 / 作文；GPT-4o、GLM-4、Qwen）
- 校准集 392 人写 / 396 AI；阈值 0.5；交叉验证 AUROC 0.9993；特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, lp_burstiness, style_cv, style_phrases
- **NLPCC 测试集（训练时未见，含 DeepSeek-V3）**（197 AI / 183 人写）：AUROC 0.975；检出率 90.9%；误判率 3.3%
  - 各特征 AUROC：fastdetect 0.8535，binoculars 0.1447，logit_classifier 0.9733，fastdetect_norm 0.8539，lrr 0.8526，log_rank 0.169，entropy 0.2171，top10 0.8663，lp_burstiness 0.8566，style_cv 0.2225，style_phrases 0.514
- **CSL 学术摘要保留集**（180 AI / 60 人写）：AUROC 1.0；检出率 100.0%；误判率 0.0%
  - 各来源检出率：glm 100%，gpt4o 100%，qwen 100%
  - 各来源误判率：human 0%
  - 各特征 AUROC：fastdetect 0.9662，binoculars 0.0361，logit_classifier 0.9997，fastdetect_norm 0.9641，lrr 0.9775，log_rank 0.0315，entropy 0.1517，top10 0.9758，lp_burstiness 0.5981，style_cv 0.215，style_phrases 0.8453

## 文言
- 数据：NiuTrans 古文语料（人写）+ 大语言模型生成的文言样本
- 校准集 236 人写 / 80 AI；阈值 0.7309；交叉验证 AUROC 0.9427；特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, lp_burstiness, style_cv, style_phrases
- **文言保留集（另一组古籍 + 未参与校准的 AI 文言）**（40 AI / 143 人写）：AUROC 0.967；检出率 72.5%；误判率 2.8%
  - 各来源检出率：llm-classical 72%
  - 各来源误判率：入蜀记 0%，唐传奇 0%，困学纪闻 9%，幽明录 0%，搜神记 0%，新唐书 9%，旧五代史 0%，明夷待访录 0%，武林旧事 0%，聊斋志异 0%，西湖梦寻 18%，资治通鉴 0%，金史 0%，陶庵梦忆 0%
  - 各特征 AUROC：fastdetect 0.9231，binoculars 0.0727，logit_classifier 0.718，fastdetect_norm 0.9208，lrr 0.9413，log_rank 0.0437，entropy 0.1014，top10 0.9411，lp_burstiness 0.5502，style_cv 0.4753，style_phrases 0.5
