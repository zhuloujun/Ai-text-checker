# 检测效果评估报告

每种文体各自校准：先识别段落是现代汉语、文言还是英文，再用对应的分类器和阈值判断。
语言模型：Qwen/Qwen2.5-1.5B + Qwen/Qwen2.5-1.5B-Instruct；中文分类器 yuchuantian/AIGC_detector_zhv3；英文分类器 desklib/ai-text-detector-v1.01。

“检出率”= AI 文本被判为 AI 的比例；“误判率”= 人写文本被误判为 AI 的比例；AUROC 1 为完美区分，0.5 为随机。

## 现代汉语
- 数据：NLPCC 2025 Task 1（CSL 学术摘要 / 新闻 / 作文；GPT-4o、GLM-4、Qwen）
- 校准集 400 人写 / 396 AI；阈值 0.5；交叉验证 AUROC 0.9998；特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, lp_burstiness, style_cv, style_phrases
- 特征组合比较（校准集交叉验证 AUROC）：全部特征 0.9998，三个主信号 0.9999，语言模型特征 0.9782 → 选用全部特征
- **NLPCC 测试集（训练时未见，含 DeepSeek-V3）**（200 AI / 200 人写）：AUROC 0.9738；检出率 84.0%；误判率 3.5%
  - 各特征 AUROC：fastdetect 0.8466，binoculars 0.1525，logit_classifier 0.9727，fastdetect_norm 0.846，lrr 0.848，log_rank 0.1702，entropy 0.1818，top10 0.8686，lp_burstiness 0.8842，style_cv 0.2175，style_phrases 0.5042
- **CSL 学术摘要保留集**（180 AI / 60 人写）：AUROC 1.0；检出率 100.0%；误判率 0.0%
  - 各来源检出率：glm 100%，gpt4o 100%，qwen 100%
  - 各来源误判率：human 0%
  - 各特征 AUROC：fastdetect 0.9587，binoculars 0.0421，logit_classifier 0.9997，fastdetect_norm 0.9558，lrr 0.9843，log_rank 0.0313，entropy 0.1302，top10 0.9761，lp_burstiness 0.6775，style_cv 0.2154，style_phrases 0.8452

## 现代汉语短段
- 数据：NLPCC 2025 Task 1 样本截成 80–260 字的短段
- 校准集 362 人写 / 370 AI；阈值 0.7692；交叉验证 AUROC 0.9967；特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, lp_burstiness, style_cv, style_phrases
- 特征组合比较（校准集交叉验证 AUROC）：全部特征 0.9967，三个主信号 0.9957，语言模型特征 0.9352 → 选用全部特征
- **NLPCC 测试集截成 80–260 字的短段（含本仓库 AI 读后感 / 散文）**（207 AI / 186 人写）：AUROC 0.9311；检出率 76.8%；误判率 3.8%
  - 各来源检出率：repo-ai-zh-essay 96%
  - 各特征 AUROC：fastdetect 0.821，binoculars 0.1762，logit_classifier 0.9266，fastdetect_norm 0.8217，lrr 0.7326，log_rank 0.2492，entropy 0.3094，top10 0.7822，lp_burstiness 0.7257，style_cv 0.3441，style_phrases 0.5053

## 英文
- 数据：MAGE（人写文本与 GPT-3.5 / GPT-4 等生成文本）
- 校准集 296 人写 / 300 AI；阈值 0.6601；交叉验证 AUROC 0.9653；特征 fastdetect, binoculars, logit_classifier
- **MAGE：GPT-4 在未见过的领域生成的文本**（150 AI / 150 人写）：AUROC 0.9856；检出率 95.3%；误判率 5.3%
  - 各来源检出率：cnn_gpt4 91%，imdb_gpt4 96%，pubmed_gpt4 94%，dialogsum_gpt4 100%
  - 各来源误判率：pubmed_human 6%，dialogsum_human 15%，imdb_human 0%，cnn_human 0%
  - 各特征 AUROC：fastdetect 0.7613，binoculars 0.2452，logit_classifier 0.9828，fastdetect_norm 0.7597，lrr 0.7597，log_rank 0.1868，entropy 0.2384，top10 0.8018，lp_burstiness 0.7254，style_cv 0.3014，style_phrases 0.6956
- **MAGE：GPT-4 文本经改写后（含本仓库英文 AI 样本）**（173 AI / 150 人写）：AUROC 0.8822；检出率 71.1%；误判率 13.3%
  - 各来源检出率：pubmed_gpt4_para 84%，cnn_gpt4_para 58%，imdb_gpt4_para 77%，dialogsum_gpt4_para 74%，repo-ai-english 57%
  - 各来源误判率：pubmed_human 11%，imdb_human 0%，pubmed_human_para 0%，cnn_human 6%，dialogsum_human_para 39%，cnn_human_para 19%，imdb_human_para 24%，dialogsum_human 0%
  - 各特征 AUROC：fastdetect 0.5498，binoculars 0.4572，logit_classifier 0.8832，fastdetect_norm 0.5465，lrr 0.6303，log_rank 0.3566，entropy 0.3618，top10 0.6513，lp_burstiness 0.5175，style_cv 0.3114，style_phrases 0.5934

## 文言
- 数据：NiuTrans 古文语料（人写）+ 大语言模型生成的文言样本
- 校准集 236 人写 / 80 AI；阈值 0.6724；交叉验证 AUROC 0.9666；特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, lp_burstiness, style_cv, style_phrases
- 特征组合比较（校准集交叉验证 AUROC）：全部特征 0.9666，三个主信号 0.8793，语言模型特征 0.9443 → 选用全部特征
- **文言保留集（另一组古籍 + 未参与校准的 AI 文言）**（40 AI / 143 人写）：AUROC 0.9572；检出率 82.5%；误判率 5.6%
  - 各来源检出率：llm-classical 82%
  - 各来源误判率：入蜀记 0%，唐传奇 0%，困学纪闻 18%，幽明录 0%，搜神记 0%，新唐书 0%，旧五代史 9%，明夷待访录 18%，武林旧事 0%，聊斋志异 0%，西湖梦寻 9%，资治通鉴 18%，金史 0%，陶庵梦忆 0%
  - 各特征 AUROC：fastdetect 0.9117，binoculars 0.0897，logit_classifier 0.718，fastdetect_norm 0.9101，lrr 0.9608，log_rank 0.0497，entropy 0.1253，top10 0.9465，lp_burstiness 0.5881，style_cv 0.4753，style_phrases 0.5

## 诗词
- 数据：ChangAn 当代旧体诗词（人写）+ DeepSeek / 豆包 / GPT-4.1 生成诗词；诗词专用分类器（ChangAn 训练集微调）
- 校准集 400 人写 / 399 AI；阈值 0.7865；交叉验证 AUROC 0.9776；特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, style_cv, style_phrases, logit_classifier_mpu
- 特征组合比较（四种独立对照的 AUROC，按最差情况选）：
  - 诗词分类器 + 语言模型：ChangAn 保留集 0.9596，故事诗 vs 当代人写 0.7317，ChangAn AI vs 唐宋名篇 0.4735，故事诗 vs 唐宋名篇 0.0683（最差 0.0683）
  - 诗词分类器 + 通用分类器 + 语言模型：ChangAn 保留集 0.9581，故事诗 vs 当代人写 0.6825，ChangAn AI vs 唐宋名篇 0.4526，故事诗 vs 唐宋名篇 0.0749（最差 0.0749）
  - 通用分类器 + 语言模型：ChangAn 保留集 0.9015，故事诗 vs 当代人写 0.6828，ChangAn AI vs 唐宋名篇 0.0735，故事诗 vs 唐宋名篇 0.019（最差 0.019）
  - 只用语言模型：ChangAn 保留集 0.8665，故事诗 vs 当代人写 0.7508，ChangAn AI vs 唐宋名篇 0.0315，故事诗 vs 唐宋名篇 0.0109（最差 0.0109）
- 选用：诗词分类器 + 通用分类器 + 语言模型；**诗词结果只作参考，不计入 AI 率**
- **ChangAn 保留集（另一批作者 + 没见过的 Kimi-K2 与其他模型的新诗词）**（300 AI / 300 人写）：AUROC 0.9581；检出率 76.0%；误判率 2.3%
  - 各来源检出率：Deepseek 69%，gpt-4.1 87%，kimi-k2 71%，seed 77%
  - 各来源误判率：human 2%
  - 各特征 AUROC：fastdetect 0.7767，binoculars 0.2338，logit_classifier 0.9532，fastdetect_norm 0.7742，lrr 0.8432，log_rank 0.1452，entropy 0.2052，top10 0.7993，lp_burstiness 0.5925，style_cv 0.4707，style_phrases 0.5
- **复述故事情节的 AI 诗词（本仓库自带，40 首）**（40 AI / 0 人写）：AUROC None；检出率 2.5%
  - 各来源检出率：repo-ai-story-poem 2%
  - 各特征 AUROC：
- **唐诗三百首 + 宋词三百首（人写名篇，检查误判）**（0 AI / 646 人写）：AUROC None；误判率 68.3%
  - 各来源误判率：唐诗三百首 68%，宋词三百首 69%
  - 各特征 AUROC：
