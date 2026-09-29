# 检测效果评估报告

每种文体各自校准：先识别段落是现代汉语、文言还是英文，再用对应的分类器和阈值判断。
语言模型：Qwen/Qwen2.5-0.5B + Qwen/Qwen2.5-0.5B-Instruct；中文分类器 yuchuantian/AIGC_detector_zhv3；英文分类器 desklib/ai-text-detector-v1.01。

“检出率”= AI 文本被判为 AI 的比例；“误判率”= 人写文本被误判为 AI 的比例；AUROC 1 为完美区分，0.5 为随机。

## 现代汉语
- 数据：NLPCC 2025 Task 1（CSL 学术摘要 / 新闻 / 作文；GPT-4o、GLM-4、Qwen）
- 校准集 392 人写 / 396 AI；阈值 0.5；交叉验证 AUROC 0.9993；特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, lp_burstiness, style_cv, style_phrases
- 特征组合比较（校准集交叉验证 AUROC）：全部特征 0.9993，三个主信号 0.9994，语言模型特征 0.9699 → 选用全部特征
- **NLPCC 测试集（训练时未见，含 DeepSeek-V3）**（197 AI / 183 人写）：AUROC 0.975；检出率 90.9%；误判率 3.3%
  - 各特征 AUROC：fastdetect 0.8535，binoculars 0.1447，logit_classifier 0.9733，fastdetect_norm 0.8539，lrr 0.8526，log_rank 0.169，entropy 0.2171，top10 0.8663，lp_burstiness 0.8566，style_cv 0.2225，style_phrases 0.514
- **CSL 学术摘要保留集**（180 AI / 60 人写）：AUROC 1.0；检出率 100.0%；误判率 0.0%
  - 各来源检出率：glm 100%，gpt4o 100%，qwen 100%
  - 各来源误判率：human 0%
  - 各特征 AUROC：fastdetect 0.9662，binoculars 0.0361，logit_classifier 0.9997，fastdetect_norm 0.9641，lrr 0.9775，log_rank 0.0315，entropy 0.1517，top10 0.9758，lp_burstiness 0.5981，style_cv 0.215，style_phrases 0.8453

## 现代汉语短段
- 数据：NLPCC 2025 Task 1 样本截成 80–260 字的短段
- 校准集 362 人写 / 370 AI；阈值 0.7826；交叉验证 AUROC 0.9974；特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, lp_burstiness, style_cv, style_phrases
- 特征组合比较（校准集交叉验证 AUROC）：全部特征 0.9974，三个主信号 0.997，语言模型特征 0.9425 → 选用全部特征
- **NLPCC 测试集截成 80–260 字的短段（含本仓库 AI 读后感 / 散文）**（207 AI / 186 人写）：AUROC 0.9276；检出率 74.9%；误判率 2.7%
  - 各来源检出率：repo-ai-zh-essay 96%
  - 各特征 AUROC：fastdetect 0.8117，binoculars 0.1871，logit_classifier 0.9266，fastdetect_norm 0.8107，lrr 0.7218，log_rank 0.2551，entropy 0.3652，top10 0.7564，lp_burstiness 0.7114，style_cv 0.3441，style_phrases 0.5053

## 英文
- 数据：MAGE（人写文本与 GPT-3.5 / GPT-4 等生成文本）
- 校准集 296 人写 / 300 AI；阈值 0.6437；交叉验证 AUROC 0.9649；特征 fastdetect, binoculars, logit_classifier
- **国产新模型 AI 英文短篇（DeepSeek / Kimi / 文心一言，用户提供，不参与校准）**（46 AI / 0 人写）：AUROC None；检出率 4.3%
  - 各来源检出率：repo-ai-deepseek 0%，repo-ai-kimi 0%，repo-ai-wenxin 20%
  - 各特征 AUROC：
- **MAGE：GPT-4 在未见过的领域生成的文本**（150 AI / 150 人写）：AUROC 0.9852；检出率 96.0%；误判率 6.0%
  - 各来源检出率：cnn_gpt4 94%，imdb_gpt4 96%，pubmed_gpt4 94%，dialogsum_gpt4 100%
  - 各来源误判率：pubmed_human 6%，dialogsum_human 18%，imdb_human 0%，cnn_human 0%
  - 各特征 AUROC：fastdetect 0.8273，binoculars 0.1768，logit_classifier 0.9828，fastdetect_norm 0.8235，lrr 0.7548，log_rank 0.1979，entropy 0.2756，top10 0.7896，lp_burstiness 0.7117，style_cv 0.3014，style_phrases 0.6956
- **MAGE：GPT-4 文本经改写后（含本仓库英文 AI 样本）**（173 AI / 150 人写）：AUROC 0.8862；检出率 73.4%；误判率 14.0%
  - 各来源检出率：pubmed_gpt4_para 88%，cnn_gpt4_para 58%，imdb_gpt4_para 81%，dialogsum_gpt4_para 74%，repo-ai-english 61%
  - 各来源误判率：pubmed_human 11%，imdb_human 0%，pubmed_human_para 0%，cnn_human 6%，dialogsum_human_para 39%，cnn_human_para 24%，imdb_human_para 24%，dialogsum_human 0%
  - 各特征 AUROC：fastdetect 0.5799，binoculars 0.4234，logit_classifier 0.8832，fastdetect_norm 0.5805，lrr 0.6235，log_rank 0.3531，entropy 0.3639，top10 0.6535，lp_burstiness 0.5053，style_cv 0.3114，style_phrases 0.5934

## 文言
- 数据：NiuTrans 古文语料（人写）+ 大语言模型生成的文言样本
- 校准集 236 人写 / 80 AI；阈值 0.7309；交叉验证 AUROC 0.9427；特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, lp_burstiness, style_cv, style_phrases
- 特征组合比较（校准集交叉验证 AUROC）：全部特征 0.9427，三个主信号 0.9127，语言模型特征 0.9177 → 选用全部特征
- **文言保留集（另一组古籍 + 未参与校准的 AI 文言）**（40 AI / 143 人写）：AUROC 0.967；检出率 72.5%；误判率 2.8%
  - 各来源检出率：llm-classical 72%
  - 各来源误判率：入蜀记 0%，唐传奇 0%，困学纪闻 9%，幽明录 0%，搜神记 0%，新唐书 9%，旧五代史 0%，明夷待访录 0%，武林旧事 0%，聊斋志异 0%，西湖梦寻 18%，资治通鉴 0%，金史 0%，陶庵梦忆 0%
  - 各特征 AUROC：fastdetect 0.9231，binoculars 0.0727，logit_classifier 0.718，fastdetect_norm 0.9208，lrr 0.9413，log_rank 0.0437，entropy 0.1014，top10 0.9411，lp_burstiness 0.5502，style_cv 0.4753，style_phrases 0.5
- **国产新模型 AI 文言故事（DeepSeek / Kimi / 文心一言，用户提供，不参与校准）**（50 AI / 0 人写）：AUROC None；检出率 8.0%
  - 各来源检出率：repo-ai-deepseek 0%，repo-ai-kimi 0%，repo-ai-wenxin 40%
  - 各特征 AUROC：

## 诗词
- 数据：ChangAn 当代旧体诗词（人写）+ DeepSeek / 豆包 / GPT-4.1 生成诗词；诗词专用分类器（ChangAn 训练集微调）
- 校准集 400 人写 / 399 AI；阈值 0.8225；交叉验证 AUROC 0.9755；特征 fastdetect, binoculars, logit_classifier, fastdetect_norm, lrr, log_rank, entropy, top10, style_cv, style_phrases
- 特征组合比较（四种独立对照的 AUROC，按最差情况选）：
  - 诗词分类器 + 语言模型：ChangAn 保留集 0.9545，故事诗 vs 当代人写 0.6749，ChangAn AI vs 唐宋名篇 0.8112，故事诗 vs 唐宋名篇 0.284，国产新模型诗 vs 当代人写 0.8889，国产新模型诗 vs 唐宋名篇 0.6203（最差 0.284）
  - 诗词分类器 + 通用分类器 + 语言模型：ChangAn 保留集 0.9536，故事诗 vs 当代人写 0.6343，ChangAn AI vs 唐宋名篇 0.78，故事诗 vs 唐宋名篇 0.2678，国产新模型诗 vs 当代人写 0.8558，国产新模型诗 vs 唐宋名篇 0.5664（最差 0.2678）
  - 通用分类器 + 语言模型：ChangAn 保留集 0.8769，故事诗 vs 当代人写 0.5923，ChangAn AI vs 唐宋名篇 0.3359，故事诗 vs 唐宋名篇 0.1325，国产新模型诗 vs 当代人写 0.8361，国产新模型诗 vs 唐宋名篇 0.3113（最差 0.1325）
  - 只用语言模型：ChangAn 保留集 0.8379，故事诗 vs 当代人写 0.676，ChangAn AI vs 唐宋名篇 0.2267，故事诗 vs 唐宋名篇 0.1128，国产新模型诗 vs 当代人写 0.9366，国产新模型诗 vs 唐宋名篇 0.3101（最差 0.1128）
- 选用：诗词分类器 + 语言模型；**诗词结果只作参考，不计入 AI 率**
- **国产新模型 AI 诗词（DeepSeek / Kimi / 文心一言，用户提供，不参与校准）**（70 AI / 0 人写）：AUROC None；检出率 38.6%
  - 各来源检出率：repo-ai-deepseek 46%，repo-ai-kimi 25%，repo-ai-wenxin 40%
  - 各特征 AUROC：
- **ChangAn 保留集（另一批作者 + 没见过的 Kimi-K2 与其他模型的新诗词）**（300 AI / 300 人写）：AUROC 0.9545；检出率 73.7%；误判率 2.3%
  - 各来源检出率：Deepseek 61%，gpt-4.1 88%，kimi-k2 71%，seed 75%
  - 各来源误判率：human 2%
  - 各特征 AUROC：fastdetect 0.7596，binoculars 0.2502，logit_classifier 0.9532，fastdetect_norm 0.7519，lrr 0.8175，log_rank 0.1699，entropy 0.2499，top10 0.7724，lp_burstiness 0.5953，style_cv 0.4707，style_phrases 0.5
- **复述故事情节的 AI 诗词（本仓库自带，40 首）**（40 AI / 0 人写）：AUROC None；检出率 0.0%
  - 各来源检出率：repo-ai-story-poem 0%
  - 各特征 AUROC：
- **唐诗三百首 + 宋词三百首（人写名篇，检查误判）**（0 AI / 646 人写）：AUROC None；误判率 18.6%
  - 各来源误判率：唐诗三百首 23%，宋词三百首 12%
  - 各特征 AUROC：
