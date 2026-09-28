---
title: Ai Text Checker
emoji: 🔍
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: 中文论文 AI 文本检测：Fast-DetectGPT + Binoculars + MPU
---

# 审读 · 中文论文 AI 文本检测（自部署版）

> 代码在 GitHub 仓库 [zhuloujun/Ai-text-checker](https://github.com/zhuloujun/Ai-text-checker) 的 `server/` 文件夹。默认通过 GitHub Actions 自动部署到 **Modal**（见仓库首页 README）；也可以部署到 Hugging Face（需要 PRO）或任何能运行 Docker 的服务器。

部署在你自己的服务器上（默认 Modal），自己签发 API Key，不依赖任何付费检测服务。

## 它怎么判断

| 方法 | 出处 | 原理 | 本项目使用的模型 |
|---|---|---|---|
| **Fast-DetectGPT** | Bao et al., ICLR 2024（[代码](https://github.com/baoguangsheng/fast-detect-gpt)，MIT） | "条件概率曲率"：AI 写作几乎每一步都挑模型认为最可能的字，所以原文的对数概率明显高于"模型随机续写"的期望值。计算 (实际对数概率 − 期望) ÷ 标准差，越高越像 AI。 | Qwen2.5-0.5B（采样）+ Qwen2.5-0.5B-Instruct（打分） |
| **Binoculars** | Hans et al., ICML 2024（[代码](https://github.com/ahans30/Binoculars)） | 用两个模型"双眼"对照：困惑度 ÷ 交叉困惑度。单看困惑度会把"话题本身就少见"的文字误判为人写，除以交叉困惑度可以抵消话题的影响。越低越像 AI。 | 同上两个模型 |
| **MPU 中文分类器** | Tian et al., ICLR 2024（[代码](https://github.com/YuchuanTian/AIGC_text_detector)，Apache-2.0） | 用人写 / AI 写的中文语料专门训练的分类模型（RoBERTa），v3 版覆盖推理类大模型。 | [yuchuantian/AIGC_detector_zhv3](https://huggingface.co/yuchuantian/AIGC_detector_zhv3) |
| **desklib 英文分类器** | desklib（RAID 基准，MIT） | 英文段落专用：DeBERTa-v3-large，发布时位列 RAID 排行榜首位。 | [desklib/ai-text-detector-v1.01](https://huggingface.co/desklib/ai-text-detector-v1.01) |
| **诗词分类器** | 本仓库 `tools/train_poetry.py` | 诗词、对联专用：在 ChangAn（ACL 2026）上微调 hfl/chinese-roberta-wwm-ext；对没见过的作者与模型 AUROC 约 0.95。模型文件在本仓库 Release `poetry-classifier-v1`，部署时自动取用。 | 本仓库 Release |

另有两部分只做参考、不参与 AI 概率：
- **统计特征**：句长变化、字符重复率、套话（"综上所述""值得注意的是"等）。
- **标点与排版检查**（在浏览器里完成）：隐藏文字、白色文字、零宽字符、形近字母、标点规范、字体字号、行距缩进、表格、修订痕迹、文档属性等约 40 项。

**AI 率** = 判定为疑似 AI 的正文字数 ÷ 参与计算的正文字数。"参考文献"之后的内容，以及以引文或文言为主的段落，默认不计入（可在页面上关闭）。

### 实测准确率（内置默认校准）

用公开数据集 [NLPCC 2025 中文 AI 文本检测评测](https://github.com/NLP2CT/NLPCC-2025-Task1) 在 GitHub 上用真实模型评估（`tools/evaluate.py`，报告见 `tools/EVAL_REPORT.md`）。校准集 392 段人写 + 396 段 AI（中文科技论文摘要、新闻、作文；GPT-4o、GLM-4、通义千问）：

| 数据（均未参与拟合） | AI 检出率 | 人写误判率 | AUROC |
|---|---|---|---|
| NLPCC 带标签测试集（含 DeepSeek-V3 生成的文本） | 90.9% | 3.3% | 0.975 |
| CSL 中文科技论文摘要保留集 | 100% | 0% | 1.000 |
| 对照：校准前的经验参数（同一测试集） | 76.6% | 7.1% | 0.941 |

这是按段落（约 200–1800 字）统计的结果。换成古籍引文多的史学论文、或经过人工改写的 AI 文本，效果会下降；用你自己的文字在管理页再校准一次，会更贴合你的文风。


- 三种方法的原始阈值都是在英文或别的模型上定的。**不校准时，结果只宜作相对参考**。管理页有校准功能，见下文。
- 小模型（0.5B）打分比论文里用的 7B 模型弱，这是为了在免费 CPU 上跑得动而做的取舍。
- 任何 AI 检测都会误判，术语密集、表述规范的学术文字尤其容易被误判。结果只供作者自查，不能作为学术不端的判定依据。
- 知网、维普、万方等商业系统的算法不公开，本工具的数字与它们**不可直接换算**。

---

## 部署到 Hugging Face（需要 PRO 订阅）

> Hugging Face 自 2026 年 7 月起，免费账号不能在 CPU 上托管 Docker Space。推荐改用 Modal（见仓库首页 README）。

> 开通 PRO 后，最简单的方法是在 GitHub Actions 页面手动运行 **（备用）部署到 Hugging Face Space**。下面是不经过 GitHub、手动上传的方法。


1. 登录 Hugging Face，打开 <https://huggingface.co/new-space>
   - **Owner** 选 `tdyso`，**Space name** 填 `ai-text-checker`
   - **SDK** 选 **Docker** → **Blank**
   - **Space hardware** 选 **CPU basic · 2 vCPU · 16 GB · FREE**
   - **Public**（公开；接口仍需 API Key 才能用）→ **Create Space**
2. 在新 Space 页面点 **Files** → **Add file** → **Upload files**，把本压缩包解压后的**全部文件和文件夹**拖进去（`README.md`、`Dockerfile`、`app/`、`static/` 等，保持目录结构），点 **Commit changes to main**。
   - 注意：这个 `README.md` 开头的 `---` 配置块必须保留，Space 靠它知道用 Docker、端口 7860。
3. 点 **Settings** → **Variables and secrets** → **New secret**：
   - 名称 `ADMIN_TOKEN`，值是你自己定的管理员密码（长一点，例如 20 位以上随机字符）。
4. 回到 **App** 标签，等待构建（首次约 5–15 分钟，要下载约 2 GB 模型）。状态从 *Building* 变为 *Running* 后即可使用。
5. 打开 `https://tdyso-ai-text-checker.hf.space/admin`，输入管理员密码，**生成 API Key**，复制保存。
6. 打开 `https://tdyso-ai-text-checker.hf.space/`，填入 Key，上传论文检测。

用命令行部署（可选）：
```bash
git clone https://huggingface.co/spaces/tdyso/ai-text-checker
# 把本项目文件复制进去
cd ai-text-checker && git add . && git commit -m "init" && git push   # 密码处填 HF 的 Access Token（write 权限）
```

### 免费 Space 需要知道的
- **会休眠**：一段时间没人访问会自动休眠，下次打开时自动唤醒，需要等 1–3 分钟加载模型。
- **磁盘不持久**：重启后内存里的用量统计、临时作废、页面上启用的校准都会清空。所以：
  - Key 设计成"自带签名"，重启后照常有效；
  - 永久作废 Key → 变量 `REVOKED_KEY_IDS`；永久保存校准 → 变量 `CALIBRATION_JSON`。
- **速度**（2 vCPU，按 Qwen2.5-0.5B 实测）：完整检测约每 400 字 3.3 秒，20 万字约 25–30 分钟；快速检测（语言模型抽样 60 段，分类器全文）约 3–5 分钟。检测在后台进行，关掉网页不影响，结果保留 1 小时。需要更快可在 Settings 里升级付费 CPU，并把 `TORCH_THREADS` 设为对应核数。

---

## API Key

- 在 `/admin` 页面签发。可以设名称、有效天数、每日字数额度。
- 签名密钥来自 `ADMIN_TOKEN`（或单独设置的 `KEY_SECRET`）。**修改它会让之前所有 Key 失效**，相当于一键全部作废。
- 作废单个 Key：管理页点"作废"立即生效；再把编号加进变量 `REVOKED_KEY_IDS`（逗号分隔），重启后也保持作废。

### 接口

完整文档见 `/docs`（自动生成，可以直接在网页上试调用）。

```bash
# 短文本：直接返回结果
curl -X POST https://tdyso-ai-text-checker.hf.space/v1/detect \
  -H "Authorization: Bearer atc-你的Key" -H "Content-Type: application/json" \
  -d '{"text": "要检测的文字……", "mode": "full"}'

# 长文档：返回任务编号（HTTP 202），再轮询
curl -X POST https://tdyso-ai-text-checker.hf.space/v1/detect/file \
  -H "Authorization: Bearer atc-你的Key" -F "file=@论文.docx" -F "mode=fast"
curl https://tdyso-ai-text-checker.hf.space/v1/jobs/<任务编号> -H "Authorization: Bearer atc-你的Key"
```

Python 示例：`examples/client.py`（只用标准库）。

返回结果的主要字段：
- `result.summary.ai_rate`：AI 率（0–1）；`mean_prob`：按字数加权的平均 AI 概率；`threshold`：判定阈值；`calibrated`：是否已校准
- `result.segments[]`：每段的 `prob`（综合 AI 概率）、`level`（high / mid / low / none）、`kind`（body 正文 / reference 参考文献 / quotation 引文为主）、`signals`（三种方法各自换算的概率）、`raw`（原始分数）、`style`（统计特征）

---

## 校准（强烈建议做一次）

1. 准备两组文字：
   - **人写**：确定是你自己写的，最好是 2022 年以前、没用过 AI 的旧稿，文风越接近要检测的论文越好（引文多的也放进来）。
   - **AI 写**：同类题目、让几个不同的 AI 写的段落。
   - 每组建议 1 万字以上（约 30 段以上）。
2. 管理页 → 校准：分别粘贴（多篇之间用单独一行 `===` 隔开）或选择多个 .txt / .docx 文件。
3. 选择"人写文字的误判率上限"（默认 5%），开始校准。完成后会显示区分能力（AUROC）、误判率、检出率。
4. 点 **立即启用**；再把显示的 JSON 保存到 Space 变量 `CALIBRATION_JSON`，以免重启后丢失。

校准做的事：为每种方法重新定中心、尺度和方向，用逻辑回归学习三种方法的组合权重，然后按你设定的误判率上限选阈值。分类器的标签方向如果被自动判断反了，校准也会自动纠正。

---

## 可调的环境变量（Settings → Variables and secrets）

| 名称 | 默认 | 说明 |
|---|---|---|
| `ADMIN_TOKEN` | — | **必填（Secret）**。管理页密码，也用来派生 Key 的签名密钥 |
| `KEY_SECRET` | 由 ADMIN_TOKEN 派生 | 可选（Secret）。单独的签名密钥，设置后改管理员密码不会让 Key 失效 |
| `REVOKED_KEY_IDS` | 空 | 永久作废的 Key 编号，逗号分隔 |
| `CALIBRATION_JSON` | 空 | 校准结果 |
| `REQUIRE_KEY` | `true` | 设为 `false` 则网页和接口都不需要 Key（公开 Space 不建议） |
| `OBSERVER_MODEL` / `PERFORMER_MODEL` | Qwen2.5-0.5B / -Instruct | 必须是同一系列的基础版 + 对话版（共用分词器）。硬件够时可换 `Qwen/Qwen2.5-1.5B` / `-Instruct`，更准但慢约 3 倍 |
| `CLASSIFIER_MODEL` | yuchuantian/AIGC_detector_zhv3 | 可换成 `..._zhv2`；留空则不用分类器 |
| `CLASSIFIER_AI_LABEL` | auto | 分类器中表示 AI 的标签（名称或下标）。自动判断不了时取下标 1；如结果明显反了，改成 `0` |
| `FAST_MODE_MAX_SEGMENTS` | 60 | 快速模式语言模型检测的段数 |
| `CLASSICAL_THRESHOLD` | 0.07 | 文言虚词比例超过此值的段落视为古籍引文，不计入 |
| `DEFAULT_DAILY_CHARS` | 1000000 | 新 Key 的默认每日字数额度 |
| `TORCH_THREADS` | CPU 核数 | 推理线程数 |

换模型需要重新构建镜像：在 Settings 里点 **Factory reboot**。

---

## 目录

```
README.md            Space 配置 + 说明（本文件）
Dockerfile           镜像：CPU 版 PyTorch，构建时预下载模型
download_models.py   构建时下载模型
requirements.txt
app/
  main.py            接口（FastAPI）
  engine.py          检测流程、后台任务队列
  segmenter.py       分段；识别参考文献、引文、文言段落
  scoring.py         信号换算为概率；校准（逻辑回归 + 按误判率选阈值）
  keys.py            自签名 API Key、每日额度
  docparse.py        服务端解析 .docx / .pdf / .txt（API 上传用）
  config.py          环境变量
  detectors/
    lm_scorer.py     Fast-DetectGPT 与 Binoculars（共用一次推理）
    classifier.py    分类器（MPU 中文、desklib 英文、诗词专用）
    stylometry.py    统计特征
static/              网页（检测页、管理页；标点排版检查在浏览器端）
examples/client.py   API 调用示例
tests/               测试（用随机小模型检查流程，并与官方公式逐项比对）
```

测试：
```bash
pip install -r requirements.txt torch pytest httpx
python tests/make_tiny_models.py /tmp/tiny && TEST_MODELS_DIR=/tmp/tiny pytest -q
```

## 许可

本项目代码 MIT。使用的模型各有许可：Qwen2.5-0.5B / -Instruct（Apache-2.0）、AIGC_detector_zhv3（见其模型页）。Fast-DetectGPT、Binoculars 的公式按其官方实现重新编写，测试中与官方代码逐项比对。
