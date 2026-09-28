# 审读 · 论文 AI 文本检测

这个仓库里有两个网站，改完代码推送到 GitHub 后都会**自动部署**：

| 文件夹 | 是什么 | 部署到哪里 | 怎么自动部署 |
|---|---|---|---|
| 仓库根目录（`worker.js`、`pages-static/` 等） | **网站入口**：把你的域名转发到完整版；`/lite/` 下保留轻量版（浏览器本地统计检测 + 标点排版检查，可选 GPTZero） | Cloudflare Worker `ai-text-checker`（免费） | Cloudflare 自动构建：`main` 分支有新提交就重新部署 |
| `server/` | **完整版检测服务**：按段落识别文体（现代汉语 / 文言 / 英文）后分别判断——Fast-DetectGPT + Binoculars + 分类器（中文 MPU、英文 desklib）+ 扩展特征，三种文体各自校准；自签发 API Key | Modal（按实际运行时间计费，有免费额度） | GitHub Actions：`server/` 有改动时先测试，通过后部署到 Modal |

```
访客 ──→ 你的域名（Cloudflare Worker）──┬─ /lite/ ──→ 轻量版（Worker 自己提供）
                                        └─ 其他 ───→ 转发到 Modal 上的完整版检测服务

你在 GitHub 上改代码 ─┬─→ Cloudflare 自动构建 ─→ Worker
                      └─→ GitHub Actions 测试 → 部署 ─→ Modal
```

Worker 转发的目标地址写在 `wrangler.toml` 的 `BACKEND_URL`。删掉那两行，Worker 就恢复为只提供轻量版。

> 为什么完整版不放 Hugging Face：Hugging Face 自 2026 年 7 月起，免费账号不能再在 CPU 上托管 Docker / Gradio Space，需要 PRO 订阅（$9/月）。如果以后开通了 PRO，也可以用备用工作流部署到 Hugging Face（见文末）。

---

## 一次性设置

### A. Cloudflare（轻量版）：已完成 ✓

Worker `ai-text-checker` 已在 Cloudflare 后台连接本仓库（设置 → 构建 → Git 存储库），`main` 分支有新提交时自动构建并部署。

- 部署记录：Cloudflare 后台 → Workers 和 Pages → `ai-text-checker` → **部署** 标签。
- 如果以后提示"已与 Git 帐户断开连接"：到 <https://github.com/settings/installations> → **Cloudflare Workers and Pages** → **Configure**，确认仓库权限包含 `Ai-text-checker`；必要时在 Cloudflare 里点 **断开连接** 再 **连接**（构建命令留空，部署命令 `npx wrangler deploy`）。

### 绑定自己的域名（在 Cloudflare 后台操作一次）

Cloudflare 后台 → **Workers 和 Pages** → `ai-text-checker` → **设置** → **域和路由** → **添加** → **自定义域** → 填入例如 `ai.wenjinge.dpdns.org`（域名需已托管在这个 Cloudflare 账户里）→ **添加域**。几分钟后即可用这个地址访问完整版，`/admin` 是管理页，`/lite/` 是轻量版。

### B. Modal（完整版）

1. **注册 Modal**：打开 <https://modal.com/signup>，用 GitHub 账号登录即可。
2. **生成令牌**：登录后点左下角或右上角的 **Settings**（设置）→ **API Tokens** → **New Token**，页面会显示两串字符：
   - `Token ID`（以 `ak-` 开头）
   - `Token Secret`（以 `as-` 开头，只显示一次）
3. **存进 GitHub**：打开 <https://github.com/zhuloujun/Ai-text-checker/settings/secrets/actions> → **New repository secret**，添加两个：
   - Name `MODAL_TOKEN_ID`，Secret 填 `ak-…`
   - Name `MODAL_TOKEN_SECRET`，Secret 填 `as-…`
   - 管理员密码沿用已添加的 `HF_ADMIN_TOKEN`，不用再加。
4. **第一次部署**：打开 <https://github.com/zhuloujun/Ai-text-checker/actions> → 左侧 **部署检测服务到 Modal** → **Run workflow**。
   - 首次要构建镜像、下载约 2 GB 模型，约 10–20 分钟。
   - 完成后，在这次运行的页面顶部（Summary）能看到网站地址，形如 `https://<你的 Modal 用户名>--ai-text-checker-web.modal.run`。
5. 打开 `网站地址/admin`，输入 `HF_ADMIN_TOKEN` 的密码，生成 API Key。

**费用**：Modal 只在服务运行时计费（8 核 CPU、12 GB 内存约 $0.47/小时）。没人访问时 10 分钟后自动关闭、不计费；下次打开自动启动（约 1 分钟加载模型）。20 万字完整检测约 $0.1，每月 $30 免费额度对个人使用绰绰有余。用量可在 Modal 的 **Settings → Usage / Billing** 页面查看。

---

## 以后怎么更新

**在网页上改（不需要装任何软件）**：在 GitHub 打开要改的文件 → 点铅笔图标 ✏️ → 修改 → **Commit changes**。几分钟后网站会自动更新。

**看部署结果**：
- 完整版：<https://github.com/zhuloujun/Ai-text-checker/actions> 里的 **部署检测服务到 Modal**。绿色 ✓ 成功；红色 ✗ 点进去看 Summary 里的错误说明。
- 轻量版：Cloudflare 后台 Worker 页面的 **部署** 标签。

**检测效果**：见 [`server/tools/EVAL_REPORT.md`](server/tools/EVAL_REPORT.md)（每种文体都在训练时没见过的数据上测过检出率和误判率）。修改 `server/tools/` 下的文件并推送后会自动重新评估。

**保存校准结果、永久作废 Key**：在 GitHub 仓库 Settings → Secrets and variables → Actions → **Variables** 标签里新建 `CALIBRATION_JSON` 或 `REVOKED_KEY_IDS`，然后在 Actions 页面重新运行一次部署。

---

## 仓库结构

```
worker.js              Cloudflare Worker（由 build.mjs 生成）
worker-template.js     Worker 模板（API 转发逻辑）
pages-static/          轻量版网页源文件
build.mjs              打包脚本（部署时自动运行）
wrangler.toml          Cloudflare 配置（name 必须与 Worker 名一致）
server/                完整版检测服务（说明见 server/README.md）
  modal_app.py         Modal 部署配置
  Dockerfile           Docker / Hugging Face 部署用
.github/workflows/
  deploy-server.yml        测试并部署完整版到 Modal
  deploy-cloudflare.yml    检查轻量版代码（部署由 Cloudflare 自动完成）
  deploy-hf-space.yml      （备用，手动运行）部署到 Hugging Face，需要 PRO
```

### 备用：部署到 Hugging Face

需要 Hugging Face PRO。已设置的 `HF_TOKEN`、`HF_ADMIN_TOKEN` 可直接使用：Actions 页面 → **（备用）部署到 Hugging Face Space** → **Run workflow**。

---

## 附录：Cloudflare 轻量版说明

### GPTZero API Key：在哪里申请？免费吗？

**不免费。** 目前没有既免费又可靠的 AI 检测 API。

| 服务 | 免费额度 | API | 准确率参考 |
|---|---|---|---|
| **GPTZero**（本工具接入） | 网页版每月约 1 万词，需注册 | 需付费 Professional 套餐，约 $25–46/月（按年/按月付费不同，以官网为准） | 独立测试中表现较好，但对格式化程度高的学术文本仍有误判 |
| ZeroGPT | 网页版有限次数 | 预充值、按量计费 | 独立测试约 67%–85% |
| Originality.ai / Copyleaks / Winston AI | 基本没有 | 均付费 | — |

申请步骤：
1. 打开 https://gptzero.me/developers ，注册 / 登录
2. 升级到包含 API 的套餐（网页免费版**不含** API）
3. 在后台的 API 设置里点 **Generate API Key**，复制保存

费用提示：GPTZero 单次请求最多约 5 万字符，20 万字的文档会自动分成 **5 次左右**请求。计费以你套餐的官方说明为准，建议先用一小段文字试一次，确认扣费方式。

配置方式（二选一）：在网页"⚙ 官方 API 增强检测"里直接填写 Key；或在 Cloudflare 的 Worker `ai-text-checker` → **Settings → Variables and Secrets** 添加两个 Secret：`GPTZERO_API_KEY` 和 `ACCESS_PASSWORD`（后者必须设置，防止别人用掉你的额度），然后在网页上填写访问密码。

**关于隐私**：本地检测不上传任何内容；启用 GPTZero 后，文本会发送到 GPTZero 服务器。未发表的论文请自行权衡。

---

### 标点与排版格式检查包括哪些

先说清楚一点：知网 AIGC、GPTZero 这类 AI 检测，是把文档**转成纯文本**后再判断的，字体、字号、表格样式本身不会直接改变 AI 分数。排版之所以和"查重"有关，主要在于：
- 隐藏文字、白色文字、零宽字符、形近字母替换等**规避手法**，检测系统（尤其是文字复制比检测）会专门识别，一旦发现通常会人工复核；
- 格式混乱（多种字体、字号混杂）往往说明文本由多个来源**拼接**，审稿人会留意；
- 修订痕迹、批注、文档属性里的作者名等会**随文件一起提交**。

所以这部分的作用是：提交前自查，把这些问题清理掉。

| 分组 | 检查项 | 纯文本 | .docx | .pdf |
|---|---|:-:|:-:|:-:|
| 规避痕迹 | 零宽 / 不可见字符 | ✓ | ✓ | ✓ |
| | 形近字母混用（拉丁 + 西里尔/希腊） | ✓ | ✓ | ✓ |
| | 全角英文字母 / 数字、汉字之间夹空格 | ✓ | ✓ | ✓ |
| | 隐藏文字、白色文字、≤3 磅极小字 | | ✓ | 极小字、白色填充 |
| | 字符极度压缩、文本框 | | ✓ | |
| | 不可见渲染模式的文字 | | | ✓ |
| 标点规范 | 中文语境中的半角 , ; : ! ? 和英文句点 | ✓ | ✓ | ✓ |
| | 中文内容用半角括号、引号样式混用 | ✓ | ✓ | ✓ |
| | 成对标点不配对（“” 《》 （）等） | ✓ | ✓ | ✓ |
| | 标点重复 / 叠用、不规范省略号和破折号 | ✓ | ✓ | ✓ |
| 空格与段落 | 标点前后多余空格、连续空格、连续空行 | ✓ | ✓ | ✓ |
| | 段首缩进方式、句号样式（。/ ．）不统一 | ✓ | ✓ | ✓ |
| 字体与字号 | 实际使用的字体及次数、样式表字体 | | ✓ | ✓ |
| | 字号种类、文字颜色、高亮底纹 | | ✓ | 字号 |
| 段落排版 | 行距、首行缩进、对齐方式、段落样式 | | ✓ | |
| 表格与对象 | 表格（行数、单元格、嵌套）、图片、公式、脚注尾注、域代码 | | ✓ | 图片 |
| 修订与元数据 | 未处理的修订、批注、作者 / 软件 / 编辑时间 | | ✓ | 元数据 |

每一项点开都有说明，有问题的会列出原文位置。**上传 .docx 检查得最完整**；PDF 无法可靠识别表格结构；扫描版 PDF 提取不到文字，会有提示。

"标点规范"里的"统计"表示数量较少、仅供参考；学术论文里引用外文文献、公式、网址时出现半角标点是正常的。

---

### 已知局限

- 本地"AI 疑似度"是统计方法，准确率明显低于商用神经网络检测器，误判较多。
- **史学、人文社科论文尤其容易被误判**：大量古籍引文、术语、固定学术表述本身就句式规整、用词集中。GPTZero 同样可能误判这类文本。
- 用 AI 辅助翻译或润色过（但内容原创）的文字，统计特征也会偏向"AI 风格"。
- 所有分数只适合自查"哪些段落写得偏套路化"，**不能作为投稿、查重或学术诚信判定的依据**。

---

