# 审读 · 论文 AI 文本检测

这个仓库里有两个互相独立的网站，改完代码推送到 GitHub 后都会**自动部署**：

| 文件夹 | 是什么 | 部署到哪里 | 怎么自动部署 |
|---|---|---|---|
| 仓库根目录（`worker.js`、`pages-static/` 等） | 轻量版：浏览器本地统计检测 + 标点排版检查，可选接入 GPTZero | Cloudflare Worker `ai-text-checker` | Cloudflare 自动构建：`main` 分支有新提交就重新部署 |
| `hf-space/` | 完整版：Fast-DetectGPT + Binoculars + MPU 中文分类器，自签发 API Key，可校准 | Hugging Face Space `tdyso/ai-text-checker` | GitHub Actions：`hf-space/` 有改动时先测试，通过后同步到 Space |

```
你在 GitHub 上改代码 ─┬─→ Cloudflare 自动构建 ──→ Cloudflare 网站
                      └─→ GitHub Actions 测试 ──→ 同步到 Hugging Face ──→ Space 自动重建
```

---

## 一次性设置（只需做一次）

### A. Hugging Face：让 GitHub 有权限更新你的 Space

1. **生成 Hugging Face 令牌**
   打开 <https://huggingface.co/settings/tokens> → **Create new token** → 类型选 **Write** → 名称随便填（如 `github-deploy`）→ **Create token**，复制保存（以 `hf_` 开头，只显示一次）。
2. **把令牌存进 GitHub**
   打开 <https://github.com/zhuloujun/Ai-text-checker/settings/secrets/actions> → **New repository secret**：
   - Name：`HF_TOKEN`，Secret：刚才复制的令牌 → **Add secret**
3. **设置 Space 管理员密码**（用来登录 `/admin` 签发 API Key）
   同一页面再点 **New repository secret**：
   - Name：`HF_ADMIN_TOKEN`，Secret：你自己定的密码（建议 20 位以上）→ **Add secret**
4. **第一次部署**
   打开 <https://github.com/zhuloujun/Ai-text-checker/actions> → 左侧选 **部署到 Hugging Face Space** → 右侧 **Run workflow** → **Run workflow**。
   - 约 5 分钟后两个步骤都变成绿色 ✓。Space 会被自动创建在 `tdyso/ai-text-checker`。
   - 然后 Hugging Face 开始构建镜像（首次约 5–15 分钟，要下载约 2 GB 模型）。可在 <https://huggingface.co/spaces/tdyso/ai-text-checker> 看进度，显示 *Running* 即完成。
5. 打开 <https://tdyso-ai-text-checker.hf.space/admin>，输入第 3 步的密码，生成 API Key。

> 想换 Space 名称：在 GitHub 仓库 Settings → Secrets and variables → Actions → **Variables** 标签页新建变量 `HF_SPACE`，值如 `tdyso/另一个名字`。

### B. Cloudflare：已完成 ✓

Worker `ai-text-checker` 已在 Cloudflare 后台连接本仓库（设置 → 构建 → Git 存储库），`main` 分支有新提交时 Cloudflare 会自动构建并部署，**不需要**另外设置令牌。

- 部署记录：Cloudflare 后台 → Workers 和 Pages → `ai-text-checker` → **部署** 标签。
- 可选：在 设置 → 构建 → **构建监视路径** 的"排除路径"里填 `hf-space/**`，只改 Hugging Face 部分时 Cloudflare 就不会重复部署。
- 如果以后页面提示"已与 Git 帐户断开连接"：点 **断开连接**，再点 **连接**，重新选择 `zhuloujun` / `Ai-text-checker` / `main`，构建命令留空，部署命令 `npx wrangler deploy`。

> 仓库里的"部署到 Cloudflare"GitHub Actions 工作流只做代码检查；没有设置 `CLOUDFLARE_API_TOKEN` 时会自动跳过部署（黄色提示，不是错误），不影响 Cloudflare 自己的自动部署。

---

## 以后怎么更新

**在网页上改（不需要装任何软件）**：在 GitHub 打开要改的文件 → 点右上角铅笔图标 ✏️ → 修改 → **Commit changes**。几分钟后两个网站会自动更新。

**看部署是否成功**：
- Hugging Face：<https://github.com/zhuloujun/Ai-text-checker/actions> 里的 **部署到 Hugging Face Space**，绿色 ✓ 表示成功，红色 ✗ 点进去看哪一步出错（最常见：令牌过期 → 重新生成并更新 `HF_TOKEN`）。
- Cloudflare：Cloudflare 后台 Worker 页面的 **部署** 标签。

**改网页界面时注意**：Cloudflare 版的网页源文件在 `pages-static/`，部署时会自动打包成 `worker.js`，不用手动运行 `build.mjs`。

---

## 仓库结构

```
worker.js              Cloudflare Worker（由 build.mjs 生成，已提交一份便于在后台直接粘贴）
worker-template.js     Worker 模板（API 转发逻辑）
pages-static/          Cloudflare 版网页源文件
build.mjs              打包脚本（部署时自动运行）
wrangler.toml          Cloudflare 配置（name 必须与 Worker 名一致；含账户 ID）
hf-space/              Hugging Face 完整版（说明见 hf-space/README.md）
.github/workflows/     自动测试与部署
.github/scripts/       同步到 Hugging Face 的脚本
```

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

