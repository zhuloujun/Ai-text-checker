/* ============================================================
   审读 · 文本特征检测
   全部计算在浏览器本地完成，不向任何服务器发送文本内容。
   ============================================================ */

/* ---------------- DOM refs ---------------- */
const pasteArea   = document.getElementById('pasteArea');
const dropzone     = document.getElementById('dropzone');
const fileInput    = document.getElementById('fileInput');
const fileInfo      = document.getElementById('fileInfo');
const fileName      = document.getElementById('fileName');
const fileMeta       = document.getElementById('fileMeta');
const fileClear      = document.getElementById('fileClear');
const charCount      = document.getElementById('charCount');
const analyzeBtn     = document.getElementById('analyzeBtn');
const progressZone   = document.getElementById('progressZone');
const progressFill   = document.getElementById('progressFill');
const progressLabel  = document.getElementById('progressLabel');
const resultsZone    = document.getElementById('resultsZone');
const paragraphList  = document.getElementById('paragraphList');
const sealNum        = document.getElementById('sealNum');
const exportBtn      = document.getElementById('exportBtn');

const settingsToggle = document.getElementById('settingsToggle');
const settingsCaret  = document.getElementById('settingsCaret');
const settingsPanel  = document.getElementById('settingsPanel');
const useApiToggle   = document.getElementById('useApiToggle');
const apiStatusMsg   = document.getElementById('apiStatusMsg');
const apiSection     = document.getElementById('apiSection');
const apiResultBody  = document.getElementById('apiResultBody');
const formatGrid     = document.getElementById('formatGrid');
const formatDetail   = document.getElementById('formatDetail');

let currentText = '';       // 当前待分析文本（粘贴或从文件解析出的）
let currentSource = '粘贴文本';
let currentFileExt = null;  // 'docx' | 'pdf' | 'txt' | null（粘贴文本时为 null）
let currentArrayBuffer = null; // 原始文件字节（.docx 格式分析需要）
let currentPdfDoc = null;   // pdf.js 文档对象（.pdf 字体分析需要）
let lastReport = null;      // 最近一次分析结果，供导出使用
let lastFormatItems = null; // 最近一次标点与排版检查结果
let lastApiSummary = null;  // 最近一次官方 API 结果

/* ---------------- 设置面板折叠 ---------------- */
settingsToggle.addEventListener('click', ()=>{
  settingsPanel.classList.toggle('hidden');
  settingsCaret.classList.toggle('open');
});
const apiKeyInput   = document.getElementById('apiKeyInput');
const accessPwInput = document.getElementById('accessPwInput');
const rememberKey   = document.getElementById('rememberKey');
const keyShowBtn    = document.getElementById('keyShowBtn');
const STORE_KEY = 'shendu_api_settings';

// 读取已记住的 Key（浏览器存储可能被禁用，全部 try/catch）
try{
  const saved = JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
  if(saved){
    apiKeyInput.value = saved.key || '';
    accessPwInput.value = saved.pw || '';
    rememberKey.checked = true;
  }
}catch(e){}

function persistKey(){
  try{
    if(rememberKey.checked){
      localStorage.setItem(STORE_KEY, JSON.stringify({ key: apiKeyInput.value.trim(), pw: accessPwInput.value }));
    } else {
      localStorage.removeItem(STORE_KEY);
    }
  }catch(e){}
}
[apiKeyInput, accessPwInput].forEach(el=>el.addEventListener('input', persistKey));
rememberKey.addEventListener('change', persistKey);

keyShowBtn.addEventListener('click', ()=>{
  const show = apiKeyInput.type === 'password';
  apiKeyInput.type = show ? 'text' : 'password';
  keyShowBtn.textContent = show ? '隐藏' : '显示';
});

useApiToggle.addEventListener('change', ()=>{
  apiStatusMsg.textContent = useApiToggle.checked
    ? '提示：需要部署成 Cloudflare Worker 后才能调用；直接双击打开本地 html 文件时无法使用此功能。'
    : '';
});

/* ---------------- Tab switching ---------------- */
document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-paste').classList.toggle('hidden', btn.dataset.tab!=='paste');
    document.getElementById('panel-upload').classList.toggle('hidden', btn.dataset.tab!=='upload');
    syncAnalyzeState();
  });
});

/* ---------------- Paste input ---------------- */
pasteArea.addEventListener('input', ()=>{
  currentText = pasteArea.value;
  currentSource = '粘贴文本';
  currentFileExt = null;
  currentArrayBuffer = null;
  currentPdfDoc = null;
  updateCharCount();
  syncAnalyzeState();
});

/* ---------------- File input / drag & drop ---------------- */
dropzone.addEventListener('click', (e)=>{ if(e.target.tagName!=='LABEL') fileInput.click(); });
dropzone.addEventListener('dragover', e=>{ e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', ()=> dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e=>{
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if(e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e=>{
  if(e.target.files.length) handleFile(e.target.files[0]);
});
fileClear.addEventListener('click', ()=>{
  currentText = '';
  currentFileExt = null;
  currentArrayBuffer = null;
  currentPdfDoc = null;
  fileInfo.classList.add('hidden');
  fileInput.value = '';
  updateCharCount();
  syncAnalyzeState();
});

async function handleFile(file){
  const ext = file.name.split('.').pop().toLowerCase();
  fileName.textContent = file.name;
  fileMeta.textContent = `解析中… (${(file.size/1024).toFixed(0)} KB)`;
  fileInfo.classList.remove('hidden');
  analyzeBtn.disabled = true;
  currentArrayBuffer = null;
  currentPdfDoc = null;
  currentFileExt = ext;

  try{
    let text = '';
    if(ext === 'txt'){
      text = await file.text();
    } else if(ext === 'docx'){
      const buf = await file.arrayBuffer();
      currentArrayBuffer = buf; // 供后续字体/隐藏文字/表格分析使用
      const result = await mammoth.extractRawText({ arrayBuffer: buf });
      text = result.value;
    } else if(ext === 'pdf'){
      const buf = await file.arrayBuffer();
      currentArrayBuffer = buf;
      text = await extractPdfText(buf);
    } else {
      throw new Error('不支持的文件格式');
    }
    currentText = text;
    currentSource = file.name;
    fileMeta.textContent = `${text.length.toLocaleString()} 字 · 解析完成`;
    if(ext === 'pdf' && currentPdfDoc && text.replace(/\s/g,'').length < currentPdfDoc.numPages * 20){
      fileMeta.textContent += ' · ⚠ 提取到的文字很少，这个 PDF 可能是扫描件或图片，无法做文字检测；请改传 .docx 版本';
    }
    updateCharCount();
  }catch(err){
    fileMeta.textContent = '解析失败：' + err.message;
    currentText = '';
  }
  syncAnalyzeState();
}

async function extractPdfText(buf){
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  // cMap 是 pdf.js 解码中文等 CJK 字体所需的字符映射表；不加载时，
  // 使用内置中文字体（很多知网下载的 PDF 属于这种）的文件会提取不出汉字。
  const pdf = await pdfjsLib.getDocument({
    data: buf.slice(0),
    cMapUrl: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/cmaps/',
    cMapPacked: true,
    standardFontDataUrl: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/standard_fonts/'
  }).promise;
  currentPdfDoc = pdf;
  let text = '';
  for(let i=1;i<=pdf.numPages;i++){
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    text += content.items.map(it=>it.str).join('') + '\n';
    progressLabel.textContent = `正在解析 PDF … 第 ${i} / ${pdf.numPages} 页`;
  }
  return text;
}

function updateCharCount(){
  charCount.textContent = `${currentText.length.toLocaleString()} 字`;
}
function syncAnalyzeState(){
  analyzeBtn.disabled = currentText.trim().length < 30;
}

/* ---------------- Analyze button ---------------- */
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis(){
  // 如果当前激活的是"粘贴文本"标签，优先使用文本框内容
  const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
  if(activeTab === 'paste') {
    currentText = pasteArea.value;
    currentSource = '粘贴文本';
    currentFileExt = null;
    currentArrayBuffer = null;
    currentPdfDoc = null;
  }

  const text = currentText.trim();
  if(text.length < 30) return;

  analyzeBtn.disabled = true;
  resultsZone.classList.add('hidden');
  apiSection.classList.add('hidden');
  progressZone.classList.remove('hidden');
  progressFill.style.width = '0%';
  progressLabel.textContent = '正在切分段落…';

  await sleep(10); // 让出主线程以更新界面

  const chunks = splitIntoChunks(text);
  const results = [];
  const BATCH = 25;

  for(let i=0;i<chunks.length;i+=BATCH){
    const slice = chunks.slice(i, i+BATCH);
    slice.forEach(c => results.push(scoreChunk(c)));
    const pct = Math.min(100, Math.round(((i+BATCH)/chunks.length)*100));
    progressFill.style.width = pct + '%';
    progressLabel.textContent = `正在分析段落 ${Math.min(i+BATCH, chunks.length)} / ${chunks.length} …`;
    await sleep(0); // 让出主线程，避免长文档卡死页面
  }

  renderResults(results, text.length);

  // 标点与格式特征检查（本地，与 AI 疑似度无关）
  progressLabel.textContent = '正在检查标点与格式特征…';
  const formatReport = await buildFormatReport(text);
  renderFormatReport(formatReport);

  // 可选：官方 API 增强检测
  lastApiSummary = null;
  if(useApiToggle.checked){
    await runApiDetection(text);
  } else {
    apiSection.classList.add('hidden');
  }

  progressZone.classList.add('hidden');
  resultsZone.classList.remove('hidden');
  analyzeBtn.disabled = false;
}

function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }

/* ============================================================
   文本切分：按段落切分，过长段落再按句子分组，
   过短段落（如标题、单行引文）合并到下一段，避免噪声评分。
   ============================================================ */
function splitIntoChunks(text){
  const rawParas = text.split(/\n+/).map(p=>p.trim()).filter(Boolean);
  const chunks = [];
  let buffer = '';

  for(const p of rawParas){
    buffer = buffer ? buffer + '\n' + p : p;
    if(buffer.length >= 120){
      // 超长段落按句子分组，每组约 300~500 字
      if(buffer.length > 600){
        chunks.push(...splitLongParagraph(buffer));
      } else {
        chunks.push(buffer);
      }
      buffer = '';
    }
  }
  if(buffer.length >= 30) {
    if(buffer.length > 600) chunks.push(...splitLongParagraph(buffer));
    else chunks.push(buffer);
  }
  return chunks;
}

function splitLongParagraph(text){
  const sentences = text.split(/(?<=[。！？!?])/).filter(s=>s.trim());
  const groups = [];
  let cur = '';
  for(const s of sentences){
    cur += s;
    if(cur.length >= 400){ groups.push(cur); cur=''; }
  }
  if(cur.trim()) groups.push(cur);
  return groups;
}

/* ============================================================
   启发式评分核心
   四个统计特征，加权合成 0-100 的"疑似 AI 特征强度"分数。
   这不是概率，也不是分类模型输出，只是风格统计信号的聚合。
   ============================================================ */

// 学术 / 通用中文写作中常见的 AI 高频过渡词与套话
const AI_PHRASES = [
  '首先','其次','再次','此外','另外','值得注意的是','值得强调的是',
  '综上所述','总的来说','总而言之','由此可见','不仅如此','不可否认',
  '毋庸置疑','需要指出的是','与此同时','事实上','换言之','众所周知',
  '在这种背景下','从某种意义上说','不难发现','可以看出','综合来看',
  '一方面','另一方面','显而易见','总体而言','在很大程度上'
];

function scoreChunk(text){
  const sentences = text.split(/(?<=[。！？!?])/).map(s=>s.trim()).filter(s=>s.length>1);
  const len = text.length;

  // 1) 句长突发性 burstiness：人类写作句长波动大，AI 生成往往趋于均匀
  const lens = sentences.map(s=>s.length);
  const mean = lens.reduce((a,b)=>a+b,0) / (lens.length||1);
  const variance = lens.reduce((a,b)=>a+(b-mean)*(b-mean),0) / (lens.length||1);
  const std = Math.sqrt(variance);
  const cv = mean>0 ? std/mean : 0; // 变异系数，越低越"整齐"
  const burstinessScore = clamp(100 - cv*130, 0, 100);

  // 2) 字符 bigram 重复率：AI 生成文本局部重复模式偏多
  const bigrams = [];
  for(let i=0;i<text.length-1;i++) bigrams.push(text.substring(i,i+2));
  const uniqueRatio = bigrams.length ? new Set(bigrams).size / bigrams.length : 1;
  const repetitionScore = clamp((0.72 - uniqueRatio) * 260, 0, 100);

  // 3) AI 高频过渡词密度
  let phraseHits = 0;
  for(const p of AI_PHRASES){
    if(text.includes(p)) phraseHits++;
  }
  const phraseDensity = (phraseHits / Math.max(len,1)) * 1000; // 每千字命中数
  const phraseScore = clamp(phraseDensity * 22, 0, 100);

  // 4) 平均句长集中度：句子普遍落在 18-42 字区间且方差小，是常见 AI 特征
  const midRangeRatio = lens.length ? lens.filter(l=>l>=18 && l<=42).length / lens.length : 0;
  const rangeScore = clamp((midRangeRatio - 0.4) * 140, 0, 100);

  const final = clamp(
    burstinessScore*0.35 + repetitionScore*0.25 + phraseScore*0.25 + rangeScore*0.15,
    0, 100
  );

  const tags = [];
  if(burstinessScore>60) tags.push('句长均匀');
  if(repetitionScore>55) tags.push('局部重复');
  if(phraseHits>0) tags.push(`套语×${phraseHits}`);
  if(rangeScore>55) tags.push('句长集中');

  return { text, score: Math.round(final), len, tags };
}

function clamp(v, min, max){ return Math.max(min, Math.min(max, v)); }

/* ---------------- 渲染结果 ---------------- */
function renderResults(results, totalLen){
  lastReport = { results, totalLen, source: currentSource, time: new Date() };

  const weightedSum = results.reduce((a,r)=>a + r.score*r.len, 0);
  const overall = totalLen ? Math.round(weightedSum/totalLen) : 0;

  const high = results.filter(r=>r.score>=60).length;
  const mid  = results.filter(r=>r.score>=30 && r.score<60).length;

  sealNum.textContent = overall + '%';
  document.getElementById('sumOverall').textContent = overall + '%';
  document.getElementById('sumHigh').textContent = high;
  document.getElementById('sumMid').textContent = mid;
  document.getElementById('sumTotal').textContent = results.length;

  paragraphList.innerHTML = '';
  results.forEach((r, idx)=>{
    const level = r.score>=60 ? 'high' : (r.score>=30 ? 'mid' : 'low');
    const div = document.createElement('div');
    div.className = `para-item level-${level}`;
    const preview = r.text.length > 260 ? r.text.slice(0,260) + '……' : r.text;
    div.innerHTML = `
      <div class="para-badge">${r.score}</div>
      <div class="para-body">
        <div class="para-text">${escapeHtml(preview)}</div>
        <div class="para-tags">
          ${r.tags.map(t=>`<span class="para-tag">${t}</span>`).join('') || `<span class="para-tag">—</span>`}
          <span class="para-tag">第 ${idx+1} 段 · ${r.len} 字</span>
        </div>
      </div>`;
    paragraphList.appendChild(div);
  });
}

function escapeHtml(s){
  return s.replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/* ============================================================
   标点符号与排版格式检查
   这部分和"是否 AI 生成"是两回事：它查的是标点是否规范、
   排版是否统一，以及隐藏文字、零宽字符、形近字母替换等
   常被检测系统视为"规避检测"的痕迹。全部在浏览器本地计算。
   .docx 通过 JSZip 直接读取文档内部 XML；
   .pdf 通过 pdf.js 读取字体、字号与绘制指令。
   ============================================================ */

// 严重程度：danger = 可能被视为规避检测；warn = 不规范/不统一；info = 仅统计
const SEV_LABEL = { danger:'需重点核查', warn:'不规范', info:'统计', ok:'正常' };

function ctx(text, index, len, radius){
  radius = radius || 10;
  const before = text.slice(Math.max(0, index-radius), index);
  const hit = text.slice(index, index+len);
  const after = text.slice(index+len, index+len+radius);
  return { before, hit, after };
}

// 统一的"按正则计数 + 取样"工具
function scan(text, re, opts){
  opts = opts || {};
  const samples = [];
  let count = 0;
  re.lastIndex = 0;
  let m;
  while((m = re.exec(text))){
    count++;
    if(samples.length < (opts.maxSamples || 6)){
      const off = opts.offset ? opts.offset(m) : 0;
      const len = opts.len ? opts.len(m) : m[0].length - off;
      samples.push(ctx(text, m.index + off, Math.max(1,len)));
    }
    if(m[0].length === 0) re.lastIndex++;
  }
  return { count, samples };
}

function countOf(text, re){
  const m = text.match(re);
  return m ? m.length : 0;
}

const CJK = '\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff';

function analyzePunctuation(text){
  const items = [];
  const add = (o)=>items.push(o);

  /* ---------- 1. 规避检测类痕迹 ---------- */

  const invisible = scan(text, /[\u200B\u200C\u200D\u2060\uFEFF\u00AD\u180E\u2061-\u2064]/g);
  add({ group:'规避痕迹', name:'零宽 / 不可见字符', count: invisible.count, samples: invisible.samples,
        sev: invisible.count>0 ? 'danger':'ok',
        desc:'零宽空格、BOM、软连字符等肉眼看不见的字符。插在词语中间会打断文本比对，是常见的降重手法。正常写作几乎不会出现，从网页复制时偶尔会带进来。' });

  // 在拉丁单词里混入西里尔 / 希腊字母（如用西里尔 а 冒充拉丁 a）
  const homoglyphMixed = { count:0, samples:[] };
  {
    const re = /[A-Za-zͰ-ϿЀ-ӿ]+/g; let m;
    while((m = re.exec(text))){
      const w = m[0];
      if(/[A-Za-z]/.test(w) && /[Ͱ-ϿЀ-ӿ]/.test(w)){
        homoglyphMixed.count++;
        if(homoglyphMixed.samples.length<6) homoglyphMixed.samples.push(ctx(text, m.index, w.length));
      }
    }
  }
  add({ group:'规避痕迹', name:'形近字母混用（拉丁 + 西里尔/希腊）', count: homoglyphMixed.count, samples: homoglyphMixed.samples,
        sev: homoglyphMixed.count>0 ? 'danger':'ok',
        desc:'同一个英文单词里混有外形相同的西里尔或希腊字母，看起来一样，但比对系统会当成不同的字。' });

  const fwLatin = scan(text, /[０-９Ａ-Ｚａ-ｚ]+/g);
  add({ group:'规避痕迹', name:'全角英文字母 / 数字', count: fwLatin.count, samples: fwLatin.samples,
        sev: fwLatin.count>20 ? 'warn' : (fwLatin.count>0 ? 'info':'ok'),
        desc:'如"２０２４""ＡＩ"。中文排版中偶尔会用，大量出现时会影响比对，也不符合一般的论文格式要求（数字和字母通常用半角）。' });

  const cjkSpaced = scan(text, new RegExp(`[${CJK}][ \\t]+(?=[${CJK}])`, 'g'), { len: m=>m[0].length-1, offset: ()=>1 });
  add({ group:'规避痕迹', name:'汉字之间夹空格', count: cjkSpaced.count, samples: cjkSpaced.samples,
        sev: cjkSpaced.count>50 ? 'warn' : (cjkSpaced.count>0 ? 'info':'ok'),
        desc:'汉字与汉字之间出现空格。从 PDF 复制过来的文字常有这个现象；刻意插入则会干扰比对。少量出现通常是排版遗留。' });

  /* ---------- 2. 中英文标点混用 ---------- */

  const hwComma = scan(text, new RegExp(`(?<=[${CJK}])[,;:!?](?!\\d)|[,;:!?](?=[${CJK}])`, 'g'));
  add({ group:'标点规范', name:'中文语境中的半角标点 , ; : ! ?', count: hwComma.count, samples: hwComma.samples,
        sev: hwComma.count>10 ? 'warn' : (hwComma.count>0 ? 'info':'ok'),
        desc:'紧挨着汉字使用了英文逗号、分号、冒号、叹号或问号。中文正文应使用全角标点（，；：！？）。' });

  const hwPeriod = scan(text, new RegExp(`(?<=[${CJK}])\\.(?![\\d.])`, 'g'));
  add({ group:'标点规范', name:'中文句末使用英文句点 .', count: hwPeriod.count, samples: hwPeriod.samples,
        sev: hwPeriod.count>5 ? 'warn' : (hwPeriod.count>0 ? 'info':'ok'),
        desc:'汉字后面直接跟英文句点。中文句末应使用"。"。' });

  const hwParen = scan(text, new RegExp(`\\([^()\\n]*[${CJK}][^()\\n]*\\)`, 'g'));
  add({ group:'标点规范', name:'中文内容使用半角括号 ( )', count: hwParen.count, samples: hwParen.samples,
        sev: hwParen.count>10 ? 'warn' : (hwParen.count>0 ? 'info':'ok'),
        desc:'括号里含中文却使用了英文括号。中文正文一般用全角括号（ ）。' });

  const straightQ = countOf(text, /"/g);
  const curlyQ = countOf(text, /[“”]/g);
  const cornerQ = countOf(text, /[「」『』]/g);
  const quoteStyles = [straightQ, curlyQ, cornerQ].filter(n=>n>0).length;
  const sq = scan(text, /"/g, { maxSamples:4 });
  add({ group:'标点规范', name:'引号样式混用', count: quoteStyles>1 ? (straightQ + curlyQ + cornerQ) : 0,
        samples: quoteStyles>1 ? sq.samples : [],
        sev: quoteStyles>1 ? 'warn' : 'ok',
        extra: `直引号 " ${straightQ} 个 · 弯引号 “ ” ${curlyQ} 个 · 直角引号「」『』${cornerQ} 个`,
        desc:'同一篇文章里混用了多种引号。大陆规范通常用 “ ”，港台及部分古籍整理用「」，全文应保持一种。' });

  /* ---------- 3. 成对标点、重复、省略号、破折号 ---------- */

  const pairs = [
    ['“','”','双引号'], ['‘','’','单引号'], ['（','）','全角括号'], ['(',')','半角括号'],
    ['《','》','书名号'], ['〈','〉','单书名号'], ['【','】','方头括号'], ['「','」','直角引号'], ['『','』','双直角引号']
  ];
  const unbalanced = [];
  pairs.forEach(([o,c,label])=>{
    const no = text.split(o).length-1, nc = text.split(c).length-1;
    if(no !== nc) unbalanced.push(`${label} ${o}${no} / ${c}${nc}`);
  });
  add({ group:'标点规范', name:'成对标点不配对', count: unbalanced.length, samples: [],
        sev: unbalanced.length ? 'warn':'ok',
        extra: unbalanced.join('；'),
        desc:'左右符号数量不一致，常见于漏打、错打，或复制拼接时截断。数字为左/右符号的个数。' });

  const dupPunc = scan(text, /([，。、；：,;])\1+/g);
  add({ group:'标点规范', name:'标点重复（，，/ 。。等）', count: dupPunc.count, samples: dupPunc.samples,
        sev: dupPunc.count>0 ? 'warn':'ok',
        desc:'同一个标点连续出现，通常是录入错误。' });

  const mixedEnd = scan(text, /[，。、；：][，。、；：]/g);
  add({ group:'标点规范', name:'不同标点叠用（，。/ 。，等）', count: Math.max(0, mixedEnd.count - dupPunc.count), samples: mixedEnd.samples.filter(s=>s.hit[0]!==s.hit[1]).slice(0,6),
        sev: (mixedEnd.count - dupPunc.count)>0 ? 'warn':'ok',
        desc:'两个不同的点号连在一起，通常是修改后残留。' });

  const badEllipsis = scan(text, /\.{3,}|。{3,}|(?<!…)…(?!…)|…{3,}/g);
  add({ group:'标点规范', name:'不规范省略号', count: badEllipsis.count, samples: badEllipsis.samples,
        sev: badEllipsis.count>0 ? 'warn':'ok',
        desc:'中文省略号应为两个"…"连用（……），而不是"..."、"。。。"或单个"…"。' });

  const badDash = scan(text, new RegExp(`(?<![—\\-])—(?!—)|(?<=[${CJK}])--+(?=[${CJK}])|－{2,}`, 'g'));
  add({ group:'标点规范', name:'不规范破折号', count: badDash.count, samples: badDash.samples,
        sev: badDash.count>0 ? 'warn':'ok',
        desc:'中文破折号应为两个"—"连用（——），而不是单个"—"或"--"。' });

  /* ---------- 4. 空格与段落排版 ---------- */

  const spaceAroundPunc = scan(text, /[ \t]+[，。、；：！？）》」』”]|[（《「『“][ \t]+/g);
  add({ group:'空格与段落', name:'全角标点前后多余空格', count: spaceAroundPunc.count, samples: spaceAroundPunc.samples,
        sev: spaceAroundPunc.count>10 ? 'warn' : (spaceAroundPunc.count>0 ? 'info':'ok'),
        desc:'全角标点本身已带间距，前后不需要再加空格。' });

  const multiSpace = scan(text, / {2,}/g);
  add({ group:'空格与段落', name:'连续多个半角空格', count: multiSpace.count, samples: multiSpace.samples,
        sev: multiSpace.count>20 ? 'warn' : (multiSpace.count>0 ? 'info':'ok'),
        desc:'用多个空格来对齐或缩进，排版不稳定，建议改用段落格式设置。' });

  const lines = text.split(/\n/).filter(l=>l.trim().length>=20);
  const indentFW = lines.filter(l=>/^\u3000{2}/.test(l)).length;
  const indentSP = lines.filter(l=>/^ {2,}/.test(l)).length;
  const indentNone = lines.filter(l=>!/^[\u3000 \t]/.test(l)).length;
  const styles = [indentFW, indentSP, indentNone].filter(n=>n > lines.length*0.1).length;
  add({ group:'空格与段落', name:'段首缩进方式不统一', count: styles>1 ? lines.length : 0, samples: [],
        sev: styles>1 ? 'warn':'ok',
        extra: `共 ${lines.length} 段：全角空格缩进 ${indentFW} · 半角空格缩进 ${indentSP} · 无缩进 ${indentNone}`,
        desc:'段落开头有的用全角空格、有的用半角空格、有的不缩进。.docx 里通过段落格式设置的缩进不体现在文字中，所以这项对 .docx 仅作参考。' });

  const blankRuns = countOf(text, /\n[ \t\u3000]*\n[ \t\u3000]*\n[ \t\u3000]*\n/g);
  add({ group:'空格与段落', name:'连续 3 个以上空行', count: blankRuns, samples: [],
        sev: blankRuns>5 ? 'warn' : (blankRuns>0 ? 'info':'ok'),
        desc:'用大量空行来分页或留白，排版稳定性差。' });

  const fwPeriod = countOf(text, /．/g), cnPeriod = countOf(text, /。/g);
  add({ group:'空格与段落', name:'句号样式混用（。与 ．）', count: (fwPeriod && cnPeriod) ? Math.min(fwPeriod,cnPeriod) : 0, samples: [],
        sev: (fwPeriod && cnPeriod) ? 'warn':'ok',
        extra: `。 ${cnPeriod} 个 · ． ${fwPeriod} 个`,
        desc:'部分学科（如理工科）要求用"．"作句号，但全文应统一。' });

  return items;
}

/* ---------------- .docx 深度格式分析 ---------------- */

function attr(tag, name){
  const m = new RegExp(`${name}="([^"]*)"`).exec(tag);
  return m ? m[1] : null;
}

function runText(runXml){
  let t = '';
  const re = /<w:t(?:\s[^>]*)?>([^<]*)<\/w:t>/g; let m;
  while((m = re.exec(runXml))) t += m[1];
  return t.replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&apos;/g,"'").replace(/&amp;/g,'&');
}

function isOn(rPr, tagName){
  // <w:vanish/> 或 <w:vanish w:val="true|1|on"/> 为开启；w:val="0|false|off" 为关闭
  const m = new RegExp(`<${tagName}(\\s[^>]*)?\\/>`).exec(rPr);
  if(!m) return false;
  const v = attr(m[0], 'w:val');
  return v === null || !/^(0|false|off)$/i.test(v);
}

async function analyzeDocxFormat(arrayBuffer){
  const zip = await JSZip.loadAsync(arrayBuffer);
  const read = async (p)=>{ const f = zip.file(p); return f ? f.async('string') : ''; };
  const xml = await read('word/document.xml');
  if(!xml) throw new Error('未找到 word/document.xml，文件可能不是标准 .docx');
  const stylesXml = await read('word/styles.xml');
  const footXml = await read('word/footnotes.xml');
  const endXml = await read('word/endnotes.xml');
  const commentsXml = await read('word/comments.xml');
  const coreXml = await read('docProps/core.xml');
  const appXml = await read('docProps/app.xml');

  const fontUse = new Map();   // 字体 -> 出现次数（run 级）
  const sizeUse = new Map();   // 字号(pt) -> 次数
  const colorUse = new Map();
  const hidden = { count:0, chars:0, samples:[] };
  const white = { count:0, chars:0, samples:[] };
  const tiny = { count:0, chars:0, samples:[] };
  const squeezed = { count:0, chars:0, samples:[] };
  let highlightRuns = 0, runCount = 0;

  const pushSample = (bucket, t)=>{
    bucket.count++; bucket.chars += t.length;
    if(bucket.samples.length<6 && t.trim()) bucket.samples.push(t.length>60 ? t.slice(0,60)+'…' : t);
  };

  const runRe = /<w:r(?:\s[^>]*)?>([\s\S]*?)<\/w:r>/g;
  let m;
  while((m = runRe.exec(xml))){
    runCount++;
    const body = m[1];
    const rPrM = /<w:rPr>([\s\S]*?)<\/w:rPr>/.exec(body);
    const rPr = rPrM ? rPrM[1] : '';
    const t = runText(body);
    if(!t) continue;

    const fontsTag = /<w:rFonts\b[^>]*>/.exec(rPr);
    if(fontsTag){
      ['w:eastAsia','w:ascii','w:hAnsi'].forEach(a=>{
        const f = attr(fontsTag[0], a);
        if(f) fontUse.set(f, (fontUse.get(f)||0) + 1);
      });
    }
    const szTag = /<w:sz\s[^>]*>/.exec(rPr);
    if(szTag){
      const pt = parseInt(attr(szTag[0],'w:val'),10)/2;
      if(!isNaN(pt)){
        sizeUse.set(pt, (sizeUse.get(pt)||0)+1);
        if(pt <= 3) pushSample(tiny, t);
      }
    }
    const colTag = /<w:color\s[^>]*>/.exec(rPr);
    if(colTag){
      const c = (attr(colTag[0],'w:val')||'').toUpperCase();
      if(c && c!=='AUTO') colorUse.set(c, (colorUse.get(c)||0)+1);
      if(c==='FFFFFF' || c==='FEFEFE' || c==='FDFDFD') pushSample(white, t);
    }
    if(isOn(rPr,'w:vanish') || isOn(rPr,'w:specVanish') || isOn(rPr,'w:webHidden')) pushSample(hidden, t);
    const spTag = /<w:spacing\s[^>]*>/.exec(rPr);
    const wTag = /<w:w\s[^>]*>/.exec(rPr);
    const sp = spTag ? parseInt(attr(spTag[0],'w:val'),10) : 0;
    const scale = wTag ? parseInt(attr(wTag[0],'w:val'),10) : 100;
    if(sp <= -40 || scale <= 50) pushSample(squeezed, t);
    if(/<w:highlight\s/.test(rPr) || /<w:shd\s/.test(rPr)) highlightRuns++;
  }

  // 样式表里定义的字体（正文默认字体通常在这里）
  const styleFonts = new Set();
  { const re = /<w:rFonts\b[^>]*>/g; let s;
    while((s = re.exec(stylesXml))){
      ['w:eastAsia','w:ascii','w:hAnsi'].forEach(a=>{ const f = attr(s[0],a); if(f) styleFonts.add(f); });
    }
  }

  // 段落级：行距、首行缩进、对齐、样式
  const lineSp = new Map(), firstInd = new Map(), aligns = new Map(), pStyles = new Map();
  { const re = /<w:pPr>([\s\S]*?)<\/w:pPr>/g; let p;
    while((p = re.exec(xml))){
      const pp = p[1];
      const s = /<w:spacing\s[^>]*>/.exec(pp);
      if(s){
        const line = attr(s[0],'w:line'), rule = attr(s[0],'w:lineRule') || 'auto';
        if(line){
          const label = rule==='auto' ? `${(parseInt(line,10)/240).toFixed(2).replace(/\.?0+$/,'')} 倍` : `固定 ${(parseInt(line,10)/20).toFixed(1).replace(/\.0$/,'')} 磅`;
          lineSp.set(label, (lineSp.get(label)||0)+1);
        }
      }
      const ind = /<w:ind\s[^>]*>/.exec(pp);
      if(ind){
        const fc = attr(ind[0],'w:firstLineChars'), fl = attr(ind[0],'w:firstLine');
        const label = fc ? `${parseInt(fc,10)/100} 字符` : (fl ? `${(parseInt(fl,10)/20).toFixed(0)} 磅` : null);
        if(label) firstInd.set(label, (firstInd.get(label)||0)+1);
      }
      const jc = /<w:jc\s[^>]*>/.exec(pp);
      if(jc){ const v = attr(jc[0],'w:val'); aligns.set(v, (aligns.get(v)||0)+1); }
      const ps = /<w:pStyle\s[^>]*>/.exec(pp);
      if(ps){ const v = attr(ps[0],'w:val'); pStyles.set(v, (pStyles.get(v)||0)+1); }
    }
  }

  // 表格
  const tables = (xml.match(/<w:tbl>/g) || []).length;
  const tableRows = (xml.match(/<w:tr[\s>]/g) || []).length;
  const tableCells = (xml.match(/<w:tc>/g) || []).length;
  const nested = (()=>{ let depth=0, max=0; const re=/<w:tbl>|<\/w:tbl>/g; let t;
    while((t=re.exec(xml))){ depth += t[0]==='<w:tbl>'?1:-1; max=Math.max(max,depth); } return max>1; })();

  // 其他对象
  const textBoxes = (xml.match(/<w:txbxContent>/g) || []).length;
  const images = (xml.match(/<w:drawing>|<w:pict>/g) || []).length;
  const insCount = (xml.match(/<w:ins\s/g) || []).length;
  const delCount = (xml.match(/<w:del\s/g) || []).length;
  const footnotes = Math.max(0, (footXml.match(/<w:footnote\s[^>]*w:id="(?!-1"|0")/g) || []).length);
  const endnotes = Math.max(0, (endXml.match(/<w:endnote\s[^>]*w:id="(?!-1"|0")/g) || []).length);
  const comments = (commentsXml.match(/<w:comment\s/g) || []).length;
  const fields = (xml.match(/<w:fldSimple\s|<w:instrText/g) || []).length;
  const mathObjs = (xml.match(/<m:oMath>/g) || []).length;

  // 文档属性
  const tagVal = (src, tag)=>{ const r = new RegExp(`<${tag}[^>]*>([^<]*)</${tag}>`).exec(src); return r ? r[1] : ''; };
  const meta = {
    creator: tagVal(coreXml,'dc:creator'),
    lastModifiedBy: tagVal(coreXml,'cp:lastModifiedBy'),
    created: tagVal(coreXml,'dcterms:created'),
    modified: tagVal(coreXml,'dcterms:modified'),
    revision: tagVal(coreXml,'cp:revision'),
    application: tagVal(appXml,'Application'),
    appVersion: tagVal(appXml,'AppVersion'),
    totalTime: tagVal(appXml,'TotalTime')
  };

  const sortMap = (mp)=>[...mp.entries()].sort((a,b)=>b[1]-a[1]);

  return {
    runCount,
    fonts: sortMap(fontUse),
    styleFonts: [...styleFonts],
    sizes: sortMap(sizeUse),
    colors: sortMap(colorUse),
    hidden, white, tiny, squeezed, highlightRuns,
    lineSpacing: sortMap(lineSp),
    firstIndent: sortMap(firstInd),
    aligns: sortMap(aligns),
    pStyles: sortMap(pStyles),
    tables, tableRows, tableCells, nested,
    textBoxes, images, insCount, delCount, footnotes, endnotes, comments, fields, mathObjs,
    meta
  };
}

/* ---------------- .pdf 字体 / 字号 / 隐形文字分析 ---------------- */

async function analyzePdfFormat(pdf, onProgress){
  const OPS = pdfjsLib.OPS;
  const fontUse = new Map();
  const sizeUse = new Map();
  let invisibleModeOps = 0, whiteFillOps = 0, images = 0, tinyItems = 0;
  const tinySamples = [];
  const maxPages = Math.min(pdf.numPages, 400);

  for(let i=1;i<=maxPages;i++){
    const page = await pdf.getPage(i);
    const ops = await page.getOperatorList();
    for(let k=0;k<ops.fnArray.length;k++){
      const fn = ops.fnArray[k], args = ops.argsArray[k];
      if(fn === OPS.setTextRenderingMode && args && args[0] === 3) invisibleModeOps++;
      if(fn === OPS.setFillRGBColor && args){
        const a = args[0];
        const isWhite = (typeof a === 'string') ? /^#?f{6}$/i.test(a)
                      : (args.length>=3 && args[0]>=250 && args[1]>=250 && args[2]>=250);
        if(isWhite) whiteFillOps++;
      }
      if(fn === OPS.paintImageXObject || fn === OPS.paintInlineImageXObject || fn === OPS.paintJpegXObject) images++;
    }
    const content = await page.getTextContent();
    const nameOf = {};
    Object.keys(content.styles || {}).forEach(id=>{
      let name = null;
      try{ if(page.commonObjs.has(id)){ const f = page.commonObjs.get(id); name = f && (f.name || f.loadedName); } }catch(e){}
      if(!name) name = (content.styles[id].fontFamily || id);
      nameOf[id] = String(name).replace(/^[A-Z]{6}\+/, '');
    });
    content.items.forEach(it=>{
      if(!it.str || !it.str.trim()) return;
      const fname = nameOf[it.fontName] || it.fontName;
      fontUse.set(fname, (fontUse.get(fname)||0) + it.str.length);
      const size = Math.round(Math.hypot(it.transform[2], it.transform[3]) * 2) / 2;
      if(size>0){
        sizeUse.set(size, (sizeUse.get(size)||0) + it.str.length);
        if(size <= 3){ tinyItems++; if(tinySamples.length<6) tinySamples.push(it.str.slice(0,60)); }
      }
    });
    if(onProgress) onProgress(i, maxPages);
    page.cleanup();
    if(i % 5 === 0) await new Promise(r=>setTimeout(r,0));
  }

  let meta = {};
  try{
    const md = await pdf.getMetadata();
    const info = md.info || {};
    meta = { creator: info.Author || '', application: info.Creator || '', producer: info.Producer || '',
             created: info.CreationDate || '', modified: info.ModDate || '' };
  }catch(e){}

  const sortMap = (mp)=>[...mp.entries()].sort((a,b)=>b[1]-a[1]);
  return {
    fonts: sortMap(fontUse), sizes: sortMap(sizeUse),
    invisibleModeOps, whiteFillOps, images, tinyItems, tinySamples,
    scannedPages: maxPages, totalPages: pdf.numPages, meta
  };
}

async function buildFormatReport(text){
  const report = { punctuation: analyzePunctuation(text), docx: null, pdf: null };
  if(currentFileExt === 'docx' && currentArrayBuffer){
    progressLabel.textContent = '正在读取 Word 文档内部格式…';
    await sleep(0);
    try{ report.docx = await analyzeDocxFormat(currentArrayBuffer); }
    catch(e){ report.docx = { error: e.message }; }
  }
  if(currentFileExt === 'pdf' && currentPdfDoc){
    try{
      report.pdf = await analyzePdfFormat(currentPdfDoc, (i,n)=>{
        progressLabel.textContent = `正在检查 PDF 字体与排版 … 第 ${i} / ${n} 页`;
        progressFill.style.width = Math.round(i/n*100) + '%';
      });
    }catch(e){ report.pdf = { error: e.message }; }
  }
  return report;
}

/* ---------------- 渲染 ---------------- */

function sevBadge(sev){ return `<span class="sev sev-${sev}">${SEV_LABEL[sev]}</span>`; }

function sampleHtml(samples){
  if(!samples || !samples.length) return '';
  return `<ul class="sample-list">${samples.map(s=>{
    if(typeof s === 'string') return `<li><code>${escapeHtml(s)}</code></li>`;
    const vis = (x)=>escapeHtml(x).replace(/[\u200B\u200C\u200D\u2060\uFEFF\u00AD\u180E]/g,'<mark class="zw">⟦零宽⟧</mark>').replace(/\n/g,'↵');
    return `<li><code>${vis(s.before)}<mark>${vis(s.hit).replace(/ /g,'␣') || '·'}</mark>${vis(s.after)}</code></li>`;
  }).join('')}</ul>`;
}

function rowHtml(r){
  return `<details class="check-row sev-row-${r.sev}" ${r.sev==='danger' && r.count>0 ? 'open':''}>
    <summary>
      <span class="check-name">${escapeHtml(r.name)}</span>
      <span class="check-count">${r.count}</span>
      ${sevBadge(r.sev)}
    </summary>
    <div class="check-body">
      <p>${escapeHtml(r.desc)}</p>
      ${r.extra ? `<p class="check-extra">${escapeHtml(r.extra)}</p>` : ''}
      ${sampleHtml(r.samples)}
    </div>
  </details>`;
}

function listPairs(pairs, fmt, limit){
  limit = limit || 12;
  if(!pairs.length) return '<span class="muted">无</span>';
  const shown = pairs.slice(0, limit).map(([k,v])=>`<code>${escapeHtml(fmt ? fmt(k) : String(k))}</code><span class="muted">×${v}</span>`).join(' ');
  return shown + (pairs.length>limit ? ` <span class="muted">等 ${pairs.length} 种</span>` : '');
}

function docxItems(d){
  const items = [];
  const bucketRow = (group, name, b, sevIfAny, desc)=>({
    group, name, count: b.count, sev: b.count ? sevIfAny : 'ok',
    extra: b.count ? `共 ${b.chars} 个字符` : '', samples: b.samples, desc });

  items.push(bucketRow('规避痕迹','隐藏文字（Word"隐藏"属性）', d.hidden, 'danger',
    '设置了"隐藏"属性的文字，在 Word 里默认看不见、打印不出，但会被检测系统提取到。常被用来塞入无关文字以稀释重复率。'));
  items.push(bucketRow('规避痕迹','白色文字', d.white, 'danger',
    '字体颜色为白色，在白纸上肉眼不可见，但会被提取到。'));
  items.push(bucketRow('规避痕迹','极小字号文字（≤ 3 磅）', d.tiny, 'danger',
    '字号小到几乎看不见的文字，与隐藏文字作用相同。'));
  items.push(bucketRow('规避痕迹','字符被极度压缩', d.squeezed, 'warn',
    '字符间距被大幅压缩或宽度缩放到 50% 以下，可能用来把大量文字挤进很小的空间。'));
  items.push({ group:'规避痕迹', name:'文本框', count: d.textBoxes, sev: d.textBoxes>3 ? 'warn' : (d.textBoxes ? 'info':'ok'), samples:[],
    desc:'部分检测系统不提取文本框里的文字。正文放进文本框会导致检测不完整，一般论文也不允许这样排版。' });

  const fontCount = d.fonts.length;
  items.push({ group:'字体与字号', name:'正文中直接设置的字体', count: fontCount, sev: fontCount>6 ? 'warn' : 'info', samples:[],
    extra: '', html: listPairs(d.fonts),
    desc:'按文字片段统计的字体使用次数。字体种类过多，说明全文格式可能不统一，或由多个来源拼接而成。' });
  items.push({ group:'字体与字号', name:'样式表中定义的字体', count: d.styleFonts.length, sev:'info', samples:[],
    html: d.styleFonts.length ? d.styleFonts.map(f=>`<code>${escapeHtml(f)}</code>`).join(' ') : '<span class="muted">无</span>',
    desc:'Word 样式（正文、标题 1 等）里设定的字体。未直接设置字体的文字会使用这些字体。' });
  items.push({ group:'字体与字号', name:'字号种类', count: d.sizes.length, sev: d.sizes.length>8 ? 'warn':'info', samples:[],
    html: listPairs(d.sizes, k=>`${k} 磅`),
    desc:'正文、标题、脚注、表格一般各用一种字号，种类过多说明格式不统一。' });
  items.push({ group:'字体与字号', name:'文字颜色', count: d.colors.length, sev: d.colors.length>3 ? 'warn':'info', samples:[],
    html: listPairs(d.colors, k=>`#${k}`),
    desc:'除黑色（自动）以外设置过的字体颜色。' });
  items.push({ group:'字体与字号', name:'高亮 / 底纹', count: d.highlightRuns, sev: d.highlightRuns ? 'info':'ok', samples:[],
    desc:'提交前通常应清除修改时留下的高亮和底纹。' });

  items.push({ group:'段落排版', name:'行距种类', count: d.lineSpacing.length, sev: d.lineSpacing.length>3 ? 'warn':'info', samples:[],
    html: listPairs(d.lineSpacing), desc:'段落上直接设置的行距。' });
  items.push({ group:'段落排版', name:'首行缩进种类', count: d.firstIndent.length, sev: d.firstIndent.length>2 ? 'warn':'info', samples:[],
    html: listPairs(d.firstIndent), desc:'中文正文通常统一为首行缩进 2 字符。' });
  items.push({ group:'段落排版', name:'对齐方式', count: d.aligns.length, sev:'info', samples:[],
    html: listPairs(d.aligns, k=>({both:'两端对齐',left:'左对齐',start:'左对齐',center:'居中',right:'右对齐',end:'右对齐',distribute:'分散对齐'}[k]||k)),
    desc:'段落上直接设置的对齐方式。' });
  items.push({ group:'段落排版', name:'使用的段落样式', count: d.pStyles.length, sev:'info', samples:[],
    html: listPairs(d.pStyles, null, 20), desc:'标题是否使用了规范的"标题 1/2/3"样式，会影响目录生成和部分系统的章节识别。' });

  items.push({ group:'表格与对象', name:'表格', count: d.tables, sev:'info', samples:[],
    extra: `共 ${d.tableRows} 行、${d.tableCells} 个单元格${d.nested ? '；含嵌套表格' : ''}`,
    desc:'表格文字一般也会被提取检测。嵌套表格可能导致提取顺序错乱。' });
  items.push({ group:'表格与对象', name:'图片 / 图形', count: d.images, sev:'info', samples:[],
    desc:'以图片形式插入的文字不会被文本检测识别；如果大段正文是图片，检测结果会偏低，也可能被要求说明。' });
  items.push({ group:'表格与对象', name:'公式对象', count: d.mathObjs, sev:'info', samples:[], desc:'Word 公式编辑器插入的公式。' });
  items.push({ group:'表格与对象', name:'脚注 / 尾注', count: d.footnotes + d.endnotes, sev:'info', samples:[],
    extra: `脚注 ${d.footnotes} · 尾注 ${d.endnotes}`,
    desc:'注释内容是否计入检测，各系统做法不同。' });
  items.push({ group:'表格与对象', name:'域代码（目录、交叉引用等）', count: d.fields, sev:'info', samples:[], desc:'自动目录、页码、引用等。' });

  items.push({ group:'修订与元数据', name:'未处理的修订', count: d.insCount + d.delCount, sev: (d.insCount+d.delCount) ? 'warn':'ok', samples:[],
    extra: `插入 ${d.insCount} 处 · 删除 ${d.delCount} 处`,
    desc:'文档里还保留着"修订"痕迹。被删除的文字可能仍被提取出来，提交前应全部接受或拒绝。' });
  items.push({ group:'修订与元数据', name:'批注', count: d.comments, sev: d.comments ? 'warn':'ok', samples:[],
    desc:'提交前通常应删除批注。' });
  const m = d.meta;
  const metaStr = [
    m.creator && `作者：${m.creator}`, m.lastModifiedBy && `最后修改者：${m.lastModifiedBy}`,
    m.application && `生成软件：${m.application}${m.appVersion ? ' '+m.appVersion : ''}`,
    m.created && `创建：${m.created}`, m.modified && `修改：${m.modified}`,
    m.revision && `保存次数：${m.revision}`, m.totalTime && `累计编辑：${m.totalTime} 分钟`
  ].filter(Boolean).join('；');
  items.push({ group:'修订与元数据', name:'文档属性', count: metaStr ? '—' : 0, sev:'info', samples:[], extra: metaStr || '无',
    desc:'文件自带的作者、软件、时间信息，会随文件一起提交。如果作者名不是你、或累计编辑时间异常短，提交前可在"文件 → 信息 → 检查文档"里清理。' });
  return items;
}

function pdfItems(p){
  const items = [];
  items.push({ group:'规避痕迹', name:'不可见渲染模式的文字', count: p.invisibleModeOps, sev: p.invisibleModeOps ? 'warn':'ok', samples:[],
    desc:'PDF 中以"不可见"模式绘制的文字。扫描件经 OCR 后的文字层也是这种模式，属正常；非扫描件出现则需核查。' });
  items.push({ group:'规避痕迹', name:'白色填充指令', count: p.whiteFillOps, sev: p.whiteFillOps>20 ? 'warn':'info', samples:[],
    desc:'把填充色设为白色的次数。白色背景块、表格底色也会用到，数量多不一定有问题，仅作参考。' });
  items.push({ group:'规避痕迹', name:'极小字号文字（≤ 3 磅）', count: p.tinyItems, sev: p.tinyItems ? 'danger':'ok', samples: p.tinySamples,
    desc:'字号小到几乎看不见的文字。' });
  items.push({ group:'字体与字号', name:'实际使用的字体', count: p.fonts.length, sev: p.fonts.length>8 ? 'warn':'info', samples:[],
    html: listPairs(p.fonts), desc:'按字符数统计。名称前的 6 位随机前缀（子集嵌入标记）已去掉。' });
  items.push({ group:'字体与字号', name:'字号种类', count: p.sizes.length, sev: p.sizes.length>10 ? 'warn':'info', samples:[],
    html: listPairs(p.sizes, k=>`${k} 磅`), desc:'按字符数统计的字号分布。' });
  items.push({ group:'表格与对象', name:'图片', count: p.images, sev:'info', samples:[],
    desc:'图片中的文字不会被文本检测识别。PDF 无法可靠识别表格结构，表格请以 .docx 版本检查。' });
  const m = p.meta || {};
  const metaStr = [m.creator && `作者：${m.creator}`, m.application && `生成软件：${m.application}`, m.producer && `PDF 生成器：${m.producer}`,
    m.created && `创建：${m.created}`, m.modified && `修改：${m.modified}`].filter(Boolean).join('；');
  items.push({ group:'修订与元数据', name:'文档属性', count: metaStr ? '—' : 0, sev:'info', samples:[], extra: metaStr || '无',
    desc:'PDF 自带的作者、软件、时间信息。' });
  if(p.scannedPages < p.totalPages){
    items.push({ group:'修订与元数据', name:'检查范围', count:'—', sev:'info', samples:[], extra:`为控制耗时，仅检查了前 ${p.scannedPages} / ${p.totalPages} 页`, desc:'' });
  }
  return items;
}

function renderFormatReport(report){
  let items = report.punctuation.slice();
  let note = '';
  if(report.docx){
    if(report.docx.error) note += `<p class="check-extra">Word 格式解析失败：${escapeHtml(report.docx.error)}</p>`;
    else items = items.concat(docxItems(report.docx));
  }
  if(report.pdf){
    if(report.pdf.error) note += `<p class="check-extra">PDF 格式解析失败：${escapeHtml(report.pdf.error)}</p>`;
    else items = items.concat(pdfItems(report.pdf));
  }
  if(!report.docx && !report.pdf){
    note += `<p class="check-extra">当前是纯文本（粘贴或 .txt），只能检查标点与空格。字体、字号、表格、隐藏文字、修订等检查需要上传 .docx（最完整）或 .pdf。</p>`;
  }
  lastFormatItems = items;

  const danger = items.filter(i=>i.sev==='danger' && i.count).length;
  const warn = items.filter(i=>i.sev==='warn' && i.count).length;
  formatGrid.innerHTML =
    fmtCard(danger, '需重点核查项', danger>0) +
    fmtCard(warn, '不规范 / 不统一项', warn>0) +
    fmtCard(items.length, '检查项总数', false);

  const groups = ['规避痕迹','标点规范','空格与段落','字体与字号','段落排版','表格与对象','修订与元数据'];
  let html = note;
  groups.forEach(g=>{
    const gi = items.filter(i=>i.group===g);
    if(!gi.length) return;
    html += `<h3 class="check-group">${g}</h3>`;
    gi.forEach(r=>{
      if(r.html){
        html += `<details class="check-row sev-row-${r.sev}">
          <summary><span class="check-name">${escapeHtml(r.name)}</span><span class="check-count">${r.count}</span>${sevBadge(r.sev)}</summary>
          <div class="check-body"><p>${escapeHtml(r.desc)}</p>${r.extra?`<p class="check-extra">${escapeHtml(r.extra)}</p>`:''}<p class="pair-list">${r.html}</p></div>
        </details>`;
      } else {
        html += rowHtml(r);
      }
    });
  });
  formatDetail.innerHTML = html;
}

function fmtCard(num, label, flagged){
  return `<div class="format-card ${flagged?'flag':'ok'}">
    <span class="format-num">${num}</span>
    <span class="format-label">${label}</span>
  </div>`;
}

function formatItemsToText(items){
  if(!items || !items.length) return '';
  let out = `\n${'='.repeat(60)}\n标点与排版格式检查\n${'='.repeat(60)}\n`;
  let group = '';
  items.forEach(r=>{
    if(r.group !== group){ group = r.group; out += `\n【${group}】\n`; }
    out += `- ${r.name}：${r.count}（${SEV_LABEL[r.sev]}）`;
    if(r.extra) out += `  ${r.extra}`;
    if(r.html) out += `  ${r.html.replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').trim()}`;
    out += '\n';
    (r.samples||[]).slice(0,3).forEach(s=>{
      const t = typeof s === 'string' ? s : `${s.before}【${s.hit}】${s.after}`;
      out += `    例：${t.replace(/\n/g,'↵').replace(/[\u200B\u200C\u200D\u2060\uFEFF\u00AD\u180E]/g,'⟦零宽⟧')}\n`;
    });
  });
  return out;
}

/* ============================================================
   官方 API 增强检测（可选）
   请求发给同源的 /api/detect（由 worker.js 处理），
   再由 Worker 转发给 GPTZero。Key 不写进网页代码。
   ============================================================ */

async function runApiDetection(text){
  apiSection.classList.remove('hidden');
  apiResultBody.innerHTML = '<p class="api-note">正在调用官方 API…</p>';

  if(location.protocol === 'file:'){
    apiResultBody.innerHTML = '<p class="api-note">⚠ 当前是直接打开的本地文件，无法调用 API。请先按 README 部署成 Cloudflare Worker。</p>';
    return;
  }

  const CHUNK = 45000; // GPTZero 单次上限约 5 万字符
  const pieces = [];
  for(let i=0;i<text.length;i+=CHUNK) pieces.push(text.slice(i, i+CHUNK));

  const headers = { 'Content-Type':'application/json' };
  const k = apiKeyInput.value.trim();
  const pw = accessPwInput.value;
  if(k) headers['X-User-Api-Key'] = k;
  if(pw) headers['X-Access-Password'] = pw;

  const perPiece = [];
  const flagged = [];
  for(let i=0;i<pieces.length;i++){
    progressLabel.textContent = `正在调用官方 API … ${i+1} / ${pieces.length}`;
    progressFill.style.width = Math.round((i+1)/pieces.length*100) + '%';
    try{
      const res = await fetch('/api/detect', { method:'POST', headers, body: JSON.stringify({ text: pieces[i] }) });
      let data;
      try{ data = await res.json(); }catch(e){ data = { message: `服务器返回的不是 JSON（HTTP ${res.status}）。如果你是用 Pages 拖拽上传的，那种方式不支持 API，请改用 Worker 部署。` }; }
      if(!res.ok || data.error){
        apiResultBody.innerHTML = `<p class="api-note">⚠ ${escapeHtml(data.message || ('HTTP '+res.status))}</p>`;
        lastApiSummary = null;
        return;
      }
      const prob = data.completely_generated_prob ?? data.average_generated_prob ?? 0;
      perPiece.push({ len: pieces[i].length, prob, cls: data.predicted_class || '' });
      (data.sentences || []).forEach(s=>{ if(s.generated_prob >= 0.7 && flagged.length < 30) flagged.push(s); });
    }catch(err){
      apiResultBody.innerHTML = `<p class="api-note">⚠ 请求失败：${escapeHtml(err.message)}</p>`;
      lastApiSummary = null;
      return;
    }
  }

  const totalLen = perPiece.reduce((a,p)=>a+p.len,0) || 1;
  const weighted = perPiece.reduce((a,p)=>a + p.prob*p.len, 0) / totalLen;
  const pct = Math.round(weighted*100);
  lastApiSummary = { pct, pieces: pieces.length, flagged };

  apiResultBody.innerHTML = `
    <div class="api-card">
      <div class="api-stat"><span class="format-num">${pct}%</span><span class="format-label">GPTZero 判定 AI 生成概率</span></div>
      <div class="api-stat"><span class="format-num">${pieces.length}</span><span class="format-label">分段请求数</span></div>
      <div class="api-stat"><span class="format-num">${flagged.length}${flagged.length>=30?'+':''}</span><span class="format-label">高概率句子（≥70%）</span></div>
    </div>
    ${flagged.length ? `<details class="check-row"><summary><span class="check-name">查看高概率句子</span></summary><div class="check-body"><ul class="sample-list">${flagged.map(s=>`<li><code>${escapeHtml(s.text||'')}</code> <span class="muted">${Math.round(s.generated_prob*100)}%</span></li>`).join('')}</ul></div></details>` : ''}
    <p class="api-note">这个分数来自 GPTZero 的神经网络模型，和上面的本地统计分数是两种独立方法，可以对照看，但不要简单取平均。GPTZero 对中文古籍引文较多的文本同样可能误判。</p>`;
}

/* ---------------- 导出报告 ---------------- */
exportBtn.addEventListener('click', ()=>{
  if(!lastReport) return;
  const { results, totalLen, source, time } = lastReport;
  const weightedSum = results.reduce((a,r)=>a + r.score*r.len, 0);
  const overall = totalLen ? Math.round(weightedSum/totalLen) : 0;
  const high = results.filter(r=>r.score>=60).length;
  const mid  = results.filter(r=>r.score>=30 && r.score<60).length;

  let out = '';
  out += `审读 · 文本特征检测报告\n`;
  out += `生成时间：${time.toLocaleString('zh-CN')}\n`;
  out += `来源：${source}\n`;
  out += `总字数：${totalLen.toLocaleString()}\n`;
  out += `综合疑似度：${overall}%\n`;
  out += `高疑似段落：${high} / 中等疑似段落：${mid} / 总段落数：${results.length}\n`;
  out += `\n【方法说明】本地疑似度基于统计启发式特征（句长突发性、n-gram 重复率、AI 高频过渡词密度、句长集中度），不使用神经网络分类模型，仅供写作风格自查参考，不能作为投稿或学术诚信判定依据。\n`;
  if(lastApiSummary){
    out += `\n【GPTZero 官方模型】AI 生成概率：${lastApiSummary.pct}%（分 ${lastApiSummary.pieces} 段请求）\n`;
    lastApiSummary.flagged.forEach(s=>{ out += `    ${Math.round(s.generated_prob*100)}%：${s.text}\n`; });
  }
  out += formatItemsToText(lastFormatItems);
  out += `\n${'='.repeat(60)}\n分段详情\n${'='.repeat(60)}\n`;
  results.forEach((r, idx)=>{
    out += `\n[第 ${idx+1} 段 | 疑似度 ${r.score} | ${r.len} 字 | ${r.tags.join('、') || '无显著特征'}]\n${r.text}\n`;
  });

  const blob = new Blob([out], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `审读报告_${new Date().toISOString().slice(0,10)}.txt`;
  a.click();
  URL.revokeObjectURL(url);
});
