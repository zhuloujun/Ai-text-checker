/* ============================================================
   审读 · AI 文本检测（Hugging Face Space 版）前端
   - 文档在浏览器里解析；只把提取出的文字发给本服务的 /v1/detect
   - 标点与排版检查完全在浏览器里完成
   ============================================================ */

const $ = (id)=>document.getElementById(id);
const pasteArea = $('pasteArea'), dropzone = $('dropzone'), fileInput = $('fileInput');
const fileInfo = $('fileInfo'), fileName = $('fileName'), fileMeta = $('fileMeta'), fileClear = $('fileClear');
const charCount = $('charCount'), analyzeBtn = $('analyzeBtn');
const progressZone = $('progressZone'), progressFill = $('progressFill'), progressLabel = $('progressLabel');
const resultsZone = $('resultsZone'), paragraphList = $('paragraphList'), sealNum = $('sealNum');
const exportBtn = $('exportBtn'), errorBox = $('errorBox'), statusLine = $('statusLine');
const formatGrid = $('formatGrid'), formatDetail = $('formatDetail');
const apiKeyInput = $('apiKeyInput'), rememberKey = $('rememberKey'), keyShowBtn = $('keyShowBtn');

let currentText = '', currentSource = '粘贴文本';
let currentFileExt = null, currentArrayBuffer = null, currentPdfDoc = null;
let lastResult = null, lastFormatItems = null, lastFormatNote = '', health = null;
const Lib = ()=> window.Library;   // 本地文稿库（library.js），可能不存在
const STORE_KEY = 'shendu_hf_key';

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }
const pct = (x)=> x==null ? '—' : (Math.round(x*1000)/10) + '%';

/* ---------------- API Key ---------------- */
try{
  const saved = localStorage.getItem(STORE_KEY);
  if(saved){ apiKeyInput.value = saved; rememberKey.checked = true; }
}catch(e){}
function persistKey(){
  try{
    if(rememberKey.checked) localStorage.setItem(STORE_KEY, apiKeyInput.value.trim());
    else localStorage.removeItem(STORE_KEY);
  }catch(e){}
}
apiKeyInput.addEventListener('input', persistKey);
rememberKey.addEventListener('change', persistKey);
keyShowBtn.addEventListener('click', ()=>{
  const show = apiKeyInput.type === 'password';
  apiKeyInput.type = show ? 'text' : 'password';
  keyShowBtn.textContent = show ? '隐藏' : '显示';
});
function authHeaders(){
  const k = apiKeyInput.value.trim();
  return k ? { 'Authorization': 'Bearer ' + k } : {};
}

/* ---------------- 服务状态 ---------------- */
async function refreshHealth(){
  try{
    const r = await fetch('/health');
    health = await r.json();
    const parts = [];
    const lm = health.lm, cl = health.classifier;
    const st = (d, name)=> !d.enabled ? `${name}：未启用` : d.ready ? `${name}：就绪` : d.error ? `${name}：加载失败` : `${name}：加载中`;
    parts.push(st(lm, '语言模型（Fast-DetectGPT / Binoculars）'));
    parts.push(st(cl, '中文分类器'));
    parts.push(health.calibration.calibrated ? '已校准' : '未校准（结果仅作相对参考）');
    if(!health.key_signing_configured && health.requires_key) parts.push('⚠ 管理员尚未设置 ADMIN_TOKEN');
    statusLine.textContent = parts.join(' · ');
    statusLine.classList.toggle('warn', !!(health.loading || (lm.enabled && !lm.ready) || (cl.enabled && !cl.ready)));
    if(health.loading) setTimeout(refreshHealth, 5000);
  }catch(e){
    statusLine.textContent = '无法连接服务：' + e.message;
    statusLine.classList.add('warn');
    setTimeout(refreshHealth, 8000);
  }
}
refreshHealth();
// 通过 Cloudflare 域名访问时才有轻量版（/lite/）
if(!location.hostname.endsWith('modal.run')){ const l = document.getElementById('liteLink'); if(l) l.hidden = false; }

/* ---------------- 输入 ---------------- */
document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    $('panel-paste').classList.toggle('hidden', btn.dataset.tab!=='paste');
    $('panel-upload').classList.toggle('hidden', btn.dataset.tab!=='upload');
    if(btn.dataset.tab==='paste') usePaste(); else syncAnalyzeState();
  });
});
function usePaste(){
  currentText = pasteArea.value; currentSource = '粘贴文本';
  currentFileExt = null; currentArrayBuffer = null; currentPdfDoc = null;
  updateCharCount(); syncAnalyzeState();
}
pasteArea.addEventListener('input', ()=>{ usePaste(); Lib() && Lib().onPasteInput(pasteArea.value); });

dropzone.addEventListener('click', (e)=>{ if(e.target.tagName!=='LABEL') fileInput.click(); });
dropzone.addEventListener('dragover', e=>{ e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', ()=> dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e=>{
  e.preventDefault(); dropzone.classList.remove('dragover');
  if(e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e=>{ if(e.target.files.length) handleFile(e.target.files[0]); });
fileClear.addEventListener('click', ()=>{
  currentText = ''; currentFileExt = null; currentArrayBuffer = null; currentPdfDoc = null;
  fileInfo.classList.add('hidden'); fileInput.value = '';
  updateCharCount(); syncAnalyzeState();
  Lib() && Lib().onFileCleared();
});

async function handleFile(file, opts){
  opts = opts || {};
  const ext = file.name.split('.').pop().toLowerCase();
  fileName.textContent = file.name;
  fileMeta.textContent = `解析中… (${(file.size/1024).toFixed(0)} KB)`;
  fileInfo.classList.remove('hidden');
  analyzeBtn.disabled = true;
  currentArrayBuffer = null; currentPdfDoc = null; currentFileExt = ext;
  try{
    let text = '';
    if(ext === 'txt' || ext === 'md'){
      text = await file.text();
    } else if(ext === 'docx'){
      const buf = await file.arrayBuffer();
      currentArrayBuffer = buf;
      text = (await mammoth.extractRawText({ arrayBuffer: buf })).value;
    } else if(ext === 'pdf'){
      const buf = await file.arrayBuffer();
      currentArrayBuffer = buf;
      text = await extractPdfText(buf);
    } else {
      throw new Error('不支持的文件格式');
    }
    currentText = text; currentSource = file.name;
    fileMeta.textContent = `${text.length.toLocaleString()} 字 · 解析完成`;
    if(ext === 'pdf' && currentPdfDoc && text.replace(/\s/g,'').length < currentPdfDoc.numPages * 20){
      fileMeta.textContent += ' · ⚠ 提取到的文字很少，可能是扫描件，请改传 .docx';
    }
    updateCharCount();
    if(!opts.fromLibrary && Lib()) Lib().onFileParsed(file, text);
  }catch(err){
    fileMeta.textContent = '解析失败：' + err.message;
    currentText = '';
  }
  syncAnalyzeState();
  return currentText;
}

async function extractPdfText(buf){
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  const pdf = await pdfjsLib.getDocument({
    data: buf.slice(0),
    cMapUrl: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/cmaps/',
    cMapPacked: true,
    standardFontDataUrl: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/standard_fonts/'
  }).promise;
  currentPdfDoc = pdf;
  const pages = [];
  for(let i=1;i<=pdf.numPages;i++){
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    // 按行拼接：y 坐标变化时换行，保留段落结构
    let lastY = null, line = '', out = [];
    content.items.forEach(it=>{
      const y = Math.round(it.transform[5]);
      if(lastY !== null && Math.abs(y - lastY) > 2){ out.push(line); line = ''; }
      line += it.str; lastY = y;
      if(it.hasEOL){ out.push(line); line = ''; lastY = null; }
    });
    if(line) out.push(line);
    pages.push(out.join('\n'));
    fileMeta.textContent = `正在解析 PDF … 第 ${i} / ${pdf.numPages} 页`;
  }
  return pages.join('\n\n');
}

function updateCharCount(){ charCount.textContent = `${currentText.length.toLocaleString()} 字`; }
function syncAnalyzeState(){ analyzeBtn.disabled = currentText.trim().length < 50; }

/* ---------------- 检测 ---------------- */
function showError(msg){
  errorBox.textContent = msg;
  errorBox.classList.toggle('hidden', !msg);
}

async function api(path, opts){
  const r = await fetch(path, opts);
  let data;
  try{ data = await r.json(); }catch(e){ data = { message: `服务返回异常（HTTP ${r.status}）` }; }
  if(!r.ok && r.status !== 202) throw Object.assign(new Error(data.message || ('HTTP ' + r.status)), { status: r.status, data });
  return data;
}

function fmtSec(s){
  if(s == null) return '';
  s = Math.round(s);
  return s >= 90 ? `约 ${Math.round(s/60)} 分钟` : `约 ${s} 秒`;
}

analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis(){
  const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
  if(activeTab === 'paste') usePaste();
  const text = currentText.trim();
  if(text.length < 50) return;
  if(!apiKeyInput.value.trim() && (!health || health.requires_key)){
    showError('请先填写 API Key。Key 由部署者在 /admin 页面签发。');
    apiKeyInput.focus();
    return;
  }
  showError('');
  analyzeBtn.disabled = true;
  resultsZone.classList.add('hidden');
  progressZone.classList.remove('hidden');
  progressFill.style.width = '0%';
  progressLabel.textContent = '正在提交…';

  try{
    const mode = document.querySelector('input[name=mode]:checked').value;
    let job = await api('/v1/detect', {
      method: 'POST',
      headers: { 'Content-Type':'application/json', ...authHeaders() },
      body: JSON.stringify({ text, mode, exclude_references: $('optRefs').checked, flag_quotations: $('optQuotes').checked, wait: false })
    });
    Lib() && Lib().onJobStarted(job.id);
    await finishJob(job, text);
  }catch(err){
    showError(err.message);
    if(err.jobGone && Lib()) Lib().onJobStarted(null);
  }finally{
    progressZone.classList.add('hidden');
    analyzeBtn.disabled = false;
  }
}

/* 轮询任务直到完成，然后显示结果并保存到本地文稿库（刷新页面后也能用 resumeJob 接着等） */
async function finishJob(job, text){
  {
    while(job.status === 'queued' || job.status === 'running'){
      if(job.status === 'queued'){
        progressLabel.textContent = `排队中（第 ${job.queue_position || 1} 位）${job.eta_sec ? ' · 预计检测' + fmtSec(job.eta_sec) : ''}`;
      } else {
        progressFill.style.width = Math.round((job.progress||0)*100) + '%';
        progressLabel.textContent = job.total
          ? `语言模型打分：${job.done} / ${job.total} 段 · 剩余${fmtSec(job.eta_sec) || '计算中'}`
          : '分类器检测中…';
      }
      await sleep(job.status === 'queued' ? 2000 : 1200);
      try{
        job = await api('/v1/jobs/' + job.id, { headers: authHeaders() });
      }catch(e){
        if(e.status === 404) throw Object.assign(new Error('服务器上找不到这次检测任务（服务重启过或已超过 1 小时），请重新检测。'), { jobGone: true });
        throw e;
      }
    }
    if(job.status === 'error') throw new Error('检测失败：' + job.error);

    lastResult = job.result;
    renderResult(job.result);
    progressLabel.textContent = '正在检查标点与排版…';
    progressFill.style.width = '100%';
    const formatReport = await buildFormatReport(text);
    renderFormatReport(formatReport);
    resultsZone.classList.remove('hidden');
    Lib() && Lib().onResult(lastResult, lastFormatItems, lastFormatNote);
  }
}

/* 刷新页面后继续等待之前提交的任务 */
async function resumeJob(jobId, text){
  showError('');
  analyzeBtn.disabled = true;
  progressZone.classList.remove('hidden');
  progressLabel.textContent = '正在接着等待刷新前提交的检测…';
  try{
    const job = await api('/v1/jobs/' + jobId, { headers: authHeaders() });
    await finishJob(job, text);
  }catch(err){
    showError(err.status === 404 || err.jobGone ? '刷新前提交的检测已找不到（服务重启过或已超过 1 小时），请重新检测。' : err.message);
    Lib() && Lib().onJobStarted(null);
  }finally{
    progressZone.classList.add('hidden');
    analyzeBtn.disabled = false;
  }
}

/* ---------------- 渲染 ---------------- */
const SIG_NAME = { fastdetect:'Fast-DetectGPT', binoculars:'Binoculars', classifier:'MPU 分类器' };

function renderResult(res){
  const s = res.summary;
  sealNum.textContent = s.ai_rate == null ? '—' : pct(s.ai_rate);
  $('sumRate').textContent = pct(s.ai_rate);
  $('sumMean').textContent = pct(s.mean_prob);
  $('sumHigh').textContent = pct(s.high_rate);
  const L = s.segments_by_level || {};
  const flagged = (L.high||0) + (L.mid||0) + (L.light||0);
  const counted = flagged + (L.low||0);
  $('notesList').innerHTML = (s.reliability_notes || []).map(n=>`<li>${escapeHtml(n)}</li>`).join('');
  $('sumFlagged').textContent = `${flagged} / ${counted}`;
  $('sumExcluded').textContent = s.excluded_chars.toLocaleString();
  const methods = Object.entries(s.methods).filter(([,v])=>v).map(([k])=>SIG_NAME[k]).join('、') || '无';
  $('calibLine').innerHTML =
    `使用方法：${escapeHtml(methods)} · 判定阈值 ${pct(s.threshold)} · ` +
    (s.calibrated ? `<b>已校准</b>：${escapeHtml(s.calibration_note||'')}` : '<b>未校准</b>：阈值为经验值，结果只宜作相对参考') +
    ` · 高度 ${pct(s.high_rate)} / 中度 ${pct(s.mid_rate)} / 轻度 ${pct(s.light_rate)}` +
    ` · 相邻段落平滑 ${s.smoothing}` +
    ` · 用时 ${s.elapsed_sec} 秒` +
    (s.excluded_reference_segments || s.excluded_quotation_segments
      ? ` · 未计入：参考文献 ${s.excluded_reference_segments} 段、引文为主 ${s.excluded_quotation_segments} 段` : '');

  paragraphList.innerHTML = '';
  res.segments.forEach(seg=>{
    const div = document.createElement('div');
    const lvl = seg.kind !== 'body' ? 'none' : seg.level;
    div.className = `para-item level-${lvl}`;
    div.dataset.flagged = (lvl === 'high' || lvl === 'mid' || lvl === 'light') ? '1' : '0';
    const preview = seg.text.length > 280 ? seg.text.slice(0,280) + '……' : seg.text;
    const sig = Object.entries(seg.signals || {}).filter(([,v])=>v!=null)
      .map(([k,v])=>`<span class="para-tag">${SIG_NAME[k]} ${pct(v)}</span>`).join('');
    const raw = seg.raw || {};
    const rawBits = [
      raw.fastdetect!=null ? `Fast-DetectGPT 曲率 ${raw.fastdetect.toFixed(2)}` : '',
      raw.binoculars!=null ? `Binoculars 分数 ${raw.binoculars.toFixed(3)}` : '',
      raw.classifier!=null ? `分类器 ${pct(raw.classifier)}` : '',
      raw.ppl!=null ? `困惑度 ${Math.exp(raw.ppl).toFixed(1)}` : '',
      raw.lrr!=null ? `LRR ${raw.lrr.toFixed(3)}` : '',
      raw.log_rank!=null ? `平均对数名次 ${raw.log_rank.toFixed(2)}` : '',
      raw.top10!=null ? `前 10 名占比 ${pct(raw.top10)}` : '',
      raw.entropy!=null ? `预测熵 ${raw.entropy.toFixed(2)}` : '',
      raw.lp_burstiness!=null ? `困惑度波动 ${raw.lp_burstiness.toFixed(3)}` : '',
      (seg.prob_unsmoothed!=null && seg.prob!=null && Math.abs(seg.prob_unsmoothed-seg.prob)>0.005) ? `平滑前 ${pct(seg.prob_unsmoothed)}` : '',
      seg.style ? `句长变异 ${seg.style.sentence_len_cv}` : '',
      seg.style && seg.style.template_phrases.length ? `套话：${seg.style.template_phrases.join('、')}` : ''
    ].filter(Boolean).join(' · ');
    const kindTag = seg.kind === 'reference' ? '<span class="para-tag">参考文献 · 不计入</span>'
      : seg.kind === 'quotation' ? `<span class="para-tag">引文为主 · 不计入（${escapeHtml(seg.notes.join('；'))}）${seg.ref_prob!=null ? ' · 参考值 '+pct(seg.ref_prob) : ''}</span>`
      : (seg.notes && seg.notes.length) ? `<span class="para-tag">${escapeHtml(seg.notes.join('；'))}</span>` : '';
    div.innerHTML = `
      <div class="para-badge" ${seg.prob==null && seg.ref_prob!=null ? 'title="参考值：该段不计入 AI 率" style="opacity:.55;border-style:dashed"' : ''}>${seg.prob!=null ? Math.round(seg.prob*100) : seg.ref_prob!=null ? Math.round(seg.ref_prob*100) : '—'}</div>
      <div class="para-body">
        <div class="para-text">${escapeHtml(preview)}</div>
        <div class="para-tags">
          ${seg.label ? `<span class="para-tag strong">${seg.label}</span>` : ''}
          ${kindTag}${sig}
          <span class="para-tag">第 ${seg.index+1} 段 · ${seg.chars} 字</span>
        </div>
        ${rawBits ? `<details class="para-raw"><summary>原始分数</summary><p>${escapeHtml(rawBits)}</p></details>` : ''}
      </div>`;
    paragraphList.appendChild(div);
  });
  applyFilter();
}

function applyFilter(){
  const only = $('onlyFlagged').checked;
  paragraphList.querySelectorAll('.para-item').forEach(el=>{
    el.style.display = (only && el.dataset.flagged !== '1') ? 'none' : '';
  });
}
$('onlyFlagged').addEventListener('change', applyFilter);

/* ---------------- 导出 ---------------- */
exportBtn.addEventListener('click', ()=>{
  if(!lastResult) return;
  const s = lastResult.summary;
  let out = `审读 · AI 文本检测报告\n生成时间：${new Date().toLocaleString('zh-CN')}\n来源：${currentSource}\n`;
  out += `总字数：${s.total_chars} · 计入字数：${s.counted_chars} · 未计入：${s.excluded_chars}\n`;
  out += `AI 率：${pct(s.ai_rate)}（高度 ${pct(s.high_rate)} / 中度 ${pct(s.mid_rate)} / 轻度 ${pct(s.light_rate)}） · 平均 AI 概率：${pct(s.mean_prob)} · 阈值：${pct(s.threshold)} · ${s.calibrated ? '已校准' : '未校准'}\n`;
  (s.reliability_notes || []).forEach(n=>{ out += `提示：${n}\n`; });
  out += `方法：Fast-DetectGPT、Binoculars（Qwen2.5 打分）、MPU 中文分类器；模式：${s.mode === 'fast' ? '快速（抽样）' : '完整'}\n`;
  out += `\n【说明】任何 AI 检测都有误判，本报告只供作者自查，不能作为学术不端判定依据。\n`;
  out += `\n${'='.repeat(60)}\n分段结果\n${'='.repeat(60)}\n`;
  lastResult.segments.forEach(seg=>{
    const tag = seg.kind === 'reference' ? '参考文献，不计入' : seg.kind === 'quotation' ? '引文为主，不计入' : (seg.label || '未达阈值');
    const probStr = seg.prob!=null ? pct(seg.prob) : seg.ref_prob!=null ? `${pct(seg.ref_prob)}（参考值，不计入）` : '—';
    out += `\n[第 ${seg.index+1} 段 | AI 概率 ${probStr} | ${tag} | ${seg.chars} 字]\n`;
    const sg = seg.signals || {}, r = seg.raw || {};
    const bits = [
      sg.classifier!=null ? `分类器 ${pct(sg.classifier)}` : (r.classifier!=null ? `分类器 ${pct(r.classifier)}` : ''),
      r.fastdetect!=null ? `Fast-DetectGPT 曲率 ${r.fastdetect.toFixed(3)}` : '',
      r.binoculars!=null ? `Binoculars ${r.binoculars.toFixed(3)}` : '',
      r.ppl!=null ? `困惑度 ${Math.exp(r.ppl).toFixed(1)}` : '',
      r.lrr!=null ? `LRR ${r.lrr.toFixed(3)}` : '',
      r.top10!=null ? `前10名占比 ${pct(r.top10)}` : '',
      r.lp_burstiness!=null ? `困惑度波动 ${r.lp_burstiness.toFixed(3)}` : '',
      seg.style ? `句长变异 ${seg.style.sentence_len_cv}` : '',
      (seg.notes && seg.notes.length) ? `备注：${seg.notes.join('；')}` : ''
    ].filter(Boolean).join(' · ');
    if(bits) out += `指标：${bits}\n`;
    out += `${seg.text}\n`;
  });
  out += formatItemsToText(lastFormatItems);
  const blob = new Blob([out], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `审读报告_${new Date().toISOString().slice(0,10)}.txt`; a.click();
  URL.revokeObjectURL(url);
});

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
  renderFormatItems(items, note);
}

/* 按检查项渲染（也用于从本地文稿库恢复已保存的格式检查结果） */
function renderFormatItems(items, note){
  note = note || '';
  lastFormatItems = items;
  lastFormatNote = note;

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

