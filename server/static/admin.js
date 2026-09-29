const $ = (id)=>document.getElementById(id);
let token = '';
const esc = (s)=>String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sleep = (ms)=>new Promise(r=>setTimeout(r, ms));

async function call(path, opts = {}){
  const r = await fetch(path, { ...opts, headers: { 'Content-Type':'application/json', 'X-Admin-Token': token, ...(opts.headers||{}) } });
  let data; try{ data = await r.json(); }catch(e){ data = { message: 'HTTP ' + r.status }; }
  if(!r.ok) throw new Error(data.message || ('HTTP ' + r.status));
  return data;
}
function msg(el, text, ok){ el.textContent = text; el.className = 'msg ' + (ok ? 'good' : 'bad'); }

$('loginBtn').addEventListener('click', async ()=>{
  token = $('adminToken').value;
  try{
    await call('/admin/api/usage');
    $('sheet').classList.remove('locked');
    msg($('loginMsg'), '已登录。', true);
    renderUsage();
  }catch(e){
    $('sheet').classList.add('locked');
    msg($('loginMsg'), e.message, false);
  }
});
$('adminToken').addEventListener('keydown', e=>{ if(e.key === 'Enter') $('loginBtn').click(); });

$('issueBtn').addEventListener('click', async ()=>{
  const out = $('issueOut');
  try{
    const k = await call('/admin/api/keys', { method:'POST', body: JSON.stringify({
      name: $('keyName').value, days: parseInt($('keyDays').value||'0',10), daily_chars: parseInt($('keyQuota').value||'0',10) }) });
    out.innerHTML = `<p class="msg good">已生成（编号 <b>${esc(k.id)}</b>${k.expires ? '，到期 ' + esc(k.expires.slice(0,10)) : '，永久有效'}，每日额度 ${k.daily_chars ? k.daily_chars.toLocaleString() + ' 字' : '不限'}）。请立即复制保存：</p>
      <div class="out" id="newKey">${esc(k.key)}</div>
      <button class="ghost-btn" id="copyKey" style="margin-top:8px">复制</button>`;
    $('copyKey').addEventListener('click', async ()=>{
      try{ await navigator.clipboard.writeText(k.key); $('copyKey').textContent = '已复制'; }catch(e){ $('copyKey').textContent = '复制失败，请手动选择'; }
    });
  }catch(e){ out.innerHTML = `<p class="msg bad">${esc(e.message)}</p>`; }
});

async function renderUsage(){
  const out = $('usageOut');
  try{
    const d = await call('/admin/api/usage');
    const rows = Object.entries(d.usage);
    out.innerHTML = rows.length
      ? `<table class="usage"><tr><th>编号</th><th>名称</th><th>日期(UTC)</th><th>今日字数</th><th>今日请求</th></tr>${
          rows.map(([id,u])=>`<tr><td>${esc(id)}</td><td>${esc(u.name||'')}</td><td>${esc(u.day)}</td><td>${u.chars.toLocaleString()}</td><td>${u.requests}</td></tr>`).join('')}</table>`
      : '<p class="msg">自上次启动以来还没有调用记录。</p>';
    if(d.revoked_env.length) out.innerHTML += `<p class="msg">REVOKED_KEY_IDS 中已作废：${d.revoked_env.map(esc).join('、')}</p>`;
  }catch(e){ out.innerHTML = `<p class="msg bad">${esc(e.message)}</p>`; }
}
$('usageBtn').addEventListener('click', renderUsage);

$('revokeBtn').addEventListener('click', async ()=>{
  const id = $('revokeId').value.trim();
  if(!id) return;
  try{
    const d = await call('/admin/api/revoke', { method:'POST', body: JSON.stringify({ id }) });
    $('usageOut').innerHTML = `<p class="msg good">${esc(d.message)}</p>`;
  }catch(e){ $('usageOut').innerHTML = `<p class="msg bad">${esc(e.message)}</p>`; }
});

async function readFiles(input){
  const texts = [];
  for(const f of input.files){
    const ext = f.name.split('.').pop().toLowerCase();
    if(ext === 'docx'){
      texts.push((await mammoth.extractRawText({ arrayBuffer: await f.arrayBuffer() })).value);
    } else {
      texts.push(await f.text());
    }
  }
  return texts;
}
function splitSamples(s){ return s.split(/\n\s*===+\s*\n/).map(x=>x.trim()).filter(x=>x.length >= 16); }

$('calBtn').addEventListener('click', async ()=>{
  const m = $('calMsg'), out = $('calOut');
  out.innerHTML = '';
  try{
    const human = splitSamples($('humanText').value).concat(await readFiles($('humanFiles')));
    const ai = splitSamples($('aiText').value).concat(await readFiles($('aiFiles')));
    const builtin = $('calBuiltin').checked;
    if(builtin ? !(human.length || ai.length) : !(human.length && ai.length))
      throw new Error(builtin ? '至少提供一段样本。' : '不合并内置数据时，两边都需要提供样本。');
    $('calBtn').disabled = true;
    let job = await call('/admin/api/calibrate', { method:'POST', body: JSON.stringify({ human, ai, target_fpr: parseFloat($('targetFpr').value), profile: $('calProfile').value, include_builtin: builtin }) });
    while(job.status === 'queued' || job.status === 'running'){
      msg(m, job.status === 'queued' ? '排队中…' : `打分中：${job.done} / ${job.total} 段${job.eta_sec ? '，剩余约 ' + Math.ceil(job.eta_sec/60) + ' 分钟' : ''}`, true);
      await sleep(2000);
      job = await call('/admin/api/jobs/' + job.id);
    }
    if(job.status === 'error') throw new Error(job.error);
    const { calibration, report } = job.result;
    const au = Object.entries(report.auroc).map(([k,v])=>`${k} ${v}`).join('，');
    msg(m, '校准完成。', true);
    out.innerHTML = `
      <p class="msg">文体：<b>${esc(report.profile_name || '现代汉语')}</b>${report.skipped_other_register_segments ? `（另有 ${report.skipped_other_register_segments} 段属于其他文体，未参与）` : ''}。样本：人写 ${report.n_human} 段、AI ${report.n_ai} 段。区分能力（AUROC，1 = 完美，0.5 = 随机）：${esc(au)}；综合 ${report.combined_auroc}。<br>
      ${report.builtin_samples ? `已合并内置公开数据 ${report.builtin_samples} 段；你的样本：人写 ${report.user_human} 段、AI ${report.user_ai} 段。<br>` : ''}
      ${report.user_samples ? `<b>在你自己的样本上</b>（交叉验证，每次都用没参与拟合的样本算）：${report.user_samples.n_ai ? `AI 识别出 ${(report.user_samples.ai_caught_rate*100).toFixed(0)}%` : ''}${report.user_samples.n_ai && report.user_samples.n_human ? '，' : ''}${report.user_samples.n_human ? `人写被误判 ${(report.user_samples.human_flagged_rate*100).toFixed(0)}%` : ''}。<br>` : ''}
      参与组合的特征：${esc((report.features_used||[]).join('、') || '三个主信号')}${report.cross_validated ? '（已做 5 折交叉验证）' : ''}。<br>
      阈值 ${report.threshold}：校准样本中人写段落被误判的比例 ${(report.human_flagged_rate*100).toFixed(1)}%，AI 段落被识别出的比例 ${(report.ai_caught_rate*100).toFixed(1)}%。<br>${esc(report.note)}</p>
      <div class="out" id="calJson">${esc(JSON.stringify(calibration))}</div>
      <div class="form-row" style="margin-top:8px">
        <button class="primary-btn" id="applyBtn">立即启用</button>
        <button class="ghost-btn" id="copyCal">复制 JSON</button>
      </div>
      <p class="msg">要永久保存：先点“立即启用”（会与其他文体的现有校准合并），再点“复制 JSON”，在 GitHub 仓库 Settings → Secrets and variables → Actions 的 <b>Variables</b> 里新建或修改 <code>CALIBRATION_JSON</code>，粘贴进去，然后重新部署。</p>`;
    let merged = null;
    $('applyBtn').addEventListener('click', async ()=>{
      try{
        const d = await call('/admin/api/calibration', { method:'POST', body: JSON.stringify({ calibration }) });
        merged = d.calibration || null;
        if(merged) $('calJson').textContent = JSON.stringify(merged);
        msg(m, d.message, true);
      }
      catch(e){ msg(m, e.message, false); }
    });
    $('copyCal').addEventListener('click', async ()=>{
      try{ await navigator.clipboard.writeText(JSON.stringify(merged || calibration)); $('copyCal').textContent = '已复制'; }catch(e){}
    });
  }catch(e){
    msg(m, e.message, false);
  }finally{
    $('calBtn').disabled = false;
  }
});


/* 导入首页上"这段是 AI / 这段是人写"的标注（保存在本浏览器的 localStorage） */
$('importLabels').addEventListener('click', ()=>{
  let m = {};
  try{ m = JSON.parse(localStorage.getItem('shendu_labels') || '{}'); }catch(e){}
  const items = Object.values(m);
  if(!items.length){ msg($('calMsg'), '本浏览器里还没有标注。请先在检测结果的段落下方点"这段是 AI / 这段是人写"。', false); return; }
  const prof = $('calProfile').value;
  const byReg = {};
  items.forEach(x=>{ byReg[x.register] = (byReg[x.register] || 0) + 1; });
  const reg = prof !== 'auto' ? prof : Object.keys(byReg).sort((a,b)=>byReg[b]-byReg[a])[0];
  const pick = items.filter(x=> (x.register === reg) || (reg === 'zh' && x.register === 'zh'));
  const join = (arr)=> arr.map(x=>x.text).join('\n===\n');
  const hu = pick.filter(x=>x.label==='human'), ai = pick.filter(x=>x.label==='ai');
  $('humanText').value = [$('humanText').value.trim(), join(hu)].filter(Boolean).join('\n===\n');
  $('aiText').value = [$('aiText').value.trim(), join(ai)].filter(Boolean).join('\n===\n');
  $('calProfile').value = reg;
  const names = { zh:'现代汉语', zh_classical:'文言', zh_poetry:'诗词', en:'英文' };
  const others = Object.entries(byReg).filter(([k])=>k!==reg).map(([k,v])=>`${names[k]||k} ${v} 段`).join('、');
  msg($('calMsg'), `已导入${names[reg]||reg}的标注：人写 ${hu.length} 段、AI ${ai.length} 段。` +
      (others ? `另有 ${others}，请切换"校准哪种文体"后再导入一次、分别校准。` : '') + '勾选"与内置公开数据合并"后点"开始校准"。', true);
});
