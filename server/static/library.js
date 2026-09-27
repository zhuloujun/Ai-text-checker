/* ============================================================
   本地文稿库：把粘贴的文字、上传的文档、检测结果保存在本浏览器（IndexedDB）
   - 只存在这台电脑的这个浏览器里，不会上传到服务器
   - 刷新页面后自动恢复上次的文稿和结果；检测进行中刷新，会接着等结果
   依赖 app.js 里的函数：handleFile、usePaste、renderResult、renderFormatItems、resumeJob 等
   ============================================================ */
(function(){
  const DB_NAME = 'shendu-local', STORE = 'docs', DB_VER = 1;
  const CUR_KEY = 'shendu_current_doc', AUTO_KEY = 'shendu_autosave';
  const DRAFT_ID = 'paste-draft';
  const MAX_DOCS = 50;

  const el = (id)=>document.getElementById(id);
  const listEl = el('libList'), statusEl = el('libStatus'), autoEl = el('libAuto');
  let db = null, currentId = null, draftTimer = null, available = true;

  /* ---------- 存储小工具（任何一步失败都只影响本功能，不影响检测） ---------- */
  function openDb(){
    return new Promise((resolve, reject)=>{
      if(!('indexedDB' in window)) return reject(new Error('浏览器不支持 IndexedDB'));
      const req = indexedDB.open(DB_NAME, DB_VER);
      req.onupgradeneeded = ()=>{
        const d = req.result;
        if(!d.objectStoreNames.contains(STORE)){
          const st = d.createObjectStore(STORE, { keyPath: 'id' });
          st.createIndex('updatedAt', 'updatedAt');
        }
      };
      req.onsuccess = ()=> resolve(req.result);
      req.onerror = ()=> reject(req.error);
    });
  }
  function tx(mode, fn){
    return new Promise((resolve, reject)=>{
      const t = db.transaction(STORE, mode);
      const st = t.objectStore(STORE);
      let out;
      Promise.resolve(fn(st)).then(v=>{ out = v; });
      t.oncomplete = ()=> resolve(out);
      t.onerror = ()=> reject(t.error);
      t.onabort = ()=> reject(t.error || new Error('存储被中止（可能空间不足）'));
    });
  }
  const reqP = (r)=> new Promise((res, rej)=>{ r.onsuccess = ()=>res(r.result); r.onerror = ()=>rej(r.error); });
  const get = (id)=> db ? tx('readonly', st=> reqP(st.get(id))) : Promise.resolve(null);
  const put = (doc)=> db ? tx('readwrite', st=>{ st.put(doc); }) : Promise.resolve();
  const del = (id)=> db ? tx('readwrite', st=>{ st.delete(id); }) : Promise.resolve();
  const all = ()=> db ? tx('readonly', st=> reqP(st.getAll())) : Promise.resolve([]);

  function lsGet(k){ try{ return localStorage.getItem(k); }catch(e){ return null; } }
  function lsSet(k, v){ try{ v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); }catch(e){} }

  const autosave = ()=> autoEl ? autoEl.checked : true;
  function setCurrent(id){ currentId = id; lsSet(CUR_KEY, id); highlight(); }

  function fmtSize(n){
    if(n > 1024*1024) return (n/1024/1024).toFixed(1) + ' MB';
    if(n > 1024) return Math.round(n/1024) + ' KB';
    return n + ' B';
  }
  function fmtTime(t){
    const d = new Date(t);
    const p = (x)=> String(x).padStart(2,'0');
    return `${d.getMonth()+1}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }
  function esc(s){ return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function fileId(file){ return 'f-' + [file.name, file.size, file.lastModified || 0].join('|'); }
  function flash(msg, bad){ if(statusEl){ statusEl.textContent = msg; statusEl.classList.toggle('bad', !!bad); } }

  /* ---------- 列表 ---------- */
  async function render(){
    if(!listEl) return;
    if(!available){
      listEl.innerHTML = '<li class="lib-empty">此浏览器不允许网页保存数据（可能是无痕模式或关闭了网站数据），本地文稿库不可用；检测功能不受影响。</li>';
      return;
    }
    let docs = (await all()).filter(d=> d.id !== DRAFT_ID || (d.text && d.text.trim()));
    docs.sort((a,b)=> b.updatedAt - a.updatedAt);
    if(!docs.length){
      listEl.innerHTML = '<li class="lib-empty">还没有保存的文稿。粘贴文字或上传文档后会自动保存在这里。</li>';
    } else {
      listEl.innerHTML = docs.map(d=>{
        const s = d.lastResult && d.lastResult.summary;
        const badge = d.pendingJobId ? '<span class="lib-badge wait">检测中</span>'
          : s ? `<span class="lib-badge">AI 率 ${s.ai_rate==null?'—':(Math.round(s.ai_rate*1000)/10)+'%'}</span>` : '';
        const kind = d.kind === 'file' ? '文档' : (d.id === DRAFT_ID ? '草稿' : '文字');
        return `<li class="lib-item" data-id="${esc(d.id)}">
          <button class="lib-open" data-act="open" title="打开">
            <span class="lib-name">${esc(d.name)}</span>
            <span class="lib-meta">${kind} · ${(d.chars||0).toLocaleString()} 字${d.size ? ' · ' + fmtSize(d.size) : ''} · ${fmtTime(d.updatedAt)}</span>
          </button>
          ${badge}
          <button class="link-btn" data-act="del" title="从本浏览器删除">删除</button>
        </li>`;
      }).join('');
    }
    highlight();
    try{
      if(navigator.storage && navigator.storage.estimate){
        const e = await navigator.storage.estimate();
        el('libUsage').textContent = `已用约 ${fmtSize(e.usage||0)}`;
      }
    }catch(e){}
  }
  function highlight(){
    if(!listEl) return;
    listEl.querySelectorAll('.lib-item').forEach(li=> li.classList.toggle('current', li.dataset.id === currentId));
  }

  listEl && listEl.addEventListener('click', async (e)=>{
    const btn = e.target.closest('button');
    if(!btn) return;
    const id = btn.closest('.lib-item').dataset.id;
    if(btn.dataset.act === 'open') await openDoc(id, true);
    if(btn.dataset.act === 'del'){
      const d = await get(id);
      if(!d || !confirm(`从本浏览器删除"${d.name}"？（服务器上本来就不保存）`)) return;
      await del(id);
      if(currentId === id) setCurrent(null);
      render();
    }
  });

  el('libSavePaste') && el('libSavePaste').addEventListener('click', async ()=>{
    const text = el('pasteArea').value;
    if(!text.trim()){ flash('粘贴框是空的。', true); return; }
    const first = text.trim().replace(/\s+/g,' ').slice(0, 18);
    const doc = { id: 'p-' + Date.now(), kind: 'paste', name: first + (text.trim().length > 18 ? '…' : ''),
                  text, chars: text.length, createdAt: Date.now(), updatedAt: Date.now() };
    try{ await put(doc); setCurrent(doc.id); flash('已另存为一篇文稿。'); render(); }
    catch(err){ flash('保存失败：' + err.message, true); }
  });

  el('libClear') && el('libClear').addEventListener('click', async ()=>{
    if(!confirm('清空本浏览器里保存的全部文稿和检测结果？此操作不能撤销。')) return;
    try{ await tx('readwrite', st=>{ st.clear(); }); }catch(e){}
    setCurrent(null);
    flash('已清空。');
    render();
  });

  autoEl && autoEl.addEventListener('change', ()=>{
    lsSet(AUTO_KEY, autoEl.checked ? '1' : '0');
    flash(autoEl.checked ? '已开启自动保存。' : '已关闭自动保存：新粘贴或上传的内容不会再保存（已保存的不受影响）。');
  });

  /* ---------- 打开一篇文稿（恢复文字 / 文件和上次结果） ---------- */
  async function openDoc(id, userAction){
    const d = await get(id);
    if(!d) return false;
    setCurrent(id);
    const tab = d.kind === 'file' ? 'upload' : 'paste';
    document.querySelector(`.tab-btn[data-tab=${tab}]`).click();
    if(d.kind === 'file' && d.blob){
      const file = new File([d.blob], d.name, { type: d.blob.type || '', lastModified: d.lastModified || Date.now() });
      await handleFile(file, { fromLibrary: true });
    } else {
      el('pasteArea').value = d.text || '';
      usePaste();
    }
    // 恢复上次结果
    const rz = el('resultsZone');
    if(d.lastResult){
      lastResult = d.lastResult;
      renderResult(d.lastResult);
      renderFormatItems(d.lastFormatItems || [], d.lastFormatNote || '');
      rz.classList.remove('hidden');
    } else {
      rz.classList.add('hidden');
    }
    if(d.pendingJobId){
      resumeJob(d.pendingJobId, currentText.trim());
    }
    if(userAction){
      flash(`已打开"${d.name}"${d.lastResult ? '，并恢复了上次的检测结果' : ''}。`);
      window.scrollTo({ top: el('inputZone').offsetTop - 10, behavior: 'smooth' });
    }
    return true;
  }

  /* ---------- app.js 调用的钩子 ---------- */
  const hooks = {
    onPasteInput(text){
      if(!db || !autosave()) return;
      clearTimeout(draftTimer);
      draftTimer = setTimeout(async ()=>{
        const now = Date.now();
        const old = await get(DRAFT_ID).catch(()=>null);
        const doc = Object.assign(old || { id: DRAFT_ID, kind: 'paste', createdAt: now }, {
          name: '粘贴草稿（自动保存）', text, chars: text.length, updatedAt: now });
        // 草稿内容变了，旧结果不再对应
        if(old && old.text !== text){ delete doc.lastResult; delete doc.lastFormatItems; delete doc.pendingJobId; }
        try{ await put(doc); setCurrent(DRAFT_ID); flash('草稿已自动保存。'); render(); }
        catch(err){ flash('自动保存失败：' + err.message, true); }
      }, 800);
    },
    async onFileParsed(file, text){
      if(!db || !autosave()) return;
      const now = Date.now();
      const id = fileId(file);
      const old = await get(id).catch(()=>null);
      const doc = Object.assign(old || { id, createdAt: now }, {
        kind: 'file', name: file.name, blob: file, size: file.size, lastModified: file.lastModified,
        chars: text.length, updatedAt: now });
      try{
        await put(doc);
        setCurrent(id);
        flash(`"${file.name}"已保存在本浏览器。`);
        // 超出数量上限时删掉最旧的（草稿除外）
        const docs = (await all()).filter(d=>d.id !== DRAFT_ID).sort((a,b)=>b.updatedAt-a.updatedAt);
        for(const d of docs.slice(MAX_DOCS)) await del(d.id);
        render();
      }catch(err){ flash('保存文档失败（可能是浏览器空间不足）：' + err.message, true); }
    },
    onFileCleared(){ if(currentId && currentId.startsWith('f-')) setCurrent(null); },
    async onJobStarted(jobId){
      if(!db || !currentId) return;
      const d = await get(currentId).catch(()=>null);
      if(!d) return;
      if(jobId) d.pendingJobId = jobId; else delete d.pendingJobId;
      d.updatedAt = Date.now();
      await put(d).catch(()=>{});
      render();
    },
    async onResult(result, formatItems, formatNote){
      if(!db || !currentId) return;
      const d = await get(currentId).catch(()=>null);
      if(!d) return;
      d.lastResult = result;
      d.lastFormatItems = formatItems;
      d.lastFormatNote = formatNote;
      d.resultAt = Date.now();
      delete d.pendingJobId;
      d.updatedAt = Date.now();
      try{ await put(d); flash('检测结果已保存在本浏览器，刷新页面也不会丢失。'); }
      catch(err){ flash('结果保存失败：' + err.message, true); }
      render();
    }
  };

  /* ---------- 启动：打开数据库，恢复上次的文稿 ---------- */
  async function init(){
    if(autoEl) autoEl.checked = lsGet(AUTO_KEY) !== '0';
    try{
      db = await openDb();
      window.Library = hooks;
    }catch(e){
      available = false;
    }
    await render();
    if(!db) return;
    const last = lsGet(CUR_KEY);
    let restored = false;
    if(last) restored = await openDoc(last, false).catch(()=>false);
    if(!restored){
      const draft = await get(DRAFT_ID).catch(()=>null);
      if(draft && draft.text){ await openDoc(DRAFT_ID, false).catch(()=>{}); restored = true; }
    }
    if(restored) flash('已恢复上次的文稿' + (lastResult ? '和检测结果' : '') + '。');
  }
  init();
})();
