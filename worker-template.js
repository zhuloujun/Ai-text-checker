/**
 * 审读 · 文本特征检测 —— Cloudflare Worker 单文件版
 *
 * 这一个文件同时负责：
 *   1. 设置了变量 BACKEND_URL 时：把网站转发到完整版检测服务（Modal），轻量版放在 /lite/
 *      没设置时：直接提供轻量版网页
 *   2. /api/detect：把轻量版的检测请求转发给 GPTZero 官方接口
 *
 * 部署：Cloudflare Dashboard → Workers & Pages → 创建 → Worker（从 Hello World 开始）
 *       → 编辑代码 → 把本文件全部内容粘贴进去替换 → 部署。详见 README。
 *
 * API Key 的两种提供方式：
 *   A. 用户在网页上填写自己的 Key → 随请求头 X-User-Api-Key 发来，本 Worker 只转发，不保存。
 *   B. 在 Worker 的 Settings → Variables and Secrets 设置 Secret：
 *        GPTZERO_API_KEY  你的 GPTZero Key
 *        ACCESS_PASSWORD  访问密码（必须设置，否则拒绝使用 B 方式，防止别人用掉你的额度）
 *
 * 免费版 Worker 每次请求只有 10 毫秒 CPU 时间；等待 GPTZero 返回的时间不算 CPU 时间，
 * 本文件做的只是转发，远低于这个限制。
 *
 * 注意：此文件由 build.mjs 生成，请修改 pages-static/ 下的源文件后重新生成。
 */

const INDEX_HTML = __INDEX_HTML__;
const STYLE_CSS = __STYLE_CSS__;
const APP_JS = __APP_JS__;

const GPTZERO_ENDPOINT = 'https://api.gptzero.me/v2/predict/text';
const MAX_CHARS_PER_CALL = 48000; // GPTZero 单次约 5 万字符上限，留出余量

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // 轻量版的 GPTZero 转发接口（两种模式下都保留）
    if (url.pathname === '/api/detect') {
      if (request.method !== 'POST') {
        return json({ error: 'method_not_allowed', message: '此接口只接受 POST 请求' }, 405);
      }
      return handleDetect(request, env);
    }

    // 设置了 BACKEND_URL（完整版检测服务地址）时：
    //   /lite/...  → 轻量版（浏览器本地检测）
    //   其他路径    → 原样转发给完整版（Modal 上的检测服务）
    if (env.BACKEND_URL) {
      if (url.pathname === '/lite') return Response.redirect(url.origin + '/lite/', 301);
      if (url.pathname.startsWith('/lite/')) return serveLite(request, url.pathname.slice('/lite'.length));
      return proxy(request, env, url);
    }
    return serveLite(request, url.pathname);
  }
};

function serveLite(request, path) {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return new Response('Method Not Allowed', { status: 405 });
  }
  switch (path) {
    case '/':
    case '/index.html':
      return asset(INDEX_HTML, 'text/html; charset=utf-8');
    case '/style.css':
      return asset(STYLE_CSS, 'text/css; charset=utf-8');
    case '/app.js':
      return asset(APP_JS, 'application/javascript; charset=utf-8');
    default:
      return new Response('Not Found', { status: 404 });
  }
}

async function proxy(request, env, url) {
  const backend = env.BACKEND_URL.replace(/\/+$/, '');
  const target = backend + url.pathname + url.search;
  const headers = new Headers(request.headers);
  headers.delete('host');
  headers.set('X-Forwarded-Host', url.host);
  headers.set('X-Forwarded-Proto', 'https');
  const ip = request.headers.get('CF-Connecting-IP');
  if (ip) headers.set('X-Forwarded-For', ip);
  const init = { method: request.method, headers, redirect: 'manual' };
  if (request.method !== 'GET' && request.method !== 'HEAD') init.body = request.body;

  let resp;
  try {
    resp = await fetch(target, init);
  } catch (e) {
    return json({ error: 'backend_unreachable',
      message: '检测服务暂时连不上，可能正在启动（闲置后首次访问需要 1–3 分钟），请稍后再试。' }, 502);
  }
  const out = new Response(resp.body, resp);
  // 后端返回的跳转地址改回当前域名，避免把用户带到 modal.run
  const loc = out.headers.get('Location');
  if (loc && loc.startsWith(backend)) out.headers.set('Location', url.origin + loc.slice(backend.length));
  out.headers.set('X-Served-Via', 'cloudflare-worker');
  return out;
}

async function handleDetect(request, env) {
  const userKey = (request.headers.get('X-User-Api-Key') || '').trim();
  let apiKey = userKey;

  if (!apiKey) {
    if (!env.GPTZERO_API_KEY) {
      return json({
        error: 'not_configured',
        message: '没有可用的 API Key：请在网页"官方 API 增强检测"里填写你自己的 GPTZero Key，或在 Worker 的 Settings → Variables and Secrets 中设置 GPTZERO_API_KEY。'
      }, 400);
    }
    if (!env.ACCESS_PASSWORD) {
      return json({
        error: 'password_not_set',
        message: 'Worker 里设置了 GPTZERO_API_KEY，但没有设置 ACCESS_PASSWORD。为防止别人通过你的网址用掉你的额度，请在 Settings → Variables and Secrets 里再加一个 Secret：ACCESS_PASSWORD，然后在网页上填写这个密码。'
      }, 403);
    }
    const pw = request.headers.get('X-Access-Password') || '';
    if (!(await safeEqual(pw, env.ACCESS_PASSWORD))) {
      return json({ error: 'forbidden', message: '访问密码不正确（或未填写）。请在网页"官方 API 增强检测"里填写 ACCESS_PASSWORD 对应的密码。' }, 403);
    }
    apiKey = env.GPTZERO_API_KEY;
  }

  let body;
  try {
    body = await request.json();
  } catch (e) {
    return json({ error: 'bad_request', message: '请求内容不是合法的 JSON' }, 400);
  }
  const text = String(body.text || '');
  if (!text.trim()) return json({ error: 'bad_request', message: '文本为空' }, 400);
  if (text.length > MAX_CHARS_PER_CALL) {
    return json({ error: 'too_long', message: `单次请求超过 ${MAX_CHARS_PER_CALL} 字符，请分段发送。` }, 413);
  }

  let upstream, data;
  try {
    upstream = await fetch(GPTZERO_ENDPOINT, {
      method: 'POST',
      headers: { 'Accept': 'application/json', 'Content-Type': 'application/json', 'x-api-key': apiKey },
      body: JSON.stringify({ document: text, multilingual: true })
    });
    data = await upstream.json().catch(() => ({}));
  } catch (err) {
    return json({ error: 'network_error', message: '连接 GPTZero 失败：' + err.message }, 502);
  }

  if (!upstream.ok) {
    const hint = upstream.status === 401 || upstream.status === 403
      ? 'Key 无效，或你的 GPTZero 套餐不含 API 权限（网页免费版不含 API）。'
      : upstream.status === 429 ? '请求太频繁或额度已用完，请稍后再试或检查套餐额度。' : '';
    return json({
      error: 'upstream_error',
      message: `GPTZero 返回错误（HTTP ${upstream.status}）。${hint}`,
      detail: data && (data.error || data.message) || null
    }, upstream.status === 429 ? 429 : 502);
  }

  const doc = (data.documents && data.documents[0]) || data;
  return json({
    completely_generated_prob: doc.completely_generated_prob ?? null,
    average_generated_prob: doc.average_generated_prob ?? null,
    predicted_class: doc.predicted_class ?? null,
    overall_burstiness: doc.overall_burstiness ?? null,
    sentences: (doc.sentences || []).map(s => ({ text: s.sentence, generated_prob: s.generated_prob }))
  });
}

// 常量时间比较，避免通过响应时间猜密码
async function safeEqual(a, b) {
  const enc = new TextEncoder();
  const [ha, hb] = await Promise.all([
    crypto.subtle.digest('SHA-256', enc.encode(a)),
    crypto.subtle.digest('SHA-256', enc.encode(b))
  ]);
  const x = new Uint8Array(ha), y = new Uint8Array(hb);
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
  return diff === 0;
}

function asset(body, type) {
  return new Response(body, {
    headers: {
      'Content-Type': type,
      'Cache-Control': 'public, max-age=300',
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'no-referrer'
    }
  });
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' }
  });
}
