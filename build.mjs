// 把 pages-static/ 里的三个网页文件打包进 worker.js（单文件，便于在 Dashboard 里直接粘贴）
// 用法：node build.mjs
import { readFileSync, writeFileSync } from 'node:fs';

const read = (p) => readFileSync(new URL(p, import.meta.url), 'utf8');
const tpl = read('./worker-template.js');

// JSON.stringify 生成合法的 JS 字符串字面量（ES2019 起 U+2028/2029 在字符串中也合法）
const lit = (s) => JSON.stringify(s);

const out = tpl
  .replace('__INDEX_HTML__', () => lit(read('./pages-static/index.html')))
  .replace('__STYLE_CSS__', () => lit(read('./pages-static/style.css')))
  .replace('__APP_JS__', () => lit(read('./pages-static/app.js')));

writeFileSync(new URL('./worker.js', import.meta.url), out);
console.log('worker.js written,', out.length, 'chars');
