#!/usr/bin/env node
'use strict';

// Render Markdown and LaTeX into a self-contained, offline HTML report.
// Build dependencies: mathjax-full@3.2.2 and marked@17.0.5 (Node.js >= 20).
// The generated report needs no JavaScript, remote fonts, or CDN resources.

const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');
const { pathToFileURL } = require('node:url');

function parseArguments(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i++) {
    const name = argv[i];
    if (name === '--help') options.help = true;
    else if (['--source', '--output', '--latex-output', '--dependencies'].includes(name)) {
      if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw new Error(`Missing value for ${name}`);
      options[name.slice(2)] = argv[++i];
    } else throw new Error(`Unknown argument: ${name}`);
  }
  return options;
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[character]);
}

function escapeTeX(value) {
  return String(value).replace(/[\\{}%$&#_^~]/g, character => ({
    '\\': '\\textbackslash{}', '{': '\\{', '}': '\\}', '%': '\\%',
    '$': '\\$', '&': '\\&', '#': '\\#', '_': '\\_',
    '^': '\\textasciicircum{}', '~': '\\textasciitilde{}'
  })[character]);
}

function buildLaTeX(tokens, reportDate) {
  function inline(items) {
    return (items || []).map(token => {
      if (token.type === 'mathInline') return `$${token.text}$`;
      if (token.type === 'strong') return `\\textbf{${inline(token.tokens)}}`;
      if (token.type === 'em') return `\\emph{${inline(token.tokens)}}`;
      if (token.type === 'link') return `\\href{${escapeTeX(token.href)}}{${inline(token.tokens)}}`;
      if (token.type === 'image') return `\\textit{${escapeTeX(token.text)}（见HTML版附图）}`;
      if (token.type === 'codespan') return `\\texttt{${escapeTeX(token.text).replace(/([/.:])/g, '$1\\allowbreak{}').replaceAll('\\_', '\\_\\allowbreak{}')}}`;
      if (token.type === 'br') return '\\\\';
      return token.tokens ? inline(token.tokens) : escapeTeX(token.text || '');
    }).join('');
  }
  function blocks(items) {
    return items.map(token => {
      if (token.type === 'space') return '';
      if (token.type === 'heading') {
        if (token.depth === 1) return '';
        const command = token.depth === 2 ? 'section' : 'subsection';
        const text = inline(token.tokens);
        return `\\${command}*{${text}}\n\\addcontentsline{toc}{${command}}{${text}}\n`;
      }
      if (token.type === 'mathBlock') return `\\[\n${token.text}\n\\]\n`;
      if (token.type === 'paragraph' || token.type === 'text') return `${inline(token.tokens)}\n\n`;
      if (token.type === 'list') {
        const environment = token.ordered ? 'enumerate' : 'itemize';
        return `\\begin{${environment}}\n${token.items.map(item => `\\item ${blocks(item.tokens)}`).join('\n')}\\end{${environment}}\n`;
      }
      if (token.type === 'table') {
        const count = token.header.length;
        const column = `p{\\dimexpr\\linewidth/${count}-2\\tabcolsep-2\\arrayrulewidth\\relax}`;
        const row = cells => cells.map(cell => inline(cell.tokens)).join(' & ') + ' \\\\ \\hline\n';
        return `{\\small\n\\begin{longtable}{|${Array(count).fill(column).join('|')}|}\n\\hline\n${row(token.header)}\\endfirsthead\n\\hline\n${row(token.header)}\\endhead\n${token.rows.map(row).join('')}\\end{longtable}\n}\n`;
      }
      if (token.type === 'code') return `\\begin{verbatim}\n${token.text}\n\\end{verbatim}\n`;
      if (token.type === 'blockquote') return `\\begin{quote}\n${blocks(token.tokens)}\\end{quote}\n`;
      if (token.type === 'hr') return '\\par\\noindent\\rule{\\linewidth}{0.4pt}\n';
      throw new Error(`Unsupported LaTeX block token: ${token.type}`);
    }).join('\n');
  }
  return String.raw`\documentclass[UTF8,fontset=fandol]{ctexart}
\usepackage[a4paper,margin=21mm]{geometry}
\usepackage{amsmath,amssymb,array,longtable,booktabs,xcolor,hyperref}
\definecolor{reportblue}{HTML}{174BA8}
\hypersetup{colorlinks=true,linkcolor=reportblue,urlcolor=reportblue,pdftitle={PartAware-SG 综合研发与实验报告}}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.25}
\title{PartAware-SG 综合研发与实验报告}
\author{}
\date{${escapeTeX(reportDate)}}
\begin{document}
\maketitle
\tableofcontents
\clearpage
` + blocks(tokens) + '\n\\end{document}\n';
}

async function buildReport(options) {
  const root = path.resolve(__dirname, '..');
  const source = path.resolve(options.source || path.join(root, 'docs', 'RESEARCH_REPORT.md'));
  const output = path.resolve(options.output || path.join(root, 'RESEARCH_LOG.html'));
  if (source === output) throw new Error('Source and output must be different files');
  const loader = options.dependencies
    ? createRequire(path.join(path.resolve(options.dependencies), 'package.json'))
    : createRequire(__filename);
  const { marked } = await import(pathToFileURL(loader.resolve('marked')).href);
  const { mathjax } = loader('mathjax-full/js/mathjax.js');
  const { TeX } = loader('mathjax-full/js/input/tex.js');
  const { SVG } = loader('mathjax-full/js/output/svg.js');
  const { liteAdaptor } = loader('mathjax-full/js/adaptors/liteAdaptor.js');
  const { RegisterHTMLHandler } = loader('mathjax-full/js/handlers/html.js');
  const { AllPackages } = loader('mathjax-full/js/input/tex/AllPackages.js');
  const adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);
  const tex = new TeX({ packages: AllPackages.filter(name =>
    !['autoload', 'require', 'html', 'noerrors', 'noundefined'].includes(name)) });
  const svg = new SVG({ fontCache: 'none' });
  const document = mathjax.document('', { InputJax: tex, OutputJax: svg });
  const counts = { display: 0, inline: 0 };
  const headings = [];
  const markdown = fs.readFileSync(source, 'utf8').replace(/\r\n/g, '\n');
  const reportDate = /^更新[：:]\s*(\d{4}-\d{2}-\d{2})/m.exec(markdown)?.[1] || '';
  const reportVersion = /^当前主流程为(v\d+)/m.exec(markdown)?.[1] || '';
  const edition = ['研究报告', reportVersion, reportDate, 'LaTeX 公式渲染版'].filter(Boolean).join(' · ');

  function renderMath(expression, display) {
    let node;
    try {
      node = document.convert(expression, { display, em: 16, ex: 8, containerWidth: 800 });
    } catch (error) {
      throw new Error(`Could not render LaTeX expression: ${expression}`, { cause: error });
    }
    const rendered = adaptor.outerHTML(node);
    if (/data-mml-node="merror"|data-mjx-error=/.test(rendered)) {
      throw new Error(`Invalid LaTeX expression: ${expression}`);
    }
    counts[display ? 'display' : 'inline']++;
    const tag = display ? 'div' : 'span';
    return `<${tag} class="math-${display ? 'display' : 'inline'}" data-tex="${escapeHTML(expression)}">${rendered}</${tag}>${display ? '\n' : ''}`;
  }

  marked.use({
    gfm: true,
    extensions: [
      {
        name: 'mathBlock', level: 'block',
        start(source) { return source.indexOf('$$'); },
        tokenizer(source) {
          const match = /^\$\$[ \t]*\n([\s\S]+?)\n\$\$[ \t]*(?:\n|$)/.exec(source);
          if (match) return { type: 'mathBlock', raw: match[0], text: match[1].trim() };
        },
        renderer(token) { return renderMath(token.text, true); }
      },
      {
        name: 'mathInline', level: 'inline',
        start(source) { return source.indexOf('$'); },
        tokenizer(source) {
          const match = /^\$(?!\$)((?:\\.|[^$\\\n])+)\$/.exec(source);
          if (match) return { type: 'mathInline', raw: match[0], text: match[1].trim() };
        },
        renderer(token) { return renderMath(token.text, false); }
      }
    ],
    renderer: {
      html(token) { return escapeHTML(token.text); },
      image(token) {
        if (/^[a-z]+:|^[/\\]/i.test(token.href)) throw new Error('Report images must be local relative paths');
        const imagePath = path.resolve(path.dirname(source), token.href);
        const relative = path.relative(path.dirname(source), imagePath);
        if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Image outside report directory');
        const extension = path.extname(imagePath).toLowerCase();
        const mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}[extension];
        if (!mime) throw new Error('Unsupported report image format');
        return `<img src="data:${mime};base64,${fs.readFileSync(imagePath).toString('base64')}" alt="${escapeHTML(token.text)}" loading="lazy">`;
      },
      heading(token) {
        const label = this.parser.parseInline(token.tokens);
        const number = /^(\d+(?:\.\d+)*)(?:\.|\s)/.exec(token.text);
        const id = number ? `section-${number[1].replaceAll('.', '-')}` : `heading-${headings.length}`;
        if (token.depth > 1) headings.push({ id, depth: token.depth, label: token.text });
        return `<h${token.depth} id="${id}">${label}</h${token.depth}>\n`;
      }
    }
  });
  let body = marked.parse(markdown);
  // Turn the first chapter's nested Markdown list into an accessible tree.
  // The same source still renders as ordinary nested lists in LaTeX.
  const flowStart = body.indexOf('<h2 id="section-1"');
  const flowEnd = body.indexOf('<h2 id="section-2"', flowStart);
  if (flowStart >= 0 && flowEnd > flowStart) {
    const chapter = body.slice(flowStart, flowEnd);
    const parsed = adaptor.parse(chapter, 'text/html');
    const container = adaptor.body(parsed);
    const lists = adaptor.tags(container, 'ul');
    if (lists.length && adaptor.textContent(lists[0]).includes('FOVEA')) {
      adaptor.setAttribute(lists[0], 'class', 'pipeline-tree');
      for (const list of lists.slice(1).reverse()) {
        const parent = adaptor.parent(list);
        if (adaptor.kind(parent) !== 'li') continue;
        const children = adaptor.childNodes(parent);
        const details = adaptor.node('details', { open: '' });
        const summary = adaptor.node('summary');
        for (const child of children) {
          if (child === list) break;
          adaptor.remove(child);
          if (adaptor.kind(child) === 'p') {
            for (const inline of [...adaptor.childNodes(child)]) {
              adaptor.remove(inline); adaptor.append(summary, inline);
            }
          } else adaptor.append(summary, child);
        }
        adaptor.remove(list);
        adaptor.append(details, summary); adaptor.append(details, list);
        adaptor.append(parent, details);
      }
      body = body.slice(0, flowStart) + adaptor.innerHTML(container) + body.slice(flowEnd);
    }
  }
  if (!counts.display || !counts.inline) throw new Error('No complete report math was rendered');
  const mathSource = markdown
    .replace(/^```[^\n]*\n[\s\S]*?^```[ \t]*$/gm, '')
    .replace(/`[^`\n]*`/g, '');
  const expectedDisplay = (mathSource.match(/^\$\$[ \t]*$/gm) || []).length / 2;
  const expectedInline = (mathSource.replace(/^\$\$[ \t]*\n[\s\S]+?\n\$\$[ \t]*(?:\n|$)/gm, '')
    .match(/\$(?!\$)(?:\\.|[^$\\\n])+\$/g) || []).length;
  if (counts.display !== expectedDisplay || counts.inline !== expectedInline) {
    throw new Error(`Math count mismatch: rendered ${JSON.stringify(counts)}, expected ${expectedDisplay}/${expectedInline}`);
  }
  const navigation = headings.map(heading =>
    `<a class="toc-depth-${heading.depth}" href="#${heading.id}">${escapeHTML(heading.label)}</a>`).join('\n');
  const sourceHash = createHash('sha256').update(markdown).digest('hex');
  const mathStyles = adaptor.textContent(svg.styleSheet(document));
  const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="source-sha256" content="${sourceHash}">
<meta name="report-version" content="${escapeHTML(reportVersion)}">
<meta name="report-updated" content="${escapeHTML(reportDate)}">
<meta name="math-display-count" content="${counts.display}">
<meta name="math-inline-count" content="${counts.inline}">
<title>PartAware-SG 综合研发与实验报告 · LaTeX 渲染版</title>
<style>
${mathStyles}
:root { color-scheme: light dark; --bg:#f4f6fa; --paper:#fff; --ink:#202b3d; --muted:#64748b; --line:#dce4ef; --accent:#174ba8; --formula:#f8faff; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; color:var(--ink); background:var(--bg); font:16px/1.85 "Segoe UI","Microsoft YaHei","Noto Sans CJK SC",sans-serif; }
.pipeline-tree { list-style:none; padding:0; margin:24px 0; }
.pipeline-tree > li { margin:18px 0; padding:14px 18px; border:1px solid var(--line); border-radius:12px; background:var(--formula); }
.pipeline-tree ul { list-style:none; margin:10px 0 4px 8px; padding-left:18px; border-left:2px solid var(--line); }
.pipeline-tree ul > li { position:relative; margin:9px 0; padding-left:9px; }
.pipeline-tree ul > li::before { content:""; position:absolute; top:16px; left:-18px; width:19px; border-top:2px solid var(--line); }
.pipeline-tree summary { cursor:pointer; font-weight:600; line-height:1.6; overflow-wrap:anywhere; }
.pipeline-tree summary:hover { color:var(--accent); }
.pipeline-tree li p { margin:5px 0; }
@media(max-width:600px) { .pipeline-tree > li { padding:12px 9px; } .pipeline-tree ul { margin-left:2px; padding-left:13px; } .pipeline-tree ul > li::before { left:-13px; width:14px; } }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
.layout { display:grid; grid-template-columns:240px minmax(0,1fr); gap:30px; max-width:1400px; padding:36px 30px 64px; margin:auto; }
nav { position:sticky; top:24px; align-self:start; max-height:calc(100vh - 48px); overflow:auto; font-size:13px; line-height:1.6; padding:16px 8px; }
nav strong { display:block; color:var(--muted); font-size:12px; letter-spacing:.12em; margin-bottom:14px; }
nav a { display:block; padding:5px 0; color:var(--muted); }
nav .toc-depth-3 { padding-left:14px; font-size:12px; }
main { min-width:0; overflow-wrap:anywhere; background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:42px 48px; box-shadow:0 8px 32px #152d4c06; }
.edition { color:var(--accent); font-size:13px; font-weight:600; letter-spacing:.08em; margin-bottom:16px; }
h1 { font-size:30px; line-height:1.4; margin:0 0 24px; }
h2 { font-size:23px; line-height:1.5; margin:44px 0 18px; padding-top:20px; border-top:1px solid var(--line); }
h3 { font-size:19px; line-height:1.55; margin:32px 0 14px; }
h1,h2,h3 { scroll-margin-top:24px; }
p { margin:14px 0; }
li { margin:8px 0; }
strong { font-weight:650; }
code { font-family:Consolas,"SFMono-Regular",monospace; font-size:.88em; background:var(--formula); border-radius:4px; padding:2px 5px; overflow-wrap:anywhere; }
pre { overflow:auto; padding:16px; background:var(--formula); border:1px solid var(--line); border-radius:8px; line-height:1.65; }
pre code { padding:0; background:none; overflow-wrap:normal; }
table { border-collapse:collapse; width:100%; font-size:14px; line-height:1.7; margin:20px 0; }
img { max-width:100%; height:auto; border:1px solid #d9e1ec; border-radius:6px; }
th,td { padding:10px 12px; border:1px solid var(--line); text-align:left; vertical-align:top; }
th { background:var(--formula); font-weight:650; }
.math-display { overflow-x:auto; overflow-y:hidden; padding:18px 20px; margin:22px 0; background:var(--formula); border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:7px; font-size:1.04em; }
.math-display mjx-container[jax="SVG"][display="true"] { margin:0; text-align:left; width:max-content; min-width:100%; }
.math-inline { white-space:nowrap; }
footer { border-top:1px solid var(--line); margin-top:40px; padding-top:18px; color:var(--muted); font-size:12px; }
@media(prefers-color-scheme:dark) { :root { --bg:#111722; --paper:#19212d; --ink:#e2e8f0; --muted:#a1aec0; --line:#344155; --accent:#8db8ff; --formula:#1c293b; } }
@media(max-width:1080px) { .layout { grid-template-columns:minmax(0,1fr); padding:20px; gap:16px; } nav { position:static; max-height:190px; padding:10px 16px; border:1px solid var(--line); border-radius:8px; } main { padding:30px; } }
@media(max-width:600px) { .layout { padding:10px; } main { padding:20px 16px; } h1 { font-size:24px; } h2 { font-size:21px; } table { display:block; overflow-x:auto; } .math-display { padding:14px 12px; font-size:.93em; } }
@media print { :root { --paper:#fff; --ink:#111; --muted:#555; --line:#ddd; --accent:#222; --formula:#fff; } body { background:#fff; font-size:11pt; } .layout { display:block; padding:0; } nav { display:none; } main { border:0; box-shadow:none; padding:0; } .math-display { break-inside:avoid; overflow:visible; } h2,h3 { break-after:avoid; } a { color:inherit; } }
</style>
</head>
<body>
<div class="layout">
<nav aria-label="报告目录"><strong>报告目录</strong>${navigation}</nav>
<main>
<div class="edition">${escapeHTML(edition)}</div>
${body}
<footer>公式由 LaTeX 经 MathJax 排版，已内嵌为矢量图形。阅读无需联网。源码：docs/RESEARCH_REPORT.md。</footer>
</main>
</div>
</body>
</html>
`;
  fs.writeFileSync(output, html, 'utf8');
  if (options['latex-output']) {
    const latexOutput = path.resolve(options['latex-output']);
    if (latexOutput === source || latexOutput === output) throw new Error('LaTeX output must have a separate path');
    fs.writeFileSync(latexOutput, buildLaTeX(marked.lexer(markdown), reportDate), 'utf8');
  }
  process.stdout.write(`${JSON.stringify({ output, formulas: counts, source_sha256: sourceHash, bytes: Buffer.byteLength(html) })}\n`);
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    process.stdout.write('Build: node render_research_report.cjs [--source FILE] [--output FILE] [--latex-output FILE] [--dependencies DIR]\nDependencies: npm install --prefix DIR --ignore-scripts mathjax-full@3.2.2 marked@17.0.5\n');
    return;
  }
  await buildReport(options);
}

main().catch(error => { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; });
