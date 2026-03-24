async function loadSiteData() {
  const response = await fetch('site-data.json');
  return await response.json();
}

function findBook(data, slug) {
  return data.books.find(book => book.slug === slug);
}

function bookOptions(select, books) {
  select.innerHTML = books.map(book => `<option value="${book.slug}">${book.nom}</option>`).join('');
}

function themeOptions(select, books) {
  const map = new Map();
  books.forEach(book => (book.themes || []).forEach(theme => {
    if (!map.has(theme.id)) map.set(theme.id, theme.titre);
  }));
  select.innerHTML = [...map.entries()].map(([id, label]) => `<option value="${id}">${label}</option>`).join('');
}

function orderOptions(select, data) {
  if (!select) return;
  select.innerHTML = (data.themeOrderOptions || []).map(opt => `<option value="${opt.id}">${opt.label}</option>`).join('');
}

function getWomenPercentage(book) {
  const theme = (book.themes || []).find(t => t.id === 'genre_persos');
  const femaleBar = theme?.bars?.find(bar => /femme/i.test(bar.label));
  return femaleBar?.percentage ?? -1;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderSignificanceBadge(theme) {
  const sig = theme.significance || {};
  if (sig.significant === null || sig.significant === undefined) {
    return `<div class="sig-badge neutral">${escapeHtml(sig.label || 'Pas de test')}</div>`;
  }
  return `<div class="sig-badge ${sig.significant ? 'significant' : 'nonsignificant'}">${sig.significant ? 'Différence significative' : 'Différence non significative'} · ${escapeHtml(sig.stars || 'n.s.')}</div>`;
}

function barDetail(bar, themeId) {
  if (themeId === 'morts') {
    return `${bar.count} mort${bar.count > 1 ? 's' : ''} / ${bar.total} · ${bar.percentage.toFixed(1).replace('.', ',')} %`;
  }
  if (themeId === 'ttr') {
    return `${bar.count} types / ${bar.total} tokens · ${bar.percentage.toFixed(1).replace('.', ',')} %`;
  }
  return `${bar.count} · ${bar.percentage.toFixed(1).replace('.', ',')} %`;
}

function renderHorizontalChart(theme, { compact = false } = {}) {
  const bars = (theme.bars || []).map(bar => `
    <div class="chart-row">
      <div class="chart-label">${escapeHtml(bar.label)}</div>
      <div class="chart-bar-shell">
        <div class="chart-bar-fill" style="width:${Math.max(0, Math.min(100, bar.value || 0))}%"></div>
        <div class="chart-bar-text">${escapeHtml(barDetail(bar, theme.id))}</div>
      </div>
    </div>
  `).join('');

  return `
    <article class="chart-card ${compact ? 'compact-chart' : ''}">
      <div class="chart-head">
        <div>
          <h3>${escapeHtml(theme.titre)}</h3>
          <p class="muted">${escapeHtml(theme.sousTitre || '')}</p>
        </div>
        ${renderSignificanceBadge(theme)}
      </div>
      <div class="chart-block">${bars}</div>
      <p class="footnote">${escapeHtml(theme.note)}</p>
    </article>
  `;
}

function renderBook(container, book) {
  if (!book) {
    container.innerHTML = '<div class="panel">Aucun livre disponible.</div>';
    return;
  }
  container.innerHTML = `
    <section class="book-meta">
      <p class="eyebrow">Corpus</p>
      <h2>${escapeHtml(book.nom)}</h2>
      <p>${escapeHtml(book.description)}</p>
    </section>
    ${(book.themes || []).map(theme => renderHorizontalChart(theme)).join('')}
  `;
}

async function initLivrePage() {
  const select = document.getElementById('bookSelect');
  if (!select) return;
  const btn = document.getElementById('loadBookBtn');
  const container = document.getElementById('bookContent');
  const data = await loadSiteData();
  bookOptions(select, data.books);
  renderBook(container, data.books[0]);
  btn.addEventListener('click', () => renderBook(container, findBook(data, select.value)));
}

function renderThemeCollection(container, books, themeId, order) {
  const themeBlocks = books
    .map(book => ({ book, theme: (book.themes || []).find(theme => theme.id === themeId) }))
    .filter(item => item.theme);

  if (order === 'women') {
    themeBlocks.sort((a, b) => getWomenPercentage(b.book) - getWomenPercentage(a.book));
  } else {
    themeBlocks.sort((a, b) => a.book.nom.localeCompare(b.book.nom, 'fr'));
  }

  if (!themeBlocks.length) {
    container.innerHTML = '<div class="panel">Aucune donnée disponible pour ce thème.</div>';
    return;
  }

  container.innerHTML = themeBlocks.map(({ book, theme }) => `
    <section class="theme-book-card">
      <div class="theme-book-head">
        <div>
          <p class="eyebrow">Roman</p>
          <h2>${escapeHtml(book.nom)}</h2>
          <p class="muted">${escapeHtml(book.description)}</p>
        </div>
        <div class="metric-chip">Femmes : ${getWomenPercentage(book) >= 0 ? `${getWomenPercentage(book).toFixed(1).replace('.', ',')} %` : 'n.d.'}</div>
      </div>
      ${renderHorizontalChart(theme, { compact: true })}
    </section>
  `).join('');
}

async function initThemesPage() {
  const themeSelect = document.getElementById('themeSelect');
  if (!themeSelect) return;
  const orderSelect = document.getElementById('orderSelect');
  const btn = document.getElementById('loadThemeBtn');
  const container = document.getElementById('themeContent');
  const data = await loadSiteData();
  themeOptions(themeSelect, data.books);
  orderOptions(orderSelect, data);

  const render = () => renderThemeCollection(container, data.books, themeSelect.value, orderSelect.value);
  render();
  btn.addEventListener('click', render);
}

function pcaRanges(points, arrows) {
  const xs = [...points.map(p => p.x), ...arrows.map(a => a.x), 0];
  const ys = [...points.map(p => p.y), ...arrows.map(a => a.y), 0];
  const maxAbsX = Math.max(...xs.map(v => Math.abs(v)), 1);
  const maxAbsY = Math.max(...ys.map(v => Math.abs(v)), 1);
  return { maxAbsX: maxAbsX * 1.25, maxAbsY: maxAbsY * 1.25 };
}

function toSvgX(x, maxAbsX, width, pad) {
  return pad + ((x + maxAbsX) / (2 * maxAbsX)) * (width - 2 * pad);
}

function toSvgY(y, maxAbsY, height, pad) {
  return height - pad - ((y + maxAbsY) / (2 * maxAbsY)) * (height - 2 * pad);
}

function renderPcaSvg(pca) {
  const width = 900;
  const height = 620;
  const pad = 70;
  const points = pca.points || [];
  const arrows = pca.arrows || [];
  const { maxAbsX, maxAbsY } = pcaRanges(points, arrows);
  const x0 = toSvgX(0, maxAbsX, width, pad);
  const y0 = toSvgY(0, maxAbsY, height, pad);

  const pointData = points.map(point => ({
    ...point,
    sx: toSvgX(point.x, maxAbsX, width, pad),
    sy: toSvgY(point.y, maxAbsY, height, pad)
  }));

  const arrowData = arrows.map(arrow => ({
    ...arrow,
    sx: toSvgX(arrow.x, maxAbsX, width, pad),
    sy: toSvgY(arrow.y, maxAbsY, height, pad)
  }));

  const labelNodes = pointData.map(d => ({
    label: d.label,
    anchorX: d.sx,
    anchorY: d.sy,
    x: d.sx + 14,
    y: d.sy - 14
  }));

  // Simulation légère pour limiter le recouvrement des étiquettes
  const simulation = d3.forceSimulation(labelNodes)
    .force("x", d3.forceX(d => d.anchorX + 14).strength(0.25))
    .force("y", d3.forceY(d => d.anchorY - 14).strength(0.25))
    .force("collide", d3.forceCollide(18))
    .stop();

  for (let i = 0; i < 250; i++) simulation.tick();

  const arrowLines = arrowData.map(arrow => {
    const dx = arrow.sx - x0;
    const dy = arrow.sy - y0;
    const norm = Math.sqrt(dx * dx + dy * dy) || 1;
    const lx = arrow.sx + (dx / norm) * 12;
    const ly = arrow.sy + (dy / norm) * 12;

    return `
      <g class="pca-arrow-group">
        <line
          x1="${x0}"
          y1="${y0}"
          x2="${arrow.sx}"
          y2="${arrow.sy}"
          class="pca-arrow-line"
          marker-end="url(#arrowhead)"
        ></line>
        <text
          x="${lx}"
          y="${ly}"
          class="pca-arrow-label"
          text-anchor="${dx >= 0 ? 'start' : 'end'}"
          dominant-baseline="${dy >= 0 ? 'hanging' : 'auto'}"
        >${escapeHtml(arrow.label)}</text>
      </g>
    `;
  }).join('');

  const pointNodes = pointData.map(point => `
    <circle cx="${point.sx}" cy="${point.sy}" r="8" class="pca-point"></circle>
  `).join('');

  const pointLinks = labelNodes.map((node, i) => `
    <line
      x1="${pointData[i].sx}"
      y1="${pointData[i].sy}"
      x2="${node.x}"
      y2="${node.y}"
      class="pca-label-link"
    ></line>
  `).join('');

  const pointLabels = labelNodes.map(node => `
    <text
      x="${node.x}"
      y="${node.y}"
      class="pca-point-label"
      text-anchor="start"
      dominant-baseline="middle"
    >${escapeHtml(node.label)}</text>
  `).join('');

  return `
    <svg viewBox="0 0 ${width} ${height}" class="pca-svg" aria-label="ACP des romans">
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="8" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" class="pca-marker"></polygon>
        </marker>
      </defs>

      <rect x="0" y="0" width="${width}" height="${height}" rx="24" class="pca-bg"></rect>

      <line x1="${pad}" y1="${y0}" x2="${width - pad}" y2="${y0}" class="pca-axis"></line>
      <line x1="${x0}" y1="${pad}" x2="${x0}" y2="${height - pad}" class="pca-axis"></line>

      <text x="${width - pad}" y="${y0 - 14}" class="pca-axis-label">
        PC1 (${(pca.explainedVariance?.[0] ?? 0).toFixed(1).replace('.', ',')} %)
      </text>
      <text x="${x0 + 14}" y="${pad}" class="pca-axis-label">
        PC2 (${(pca.explainedVariance?.[1] ?? 0).toFixed(1).replace('.', ',')} %)
      </text>

      ${arrowLines}
      ${pointNodes}
      ${pointLinks}
      ${pointLabels}
    </svg>
  `;
}


async function initPcaPage() {
  const container = document.getElementById('pcaContent');
  if (!container) return;
  const data = await loadSiteData();
  const pca = data.pca || {};
  if (!pca.available) {
    container.innerHTML = `
      <section class="panel stack">
        <div class="status-box">${escapeHtml(pca.reason || 'ACP indisponible.')}</div>
        <p class="muted">Ajoutez au moins un second roman avec les mêmes fichiers CSV pour positionner plusieurs œuvres dans l'espace factoriel.</p>
      </section>
    `;
    return;
  }

  container.innerHTML = `
    <section class="panel stack">
      <div class="pca-explainer">
        <p>Cette ACP résume plusieurs indicateurs de genre en deux dimensions. Les flèches représentent les variables : plus elles sont longues, plus elles structurent l'espace. Les points correspondent aux romans. Deux romans proches ont des profils de genre similaires sur l'ensemble des indicateurs disponibles.</p>
      </div>
      ${renderPcaSvg(pca)}
    </section>
  `;
}

function getSupabaseClient() {
  const cfg = window.SUPABASE_CONFIG || {};
  if (!cfg.url || !cfg.anonKey || !window.supabase) return null;
  return window.supabase.createClient(cfg.url, cfg.anonKey);
}

function localVoteKey() { return 'votes_books_local'; }

function readLocalVotes(options) {
  const raw = localStorage.getItem(localVoteKey());
  const base = Object.fromEntries(options.map(opt => [opt, 0]));
  if (!raw) return base;
  try {
    return { ...base, ...JSON.parse(raw) };
  } catch {
    return base;
  }
}

function writeLocalVote(option, options) {
  const votes = readLocalVotes(options);
  votes[option] = (votes[option] || 0) + 1;
  localStorage.setItem(localVoteKey(), JSON.stringify(votes));
  return votes;
}

function renderVoteResults(container, votes) {
  const total = Object.values(votes).reduce((a, b) => a + b, 0);
  container.innerHTML = Object.entries(votes).map(([name, count]) => {
    const pct = total ? ((count / total) * 100).toFixed(1) : '0.0';
    return `
      <div class="result-bar-wrap">
        <div class="result-bar-label"><strong>${escapeHtml(name)}</strong><span>${count} vote(s) — ${pct.replace('.', ',')} %</span></div>
        <div class="result-bar"><div class="result-bar-fill" style="width:${pct}%"></div></div>
      </div>
    `;
  }).join('');
}

async function fetchSupabaseVotes(client, table, options) {
  const { data, error } = await client.from(table).select('book_title');
  if (error) throw error;
  const counts = Object.fromEntries(options.map(opt => [opt, 0]));
  for (const row of data) counts[row.book_title] = (counts[row.book_title] || 0) + 1;
  return counts;
}

async function submitSupabaseVote(client, table, choice) {
  const { error } = await client.from(table).insert([{ book_title: choice }]);
  if (error) throw error;
}

async function initVotePage() {
  const form = document.getElementById('voteForm');
  if (!form) return;
  const results = document.getElementById('voteResults');
  const mode = document.getElementById('voteMode');
  const btn = document.getElementById('voteBtn');
  const message = document.getElementById('voteMessage');
  const data = await loadSiteData();
  const options = data.voteOptions || [];
  form.innerHTML = options.map((opt, i) => `
    <label><input type="radio" name="voteChoice" value="${escapeHtml(opt)}" ${i === 0 ? 'checked' : ''}> ${escapeHtml(opt)}</label>
  `).join('');

  const client = getSupabaseClient();
  const cfg = window.SUPABASE_CONFIG || {};

  async function refresh() {
    if (client) {
      mode.textContent = 'Mode de vote partagé : Supabase configuré.';
      try {
        renderVoteResults(results, await fetchSupabaseVotes(client, cfg.table, options));
      } catch {
        mode.textContent = 'Supabase détecté mais inaccessible. Repli local activé.';
        renderVoteResults(results, readLocalVotes(options));
      }
    } else {
      mode.textContent = 'Mode de test local : les votes sont stockés dans ce navigateur.';
      renderVoteResults(results, readLocalVotes(options));
    }
  }

  btn.addEventListener('click', async () => {
    const selected = form.querySelector('input[name="voteChoice"]:checked');
    if (!selected) return;
    const choice = selected.value;
    try {
      if (client) {
        await submitSupabaseVote(client, cfg.table, choice);
        message.textContent = 'Vote enregistré sur la base partagée.';
      } else {
        writeLocalVote(choice, options);
        message.textContent = 'Vote enregistré localement pour vos tests.';
      }
      await refresh();
    } catch (err) {
      message.textContent = `Erreur lors du vote : ${err.message || err}`;
    }
  });

  await refresh();
}

initLivrePage();
initThemesPage();
initPcaPage();
initVotePage();
