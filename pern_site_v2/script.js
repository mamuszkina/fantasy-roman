async function loadSiteData() {
  const response = await fetch('site-data.json');
  return await response.json();
}

function renderTable(table) {
  const headers = table.colonnes.map(h => `<th>${h}</th>`).join('');
  const rows = table.lignes.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('');
  return `
    <article class="table-card">
      <h3>${table.titre}</h3>
      <div class="table-wrap">
        <table>
          <thead><tr>${headers}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <p class="footnote">${table.note}</p>
    </article>
  `;
}

function bookOptions(select, books) {
  select.innerHTML = books.map(book => `<option value="${book.slug}">${book.nom}</option>`).join('');
}

function findBook(data, slug) {
  return data.books.find(book => book.slug === slug);
}

function renderBook(container, book) {
  if (!book) {
    container.innerHTML = '<div class="panel">Aucun livre disponible.</div>';
    return;
  }
  container.innerHTML = `
    <section class="book-meta">
      <p class="eyebrow">Corpus</p>
      <h2>${book.nom}</h2>
      <p>${book.description}</p>
    </section>
    ${book.tableaux.map(renderTable).join('')}
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

function renderComparisonColumn(book) {
  return `
    <div class="compare-column stack">
      <div>
        <p class="eyebrow">Livre</p>
        <h2>${book.nom}</h2>
        <p class="muted">${book.description}</p>
      </div>
      ${book.tableaux.map(renderTable).join('')}
    </div>
  `;
}

async function initComparaisonPage() {
  const selectA = document.getElementById('bookA');
  if (!selectA) return;
  const selectB = document.getElementById('bookB');
  const btn = document.getElementById('compareBtn');
  const container = document.getElementById('comparisonContent');
  const data = await loadSiteData();
  bookOptions(selectA, data.books);
  bookOptions(selectB, data.books);
  if (data.books.length > 1) selectB.value = data.books[1].slug;
  const render = () => {
    const bookA = findBook(data, selectA.value);
    const bookB = findBook(data, selectB.value);
    container.innerHTML = renderComparisonColumn(bookA) + renderComparisonColumn(bookB);
  };
  render();
  btn.addEventListener('click', render);
}

async function initBricPage() {
  const link = document.getElementById('methodologyLink');
  if (!link) return;
  const data = await loadSiteData();
  link.href = data.methodologyGithubUrl;
  link.textContent = data.methodologyGithubUrl;
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
        <div class="result-bar-label"><strong>${name}</strong><span>${count} vote(s) — ${pct} %</span></div>
        <div class="result-bar"><div class="result-bar-fill" style="width:${pct}%"></div></div>
      </div>
    `;
  }).join('');
}

async function fetchSupabaseVotes(client, table, options) {
  const { data, error } = await client.from(table).select('book_title');
  if (error) throw error;
  const counts = Object.fromEntries(options.map(opt => [opt, 0]));
  for (const row of data) {
    counts[row.book_title] = (counts[row.book_title] || 0) + 1;
  }
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
    <label><input type="radio" name="voteChoice" value="${opt}" ${i === 0 ? 'checked' : ''}> ${opt}</label>
  `).join('');

  const client = getSupabaseClient();
  const cfg = window.SUPABASE_CONFIG || {};

  async function refresh() {
    if (client) {
      mode.textContent = 'Mode de vote partagé : Supabase configuré.';
      try {
        renderVoteResults(results, await fetchSupabaseVotes(client, cfg.table, options));
      } catch (err) {
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
initComparaisonPage();
initVotePage();
initBricPage();
