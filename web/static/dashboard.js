/* Identity Management Dashboard — grid, detail, delete, profile save. */

(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const state = {
    identities: [],            // cached list from /api/identities
    selected:   null,          // currently-viewed identity name
    filter:     '',
    bulkSet:    new Set(),     // names checked for bulk delete
    pending:    null,          // { targets, confirmWord } while delete modal is open
  };

  const el = {
    metaTotal:         $('meta-total'),
    pipelinePill:      $('pipeline-pill'),
    search:            $('search'),
    grid:              $('grid'),
    gridEmpty:         $('grid-empty'),

    detail:            $('detail'),
    detailPlaceholder: $('detail-placeholder'),
    profile:           $('profile'),
    profileThumb:      $('profile-thumb'),
    displayName:       $('profile-display-name'),
    profileInternal:   $('profile-internal'),
    chipMatches:       $('chip-matches'),
    chipFirstSeen:     $('chip-firstseen'),
    chipLastSeen:      $('chip-lastseen'),
    btnDelete:         $('btn-delete'),

    profileNotes:      $('profile-notes'),
    btnSaveFields:     $('btn-save-fields'),
    saveHint:          $('save-hint'),

    eventsBox:         $('events-box'),

    deleteModal:       $('delete-modal'),
    deleteTargetName:  $('delete-target-name'),
    deleteConfirmName: $('delete-confirm-name'),
    deleteConfirmIn:   $('delete-confirm-input'),
    deleteCancel:      $('delete-cancel'),
    deleteConfirm:     $('delete-confirm'),
    deleteError:       $('delete-error'),
    toast:             $('toast'),

    bulkBar:           $('bulk-bar'),
    bulkCountN:        $('bulk-count-n'),
    btnBulkClear:      $('btn-bulk-clear'),
    btnBulkDelete:     $('btn-bulk-delete'),
  };

  // ── Utilities ─────────────────────────────────────────────────────────

  const fmtRelative = (ts) => {
    if (!ts) return '—';
    const diff = Date.now() / 1000 - ts;
    if (diff < 5)    return 'just now';
    if (diff < 60)   return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    const d = new Date(ts * 1000);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const escapeHtml = (s) => String(s ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  async function api(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const data = await res.json(); msg += ` ${data.error || ''}`; } catch(_){}
      throw new Error(msg);
    }
    return res.json();
  }

  // ── Identity grid ─────────────────────────────────────────────────────

  async function loadIdentities() {
    try {
      const { identities } = await api('GET', '/api/identities');
      state.identities = identities || [];
      el.metaTotal.textContent = state.identities.length;
      renderGrid();
    } catch (err) {
      console.error('Failed to load identities:', err);
      el.gridEmpty.textContent = 'Failed to load identities.';
    }
  }

  function renderGrid() {
    // Drop bulk-selected names that no longer exist in the dataset.
    const known = new Set(state.identities.map((i) => i.name));
    for (const n of state.bulkSet) if (!known.has(n)) state.bulkSet.delete(n);

    const filter = state.filter.trim().toLowerCase();
    const visible = filter
      ? state.identities.filter((i) =>
          (i.name || '').toLowerCase().includes(filter) ||
          (i.display_name || '').toLowerCase().includes(filter) ||
          (i.notes || '').toLowerCase().includes(filter))
      : state.identities;

    el.grid.innerHTML = '';
    renderBulkBar();
    if (visible.length === 0) {
      const msg = document.createElement('div');
      msg.className = 'dash-empty';
      msg.textContent = filter
        ? `No matches for “${filter}”.`
        : 'No identities yet — enrol someone via the live view, then refresh.';
      el.grid.appendChild(msg);
      return;
    }

    for (const ident of visible) {
      const card = document.createElement('div');
      card.className = 'dash-card';
      if (state.selected === ident.name) card.classList.add('selected');
      card.dataset.name = ident.name;

      const check = document.createElement('input');
      check.type = 'checkbox';
      check.className = 'dash-card-check';
      check.checked = state.bulkSet.has(ident.name);
      check.setAttribute('aria-label', `Select ${ident.display_name || ident.name}`);
      check.addEventListener('click', (ev) => ev.stopPropagation());
      check.addEventListener('change', () => toggleBulk(ident.name, check.checked));
      card.appendChild(check);

      const thumb = document.createElement('div');
      thumb.className = 'dash-card-thumb';
      if (ident.has_thumbnail) {
        const img = document.createElement('img');
        img.src = `/api/thumbnail/${encodeURIComponent(ident.name)}`;
        img.alt = ident.display_name || ident.name;
        thumb.appendChild(img);
      } else {
        const ph = document.createElement('span');
        ph.className = 'thumb-placeholder';
        ph.textContent = '?';
        thumb.appendChild(ph);
      }

      const body = document.createElement('div');
      body.className = 'dash-card-body';
      body.innerHTML = `
        <div class="dash-card-name">${escapeHtml(ident.display_name || ident.name)}</div>
        ${ident.display_name && ident.display_name !== ident.name
          ? `<div class="dash-card-alias">${escapeHtml(ident.name)}</div>` : ''}
        <div class="dash-card-footer">
          <span>${ident.total_matches} matches</span>
          <span>${fmtRelative(ident.last_seen_at)}</span>
        </div>`;

      card.appendChild(thumb);
      card.appendChild(body);
      card.addEventListener('click', () => selectIdentity(ident.name));
      el.grid.appendChild(card);
    }
  }

  // ── Detail view ───────────────────────────────────────────────────────

  async function selectIdentity(name) {
    state.selected = name;
    renderGrid();   // refresh selection highlight

    el.detailPlaceholder.hidden = true;
    el.profile.hidden = false;
    setSaveHint('');

    try {
      const data = await api('GET', `/api/identities/${encodeURIComponent(name)}`);
      renderProfile(data);
    } catch (err) {
      console.error(err);
    }
  }

  function renderProfile(data) {
    if (data.has_thumbnail) {
      el.profileThumb.innerHTML = '';
      const img = document.createElement('img');
      img.src = `/api/thumbnail/${encodeURIComponent(data.name)}`;
      img.alt = data.display_name || data.name;
      el.profileThumb.appendChild(img);
    } else {
      el.profileThumb.innerHTML = '<span class="thumb-placeholder">?</span>';
    }

    el.displayName.value = data.display_name || '';
    el.displayName.placeholder = data.name;
    el.profileInternal.textContent = data.name;
    el.chipMatches.textContent = `${data.total_matches} ${data.total_matches === 1 ? 'match' : 'matches'}`;
    el.chipFirstSeen.textContent = `first seen ${fmtRelative(data.created_at)}`;
    el.chipLastSeen.textContent = `last seen ${fmtRelative(data.last_seen_at)}`;
    el.profileNotes.value = data.notes || '';

    renderEvents(data.events || []);
  }

  function renderEvents(events) {
    el.eventsBox.innerHTML = '';
    if (!events.length) {
      const empty = document.createElement('div');
      empty.className = 'dash-empty-soft';
      empty.textContent = 'No events recorded for this identity.';
      el.eventsBox.appendChild(empty);
      return;
    }
    for (const evt of events) {
      const row = document.createElement('div');
      row.className = 'event-row';
      row.innerHTML = `
        <span class="ev-type">${escapeHtml(evt.event_type || 'event')}</span>
        <span class="ev-content">${escapeHtml(evt.content || '')}</span>
      `;
      el.eventsBox.appendChild(row);
    }
  }

  // ── Actions ───────────────────────────────────────────────────────────

  async function saveFields() {
    if (!state.selected) return;
    setSaveHint('Saving…');
    try {
      await api('PATCH', `/api/identities/${encodeURIComponent(state.selected)}`, {
        display_name: el.displayName.value,
        notes:        el.profileNotes.value,
      });
      setSaveHint('Saved', true);
      await loadIdentities();
    } catch (err) {
      setSaveHint('Save failed');
    }
  }

  function setSaveHint(text, ok = false) {
    el.saveHint.textContent = text;
    el.saveHint.classList.toggle('saved', ok);
    if (ok) setTimeout(() => {
      if (el.saveHint.textContent === 'Saved') {
        el.saveHint.textContent = '';
        el.saveHint.classList.remove('saved');
      }
    }, 1800);
  }

  function disableBtn(btn, label) {
    btn.dataset.original = btn.dataset.original || btn.textContent;
    btn.disabled = true;
    btn.innerHTML = `<span class="btn-spinner"></span>${escapeHtml(label)}`;
  }

  function enableBtn(btn, fallback) {
    btn.disabled = false;
    btn.innerHTML = escapeHtml(btn.dataset.original || fallback);
  }

  // ── Toast ─────────────────────────────────────────────────────────────

  let toastTimer = null;
  function showToast(message, kind = '') {
    el.toast.hidden = false;
    el.toast.className = 'toast' + (kind ? ' ' + kind : '');
    el.toast.textContent = message;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.toast.hidden = true; }, 2800);
  }

  // ── Delete modal ──────────────────────────────────────────────────────
  //
  // Single- and bulk-delete share one flow: both set `state.pending` with the
  // list of targets and the word the user must type. `confirmDelete` then fans
  // out DELETE requests via `Promise.allSettled` so a partial failure keeps
  // the surviving selections intact instead of losing them.

  async function deleteIdentity(name) {
    const res = await fetch(`/api/identities/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok || !body.ok) throw new Error(body.error || `HTTP ${res.status}`);
    return body;
  }

  function beginDelete(targets, confirmWord, label) {
    state.pending = { targets, confirmWord };
    el.deleteTargetName.textContent = label;
    el.deleteConfirmName.textContent = confirmWord;
    el.deleteConfirmIn.value = '';
    el.deleteConfirm.disabled = true;
    el.deleteError.hidden = true;
    el.deleteError.textContent = '';
    el.deleteModal.hidden = false;
    setTimeout(() => el.deleteConfirmIn.focus(), 30);
  }

  function openDeleteModal() {
    if (!state.selected) return;
    beginDelete([state.selected], state.selected, state.selected);
  }

  function openBulkDeleteModal() {
    const names = [...state.bulkSet];
    if (!names.length) return;
    const preview = names.slice(0, 6).join(', ') +
      (names.length > 6 ? `, +${names.length - 6} more` : '');
    const noun = names.length === 1 ? 'identity' : 'identities';
    beginDelete(names, 'DELETE', `${names.length} ${noun} (${preview})`);
  }

  function closeDeleteModal() {
    el.deleteModal.hidden = true;
    el.deleteConfirmIn.value = '';
    state.pending = null;
  }

  function showDeleteError(msg) {
    el.deleteError.hidden = false;
    el.deleteError.textContent = msg;
  }

  async function confirmDelete() {
    const pending = state.pending;
    if (!pending) return;
    if (el.deleteConfirmIn.value !== pending.confirmWord) {
      showDeleteError(pending.targets.length > 1
        ? `Type ${pending.confirmWord} to confirm.`
        : 'Typed name does not match.');
      return;
    }

    disableBtn(el.deleteConfirm, 'Deleting…');
    try {
      const results = await Promise.allSettled(pending.targets.map(deleteIdentity));
      const ok = [], failed = [];
      results.forEach((r, i) => {
        const entry = { name: pending.targets[i], body: r.value, reason: r.reason };
        (r.status === 'fulfilled' ? ok : failed).push(entry);
      });

      for (const { name } of ok) state.bulkSet.delete(name);
      if (state.selected && ok.some((e) => e.name === state.selected)) {
        state.selected = null;
        el.profile.hidden = true;
        el.detailPlaceholder.hidden = false;
      }

      if (!failed.length) {
        showToast(deleteToast(ok), 'ok');
        closeDeleteModal();
      } else {
        const head = ok.length === 0
          ? 'Delete failed'
          : `Deleted ${ok.length}; ${failed.length} failed`;
        showDeleteError(`${head}: ${failed[0].reason.message}`);
      }
      await loadIdentities();
    } finally {
      enableBtn(el.deleteConfirm, 'Delete permanently');
    }
  }

  function deleteToast(ok) {
    if (ok.length === 1) {
      const { name, body = {} } = ok[0];
      return `Deleted ${name} (${body.embeddings_removed||0} embs · ` +
             `${body.events_removed||0} events · ${body.files_removed||0} files)`;
    }
    return `Deleted ${ok.length} identities`;
  }

  function toggleBulk(name, checked) {
    if (checked) state.bulkSet.add(name);
    else state.bulkSet.delete(name);
    renderBulkBar();
  }

  function clearBulkSelection() {
    state.bulkSet.clear();
    for (const cb of el.grid.querySelectorAll('.dash-card-check:checked')) {
      cb.checked = false;
    }
    renderBulkBar();
  }

  function renderBulkBar() {
    const n = state.bulkSet.size;
    el.bulkBar.hidden = n === 0;
    el.bulkCountN.textContent = String(n);
  }

  // ── Pipeline lifecycle (pause on entry, resume on leave) ──────────────

  async function pausePipeline() {
    try { await api('POST', '/api/pipeline/pause'); } catch (_) { /* swallow */ }
  }

  function resumePipelineBeacon() {
    // Best-effort resume on tab close / navigation away. sendBeacon is the
    // only reliable transport during pagehide on most browsers.
    try {
      const blob = new Blob([''], { type: 'application/json' });
      navigator.sendBeacon('/api/pipeline/resume', blob);
    } catch (_) { /* swallow */ }
  }

  // ── Wire up ───────────────────────────────────────────────────────────

  el.search.addEventListener('input', (e) => {
    state.filter = e.target.value;
    renderGrid();
  });

  el.btnSaveFields.addEventListener('click', saveFields);

  el.btnDelete.addEventListener('click', openDeleteModal);
  el.deleteCancel.addEventListener('click', closeDeleteModal);
  el.deleteConfirm.addEventListener('click', confirmDelete);
  el.deleteConfirmIn.addEventListener('input', () => {
    const expected = state.pending?.confirmWord ?? '';
    el.deleteConfirm.disabled = el.deleteConfirmIn.value !== expected;
    if (el.deleteError.textContent) {
      el.deleteError.hidden = true;
      el.deleteError.textContent = '';
    }
  });
  el.btnBulkClear.addEventListener('click', clearBulkSelection);
  el.btnBulkDelete.addEventListener('click', openBulkDeleteModal);
  el.deleteConfirmIn.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' && !el.deleteConfirm.disabled) {
      ev.preventDefault();
      confirmDelete();
    }
  });
  el.deleteModal.addEventListener('click', (ev) => {
    if (ev.target === el.deleteModal) closeDeleteModal();
  });
  document.addEventListener('keydown', (ev) => {
    if (!el.deleteModal.hidden && ev.key === 'Escape') closeDeleteModal();
  });

  // Lifecycle: pause the pipeline as soon as the dashboard loads (zero
  // CPU/GPU while the user is editing identities), and signal a resume on
  // pagehide so closing the tab — or navigating to an external page —
  // doesn't leave the camera idled.
  pausePipeline();
  window.addEventListener('pagehide', resumePipelineBeacon);

  // Auto-refresh the identity list every 10s so newly-enrolled people show up.
  loadIdentities();
  setInterval(loadIdentities, 10000);
})();
