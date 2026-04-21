/* ──────────────────────────────────────────────────────────────────────────
   Face Recognition — Dashboard frontend
   Polls /api/state at ~5 Hz and reconciles the DOM. The MJPEG <img> is
   independent of this loop so rendering a state tick never stalls the video.
   ────────────────────────────────────────────────────────────────────────── */

(() => {
  'use strict';

  const POLL_INTERVAL_MS  = 200;
  const RETRY_BACKOFF_MS  = 1500;
  const API_STATE         = '/api/state';
  const API_THUMBNAIL     = (id) => `/api/thumbnail/${encodeURIComponent(id)}`;

  // ── DOM refs (resolved once on load) ────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  const el = {
    metaKnown:        $('meta-known'),
    metaFps:          $('meta-fps'),
    metaQueue:        $('meta-queue'),
    systemPill:       $('system-pill'),
    systemDot:        $('system-dot'),
    systemStatus:     $('system-status'),

    facesTotal:       $('faces-total'),
    facesBadge:       $('faces-badge'),
    statusMsgBadge:   $('status-msg-badge'),

    identityCard:     $('identity-card'),
    thumb:            $('thumb'),
    identityDot:      $('identity-dot'),
    identityStatus:   $('identity-status'),
    identityName:     $('identity-name'),
    identityAlias:    $('identity-alias'),

    confFill:         $('conf-fill'),
    confValue:        $('conf-value'),
    qualFill:         $('qual-fill'),
    qualValue:        $('qual-value'),

    enrollmentBlock:  $('enrollment-block'),
    enrollFill:       $('enroll-fill'),
    enrollFrac:       $('enroll-frac'),

    metaEnrolled:     $('meta-enrolled'),
    metaLastseen:     $('meta-lastseen'),
    metaMatches:      $('meta-matches'),
    metaSession:      $('meta-session'),

    transcriptCard:   $('transcript-card'),
    transcriptScroll: $('transcript-scroll'),
    transcriptEmpty:  $('transcript-empty'),
    transcriptPartial:$('transcript-partial'),
    listeningPill:    $('listening-pill'),

    eventsCard:       $('events-card'),
    eventsList:       $('events-list'),

    video:            $('video'),
  };

  // ── State reconciliation helpers ────────────────────────────────────────

  /** Set text only if it changed — avoids needless DOM mutation churn. */
  function setText(node, value) {
    const v = value == null ? '' : String(value);
    if (node.textContent !== v) node.textContent = v;
  }

  /** Toggle class so it matches `token` and nothing else in `allowed`. */
  function setClassToken(node, allowed, token) {
    for (const t of allowed) node.classList.remove(t);
    if (token) node.classList.add(token);
  }

  function pctLabel(v) {
    if (v == null || isNaN(v) || v <= 0) return '—';
    return `${Math.round(v * 100)}%`;
  }

  function qualLabel(v) {
    if (v == null || isNaN(v) || v <= 0) return '—';
    return v.toFixed(2);
  }

  const STATUS_TOKENS = [
    'recognized', 'learning', 'new',
    'analyzing',  'uncertain', 'quality', 'offline',
  ];

  // ── Thumbnail cache (avoids flickering on every poll) ───────────────────
  let currentThumbId = null;

  function updateThumbnail(primary) {
    if (!primary || !primary.identity || !primary.has_thumbnail) {
      if (currentThumbId !== null) {
        el.thumb.innerHTML = '<span class="thumb-placeholder">?</span>';
        currentThumbId = null;
      }
      setClassToken(el.thumb, STATUS_TOKENS, null);
      return;
    }
    if (currentThumbId !== primary.identity) {
      // Swap to new identity thumbnail; cache-bust by identity name only.
      const img = new Image();
      img.alt = primary.display_name || primary.identity;
      img.src = API_THUMBNAIL(primary.identity);
      img.onload = () => {
        el.thumb.innerHTML = '';
        el.thumb.appendChild(img);
      };
      img.onerror = () => {
        el.thumb.innerHTML = '<span class="thumb-placeholder">?</span>';
      };
      currentThumbId = primary.identity;
    }
    setClassToken(el.thumb, STATUS_TOKENS, primary.status_token);
  }

  // ── Main reconciler ─────────────────────────────────────────────────────

  function applyState(snap) {
    const sys = snap.system || {};

    // Topbar meta
    setText(el.metaKnown, sys.known_count != null ? sys.known_count : '—');
    // Render FPS primary, detection FPS secondary — both matter.
    if (sys.fps != null) {
      const det = (sys.detection_fps != null && sys.detection_fps > 0)
        ? ` · det ${sys.detection_fps.toFixed(1)}` : '';
      setText(el.metaFps, sys.fps.toFixed(1) + det);
    } else {
      setText(el.metaFps, '—');
    }
    setText(el.metaQueue,
      (sys.queue_size != null && sys.queue_max != null)
        ? `${sys.queue_size}/${sys.queue_max}` : '—');

    // System status pill
    let sysToken = 'recognized', sysText = 'Ready';
    if (!sys.worker_ready) { sysToken = 'analyzing'; sysText = 'Starting…'; }
    else if (!sys.worker_alive) { sysToken = 'quality';   sysText = 'Recognition offline'; }
    else if (!sys.camera_ok)    { sysToken = 'uncertain'; sysText = 'Camera unavailable'; }
    else if (sys.status_message){ sysToken = 'analyzing'; sysText = sys.status_message; }
    setClassToken(el.systemDot, STATUS_TOKENS, sysToken);
    setText(el.systemStatus, sysText);

    // Faces badge
    const faces = snap.faces || { total: 0, recognized: 0, new: 0 };
    setText(el.facesTotal, faces.total);
    if (sys.status_message && !sys.worker_alive) {
      el.statusMsgBadge.hidden = false;
      setText(el.statusMsgBadge, sys.status_message);
    } else {
      el.statusMsgBadge.hidden = true;
    }

    // Primary face
    const primary = snap.primary_face;
    if (!primary) {
      el.identityCard.classList.add('is-offline');
      setClassToken(el.identityDot, STATUS_TOKENS, 'offline');
      setText(el.identityStatus, 'No face detected');
      setText(el.identityName, '—');
      el.identityAlias.hidden = true;
      updateThumbnail(null);

      setConfidence(0);
      setQuality(0);
      el.enrollmentBlock.hidden = true;
    } else {
      el.identityCard.classList.remove('is-offline');
      setClassToken(el.identityDot, STATUS_TOKENS, primary.status_token);
      setText(el.identityStatus, primary.status_label);
      setText(el.identityName, primary.display_name || primary.identity || 'Unknown');

      // Show the internal Person-N id as alias if a display name is set
      if (primary.display_name &&
          primary.identity &&
          primary.display_name !== primary.identity) {
        el.identityAlias.hidden = false;
        setText(el.identityAlias, primary.identity);
      } else {
        el.identityAlias.hidden = true;
      }

      updateThumbnail(primary);
      setConfidence(primary.confidence);
      setQuality(primary.quality);

      // Enrollment progress (only visible while REFINING)
      if (primary.decision === 'REFINING' && primary.frame_count > 0) {
        const tgt = primary.consensus_target || 1;
        const pct = Math.min(100, (primary.frame_count / tgt) * 100);
        el.enrollmentBlock.hidden = false;
        el.enrollFill.style.width = `${pct}%`;
        setText(el.enrollFrac, `${primary.frame_count}/${tgt}`);
      } else {
        el.enrollmentBlock.hidden = true;
      }
    }

    // Metadata card
    const db = snap.database;
    if (db) {
      setText(el.metaEnrolled, db.enrolled_label || '—');
      setText(el.metaLastseen, db.last_seen_label || '—');
      setText(el.metaMatches,  db.total_matches != null ? db.total_matches : '—');
    } else {
      setText(el.metaEnrolled, '—');
      setText(el.metaLastseen, '—');
      setText(el.metaMatches,  '—');
    }

    // Session label
    const sess = snap.session;
    if (sess) {
      const label = sess.state === 'ACTIVE'
        ? `Active · started ${sess.started_label}`
        : sess.state.charAt(0) + sess.state.slice(1).toLowerCase();
      setText(el.metaSession, label);
    } else {
      setText(el.metaSession, 'Idle');
    }

    // Transcript
    applyTranscript(snap.transcript || {});

    // Recent events
    applyEvents(snap.recent_events || []);
  }

  function setConfidence(v) {
    const num = Number.isFinite(v) ? v : 0;
    const pct = Math.max(0, Math.min(100, num * 100));
    el.confFill.style.width = `${pct}%`;
    setClassToken(el.confFill, ['low', 'medium', 'high'],
      pct >= 70 ? 'high' : pct >= 40 ? 'medium' : pct > 0 ? 'low' : null);
    setText(el.confValue, num > 0 ? pctLabel(num) : '—');
  }

  function setQuality(v) {
    const num = Number.isFinite(v) ? v : 0;
    const pct = Math.max(0, Math.min(100, num * 100));
    el.qualFill.style.width = `${pct}%`;
    setText(el.qualValue, num > 0 ? qualLabel(num) : '—');
  }

  // ── Transcript reconciliation ───────────────────────────────────────────
  let renderedLineCount = 0;
  let lastLineSignature = '';

  function applyTranscript(ts) {
    if (ts.is_listening) {
      el.listeningPill.hidden = false;
    } else {
      el.listeningPill.hidden = true;
    }

    const lines = Array.isArray(ts.lines) ? ts.lines : [];
    const signature = lines.join('\u0001');
    if (signature !== lastLineSignature) {
      // Only rebuild when the line list actually changed
      el.transcriptScroll.innerHTML = '';
      if (lines.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'transcript-empty';
        empty.textContent = ts.error
          ? `Speech error: ${ts.error}`
          : 'No transcript yet. Transcription starts automatically during an active session.';
        el.transcriptScroll.appendChild(empty);
      } else {
        for (const line of lines) {
          const div = document.createElement('div');
          div.className = 'transcript-line';
          div.textContent = line;
          el.transcriptScroll.appendChild(div);
        }
        el.transcriptScroll.scrollTop = el.transcriptScroll.scrollHeight;
      }
      lastLineSignature = signature;
      renderedLineCount = lines.length;
    }

    if (ts.partial) {
      el.transcriptPartial.hidden = false;
      setText(el.transcriptPartial, ts.partial);
    } else {
      el.transcriptPartial.hidden = true;
    }
  }

  // ── Events list reconciliation ──────────────────────────────────────────
  let lastEventsSignature = '';

  function applyEvents(events) {
    const sig = events.map(e => `${e.event_type}|${e.created_at}|${e.content}`).join('\u0001');
    if (sig === lastEventsSignature) return;
    lastEventsSignature = sig;

    if (events.length === 0) {
      el.eventsCard.hidden = true;
      return;
    }
    el.eventsCard.hidden = false;
    el.eventsList.innerHTML = '';
    for (const evt of events) {
      const li = document.createElement('li');
      const type = document.createElement('span');
      type.className = 'ev-type';
      type.textContent = evt.event_type || 'event';
      const content = document.createElement('span');
      content.className = 'ev-content';
      content.textContent = evt.content || '';
      content.title = evt.content || '';
      const age = document.createElement('span');
      age.className = 'ev-age';
      age.textContent = evt.age_label || '';
      li.appendChild(type);
      li.appendChild(content);
      li.appendChild(age);
      el.eventsList.appendChild(li);
    }
  }

  // ── Poll loop ───────────────────────────────────────────────────────────

  let consecutiveErrors = 0;

  async function pollOnce() {
    try {
      const res = await fetch(API_STATE, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const snap = await res.json();
      consecutiveErrors = 0;
      applyState(snap);
    } catch (err) {
      consecutiveErrors += 1;
      if (consecutiveErrors === 1) {
        setClassToken(el.systemDot, STATUS_TOKENS, 'quality');
        setText(el.systemStatus, 'Disconnected');
      }
    }
  }

  function schedule() {
    const delay = consecutiveErrors > 0 ? RETRY_BACKOFF_MS : POLL_INTERVAL_MS;
    setTimeout(async () => {
      await pollOnce();
      schedule();
    }, delay);
  }

  // ── Video recovery: if the MJPEG socket dies, re-request it ─────────────
  el.video.addEventListener('error', () => {
    setTimeout(() => {
      el.video.src = `/api/video_feed?ts=${Date.now()}`;
    }, 1000);
  });

  // ── Pipeline resume on entry ────────────────────────────────────────────
  // The dashboard pauses the camera/ML pipeline whenever it's open; visiting
  // the live page should always re-arm it. Idempotent server-side, so safe
  // to fire on every load.
  async function resumePipeline() {
    try {
      await fetch('/api/pipeline/resume', { method: 'POST', cache: 'no-store' });
      // Re-arm the MJPEG <img> in case the previous pause closed it.
      el.video.src = `/api/video_feed?ts=${Date.now()}`;
    } catch (_) {
      // Non-fatal — the polling loop will surface a "Disconnected" pill if
      // the server is actually down.
    }
  }

  // Kick off
  resumePipeline().finally(() => pollOnce().finally(schedule));
})();
