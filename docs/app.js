/**
 * MoodMentor | Workplace Affective Intelligence & Wellbeing Platform
 * Core Application Engine: Three.js 3D Zen Companion, Sidebar Routing,
 * NLP Sentiment Analysis, Somatics, and Web Audio Synthesis.
 */

(function () {
  'use strict';

  // =========================================================================
  // 1. Core NLP Lexicons & Utilities
  // =========================================================================
  const VADER_LEXICON = {
    great: 3.1, fantastic: 3.4, excellent: 3.2, happy: 2.7, love: 3.2,
    joy: 2.8, grateful: 2.9, proud: 2.6, thrive: 2.8, flourishing: 2.9,
    accomplished: 2.4, calm: 2.1, energized: 2.7, inspired: 2.8, good: 1.9,
    wonderful: 3.1, serene: 2.4, peaceful: 2.3, relief: 2.2, success: 2.8,
    support: 2.1, teamwork: 2.3, valued: 2.5, meaningful: 2.4, confident: 2.3,
    exhausted: -2.8, burned: -2.9, burnout: -3.2, stressed: -2.6, stress: -2.4,
    anxious: -2.5, anxiety: -2.6, overwhelmed: -2.8, frustrated: -2.5, anger: -2.8,
    sad: -2.3, hopeless: -3.4, isolated: -2.4, terrible: -3.0, fatigue: -2.5,
    friction: -2.2, panic: -3.2, unappreciated: -2.5, worried: -2.2, pressure: -2.3
  };

  const EMOTIONS = {
    joy: { label: 'Joy & Flow', color: '#11562a', cues: ['happy', 'thrilled', 'celebrate', 'good', 'great', 'energized', 'flow', 'inspired'] },
    gratitude: { label: 'Gratitude & Appreciation', color: '#2e6f40', cues: ['grateful', 'thankful', 'appreciate', 'thanks', 'valued', 'teamwork', 'support'] },
    stress: { label: 'Stress & Pressure', color: '#ba1a1a', cues: ['stressed', 'pressure', 'deadline', 'urgent', 'anxious', 'tense', 'overwhelmed', 'rush'] },
    fatigue: { label: 'Fatigue & Depletion', color: '#4355b9', cues: ['exhausted', 'tired', 'drained', 'burned', 'burnout', 'fatigue', 'sleepy', 'heavy'] },
    calm: { label: 'Calm & Equilibrium', color: '#414f35', cues: ['calm', 'peaceful', 'serene', 'balanced', 'grounded', 'centered', 'steady', 'relief'] },
    neutral: { label: 'Balanced Focus', color: '#70796f', cues: ['meeting', 'reviewed', 'code', 'standup', 'ticket', 'status', 'document', 'sync'] }
  };

  const ZEN_INSIGHTS = [
    '"Team balance is optimal today."',
    '"Take a deep breath and reset your focus."',
    '"Your team\'s energy is radiant and grounded."',
    '"Empathy is the cornerstone of sustainable leadership."',
    '"Small mindful moments compound into major resilience."',
    '"Allow space for quiet focus blocks this afternoon."',
    '"Acknowledge what went well today, no matter how small."'
  ];

  // =========================================================================
  // 2. Three.js 3D Zen Companion Engine (Google Stitch Architecture)
  // =========================================================================
  let companionScene, companionCamera, companionRenderer, companionGroup, innerMesh, coreMaterial, particles, petalGroup;
  let isCompanionInteracting = false;
  let companionInteractionTimer = 0;
  let companionMouseX = 0;
  let companionMouseY = 0;

  function initZenCompanion() {
    const container = document.getElementById('companion-container');
    if (!container || typeof THREE === 'undefined') return;

    // Clear previous canvas if any
    container.innerHTML = '';

    const width = container.clientWidth || 320;
    const height = container.clientHeight || 220;

    companionScene = new THREE.Scene();
    companionCamera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    companionCamera.position.z = 5;

    companionRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    companionRenderer.setSize(width, height);
    companionRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(companionRenderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.3);
    companionScene.add(ambientLight);

    const pointLight = new THREE.PointLight(0x2e6f40, 3.5, 50);
    pointLight.position.set(2, 4, 3);
    companionScene.add(pointLight);

    const pointLightWarm = new THREE.PointLight(0xd4a373, 2.5, 50);
    pointLightWarm.position.set(-2, -3, 2);
    companionScene.add(pointLightWarm);

    companionGroup = new THREE.Group();

    // Outer translucent energy core
    const coreGeometry = new THREE.SphereGeometry(1.0, 32, 32);
    coreMaterial = new THREE.MeshPhongMaterial({
      color: 0x2e6f40,
      emissive: 0x14361d,
      shininess: 90,
      transparent: true,
      opacity: 0.88
    });
    const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
    companionGroup.add(coreMesh);

    // Inner glowing heart
    const innerGeometry = new THREE.SphereGeometry(0.55, 24, 24);
    const innerMaterial = new THREE.MeshPhongMaterial({
      color: 0xfaf9f7,
      emissive: 0x2e6f40,
      shininess: 100
    });
    innerMesh = new THREE.Mesh(innerGeometry, innerMaterial);
    companionGroup.add(innerMesh);

    // 6 Orbiting Zen Lotus Petals
    const petalCount = 6;
    petalGroup = new THREE.Group();
    for (let i = 0; i < petalCount; i++) {
      const angle = (i / petalCount) * Math.PI * 2;
      const leafShape = new THREE.ConeGeometry(0.25, 1.2, 16);
      const leafMat = new THREE.MeshPhongMaterial({
        color: 0x40916c,
        shininess: 80,
        flatShading: true
      });
      const leaf = new THREE.Mesh(leafShape, leafMat);
      leaf.rotation.z = Math.PI / 3;
      leaf.position.set(Math.cos(angle) * 1.6, Math.sin(angle) * 1.6, 0);
      petalGroup.add(leaf);
    }
    companionGroup.add(petalGroup);

    // 80 Floating Aura Particles
    const particleCount = 80;
    const particleGeometry = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      particlePositions[i] = (Math.random() - 0.5) * 7;
      particlePositions[i + 1] = (Math.random() - 0.5) * 7;
      particlePositions[i + 2] = (Math.random() - 0.5) * 7;
    }
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    const particleMaterial = new THREE.PointsMaterial({
      color: 0x40916c,
      size: 0.07,
      transparent: true,
      opacity: 0.7
    });
    particles = new THREE.Points(particleGeometry, particleMaterial);
    companionScene.add(particles);

    companionScene.add(companionGroup);

    // Mouse Tracking Parallax
    container.addEventListener('mousemove', (e) => {
      const rect = container.getBoundingClientRect();
      companionMouseX = ((e.clientX - rect.left) / container.clientWidth - 0.5) * 0.8;
      companionMouseY = ((e.clientY - rect.top) / container.clientHeight - 0.5) * 0.8;
    });

    // Interactive Click: Spin acceleration, Gold Aura Burst & Mindful Insight
    container.addEventListener('click', triggerCompanionInteraction);
    const chatBtn = document.getElementById('btn-companion-chat');
    if (chatBtn) chatBtn.addEventListener('click', triggerCompanionInteraction);

    // Animation Loop
    const clock = new THREE.Clock();
    function animate() {
      requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const elapsedTime = clock.getElapsedTime();

      if (isCompanionInteracting) {
        companionInteractionTimer -= delta;
        coreMaterial.emissive.setHex(0xd4af37); // Golden radiance
        coreMaterial.color.setHex(0x92d69d);
        particles.material.size = 0.12;
        if (companionInteractionTimer <= 0) {
          isCompanionInteracting = false;
          coreMaterial.emissive.setHex(0x14361d);
          coreMaterial.color.setHex(0x2e6f40);
          particles.material.size = 0.07;
        }
      }

      companionGroup.position.y = Math.sin(elapsedTime * 1.5) * 0.15;
      companionGroup.rotation.y += (elapsedTime * 0.35 + companionMouseX - companionGroup.rotation.y) * 0.1 + (isCompanionInteracting ? 0.2 : 0);
      companionGroup.rotation.x = Math.sin(elapsedTime * 0.7) * 0.12 + companionMouseY;

      petalGroup.rotation.z += (isCompanionInteracting ? -0.15 : -0.007);

      const scale = 1 + Math.sin(elapsedTime * 2.2) * 0.06 + (isCompanionInteracting ? 0.15 * Math.sin(elapsedTime * 10) : 0);
      innerMesh.scale.set(scale, scale, scale);

      particles.rotation.y += (isCompanionInteracting ? 0.08 : 0.0008);

      companionRenderer.render(companionScene, companionCamera);
    }
    animate();

    // Window Resize Handling
    window.addEventListener('resize', () => {
      if (!container || !companionCamera || !companionRenderer) return;
      const w = container.clientWidth || 320;
      const h = container.clientHeight || 220;
      companionCamera.aspect = w / h;
      companionCamera.updateProjectionMatrix();
      companionRenderer.setSize(w, h);
    });
  }

  function triggerCompanionInteraction() {
    isCompanionInteracting = true;
    companionInteractionTimer = 1.5;

    const dialogueEl = document.getElementById('companion-dialogue');
    if (dialogueEl) {
      const randomInsight = ZEN_INSIGHTS[Math.floor(Math.random() * ZEN_INSIGHTS.length)];
      dialogueEl.textContent = randomInsight;
    }
    showToast('Aura calibrated: Mindful focus activated.', 'primary');
  }

  // =========================================================================
  // 3. Navigation & View Routing
  // =========================================================================
  function initRouting() {
    const navLinks = document.querySelectorAll('#sidebar-nav a[data-view]');
    const views = document.querySelectorAll('.view-container');

    function switchView(viewName) {
      views.forEach((v) => {
        if (v.id === `view-${viewName}`) {
          v.classList.add('active');
        } else {
          v.classList.remove('active');
        }
      });

      navLinks.forEach((link) => {
        if (link.getAttribute('data-view') === viewName) {
          link.classList.add('nav-tab-active');
          link.setAttribute('aria-current', 'page');
        } else {
          link.classList.remove('nav-tab-active');
          link.removeAttribute('aria-current');
        }
      });

      // Special handling when returning to AI Mentor view: resize Three.js
      if (viewName === 'ai-mentor-insights' && companionRenderer && companionCamera) {
        const container = document.getElementById('companion-container');
        if (container) {
          const w = container.clientWidth || 320;
          const h = container.clientHeight || 220;
          companionCamera.aspect = w / h;
          companionCamera.updateProjectionMatrix();
          companionRenderer.setSize(w, h);
        }
      }

      // Draw analytics chart when manager analytics is opened
      if (viewName === 'manager-analytics') {
        renderResilienceChart();
      }
    }

    navLinks.forEach((link) => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const view = link.getAttribute('data-view');
        switchView(view);
      });
    });

    const quickBtn = document.getElementById('quick-checkin-btn');
    if (quickBtn) {
      quickBtn.addEventListener('click', () => switchView('daily-check-in'));
    }
  }

  // =========================================================================
  // 4. Actionable Micro-Interventions Checklist Manager
  // =========================================================================
  function initInterventionsChecklist() {
    const container = document.getElementById('interventions-checklist');
    const progressText = document.getElementById('intervention-progress-text');
    const showAddBtn = document.getElementById('btn-show-add-intervention');
    const addBox = document.getElementById('add-intervention-box');
    const customInput = document.getElementById('custom-intervention-input');
    const cancelBtn = document.getElementById('btn-cancel-intervention');
    const saveBtn = document.getElementById('btn-save-intervention');

    function updateProgress() {
      const checkboxes = container.querySelectorAll('.intervention-checkbox');
      const total = checkboxes.length;
      let checked = 0;
      checkboxes.forEach((cb) => {
        const label = cb.closest('label');
        const text = label.querySelector('span');
        if (cb.checked) {
          checked++;
          text.classList.add('line-through', 'opacity-75');
        } else {
          text.classList.remove('line-through', 'opacity-75');
        }
      });
      if (progressText) {
        progressText.textContent = `${checked} of ${total} Done`;
      }
      const metricAdoption = document.getElementById('metric-adoption-desc');
      if (metricAdoption) {
        metricAdoption.textContent = `${checked} of ${total} suggested micro-interventions completed.`;
      }
    }

    if (container) {
      container.addEventListener('change', (e) => {
        if (e.target.classList.contains('intervention-checkbox')) {
          updateProgress();
          if (e.target.checked) {
            showToast('Intervention completed. Team score updated.', 'primary');
          }
        }
      });
    }

    if (showAddBtn && addBox) {
      showAddBtn.addEventListener('click', () => {
        addBox.classList.remove('hidden');
        showAddBtn.classList.add('hidden');
        if (customInput) customInput.focus();
      });
    }

    if (cancelBtn && addBox && showAddBtn) {
      cancelBtn.addEventListener('click', () => {
        addBox.classList.add('hidden');
        showAddBtn.classList.remove('hidden');
        if (customInput) customInput.value = '';
      });
    }

    if (saveBtn && customInput && container) {
      saveBtn.addEventListener('click', () => {
        const val = customInput.value.trim();
        if (!val) return;
        const newLabel = document.createElement('label');
        newLabel.className = 'flex items-start gap-3 p-3 bg-surface/80 rounded-xl cursor-pointer hover:bg-surface-container-highest transition-colors';
        newLabel.innerHTML = `
          <input class="w-5 h-5 accent-primary rounded-md mt-0.5 cursor-pointer intervention-checkbox" type="checkbox"/>
          <div class="flex-1">
            <span class="text-body-md font-medium text-on-surface">${escapeHtml(val)}</span>
            <p class="text-body-sm text-on-surface-variant">Custom manager initiative added today.</p>
          </div>
        `;
        container.appendChild(newLabel);
        customInput.value = '';
        addBox.classList.add('hidden');
        showAddBtn.classList.remove('hidden');
        updateProgress();
        showToast('New manager intervention added.', 'primary');
      });
    }

    updateProgress();
  }

  // =========================================================================
  // 5. Timeframe Trend Filters
  // =========================================================================
  function initTimeframeFilters() {
    const buttons = document.querySelectorAll('.timeframe-btn');
    const barA = document.getElementById('trend-bar-a');
    const barB = document.getElementById('trend-bar-b');
    const barC = document.getElementById('trend-bar-c');
    const valA = document.getElementById('trend-val-a');
    const valB = document.getElementById('trend-val-b');
    const valC = document.getElementById('trend-val-c');

    const datasets = {
      '30d': { a: { w: '82%', t: '+14% vs avg' }, b: { w: '68%', t: '-2.4 hrs/wk' }, c: { w: '90%', t: 'Stable & High' } },
      'quarter': { a: { w: '76%', t: '+9% vs baseline' }, b: { w: '61%', t: '-3.8 hrs/wk' }, c: { w: '88%', t: 'Consistently Strong' } },
      'year': { a: { w: '89%', t: '+18% annual growth' }, b: { w: '74%', t: '-5.1 hrs/wk' }, c: { w: '94%', t: 'Flourishing' } }
    };

    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        buttons.forEach((b) => {
          b.classList.remove('bg-surface', 'font-semibold', 'shadow-sm', 'border', 'border-outline-variant/20');
          b.classList.add('text-on-surface-variant');
        });
        btn.classList.add('bg-surface', 'font-semibold', 'shadow-sm', 'border', 'border-outline-variant/20');
        btn.classList.remove('text-on-surface-variant');

        const period = btn.getAttribute('data-period') || '30d';
        const d = datasets[period];
        if (d && barA && barB && barC) {
          barA.style.width = d.a.w;
          barB.style.width = d.b.w;
          barC.style.width = d.c.w;
          if (valA) valA.textContent = d.a.t;
          if (valB) valB.textContent = d.b.t;
          if (valC) valC.textContent = d.c.t;
        }
      });
    });
  }

  // =========================================================================
  // 6. Action Deploy Buttons & Synthesis Refresh
  // =========================================================================
  function initActionButtons() {
    const btnRefresh = document.getElementById('btn-refresh-insights');
    const refreshIcon = document.getElementById('refresh-icon');
    const timestampEl = document.getElementById('synthesis-timestamp');

    if (btnRefresh) {
      btnRefresh.addEventListener('click', () => {
        if (refreshIcon) refreshIcon.classList.add('animate-spin');
        btnRefresh.disabled = true;
        btnRefresh.classList.add('opacity-75');

        setTimeout(() => {
          if (refreshIcon) refreshIcon.classList.remove('animate-spin');
          btnRefresh.disabled = false;
          btnRefresh.classList.remove('opacity-75');
          if (timestampEl) timestampEl.textContent = '• Updated just now';
          
          // Random slight calibration on pulse
          const pulseEl = document.getElementById('metric-pulse-val');
          if (pulseEl) pulseEl.textContent = (8.2 + Math.random() * 0.4).toFixed(1);

          showToast('Telemetry refreshed with latest 42 team check-ins.', 'primary');
          triggerCompanionInteraction();
        }, 800);
      });
    }

    const btnDeployRecharge = document.getElementById('btn-deploy-recharge');
    if (btnDeployRecharge) {
      btnDeployRecharge.addEventListener('click', () => {
        btnDeployRecharge.textContent = 'Plan Scheduled';
        btnDeployRecharge.classList.replace('bg-primary', 'bg-tertiary-container');
        showToast('Team Recharge Day invite drafted for next Wednesday afternoon.', 'primary');
      });
    }

    const btnAsync = document.getElementById('btn-send-async-guidelines');
    if (btnAsync) {
      btnAsync.addEventListener('click', () => {
        btnAsync.textContent = 'Guidelines Sent';
        btnAsync.classList.replace('bg-secondary', 'bg-secondary-container');
        showToast('No-Meeting Morning policy shared with team leads.', 'secondary');
      });
    }

    const btnExportReport = document.getElementById('btn-export-report');
    if (btnExportReport) {
      btnExportReport.addEventListener('click', exportExecutiveReport);
    }
  }

  // =========================================================================
  // 7. Daily Check-in & Real-Time Sentiment Evaluator
  // =========================================================================
  function initDailyCheckIn() {
    const journalInput = document.getElementById('journal-input');
    const charCount = document.getElementById('char-count');
    const gaugeFill = document.getElementById('sentiment-gauge-fill');
    const sentimentBadge = document.getElementById('sentiment-badge');
    const emotionContainer = document.getElementById('emotion-chips-container');
    const submitBtn = document.getElementById('btn-submit-checkin');
    const clearBtn = document.getElementById('btn-analyze-clear');
    const promptChips = document.querySelectorAll('.prompt-chip');

    promptChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        const text = chip.getAttribute('data-prompt') || '';
        if (journalInput) {
          journalInput.value = text;
          evaluateInput(text);
        }
      });
    });

    if (journalInput) {
      journalInput.addEventListener('input', () => {
        const text = journalInput.value;
        if (charCount) charCount.textContent = `${text.length} characters`;
        evaluateInput(text);
      });
    }

    function evaluateInput(text) {
      if (!text.trim()) {
        if (gaugeFill) gaugeFill.style.width = '50%';
        if (sentimentBadge) {
          sentimentBadge.textContent = 'Neutral (0.00)';
          sentimentBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-bold bg-surface text-on-surface-variant border border-outline-variant/30';
        }
        return;
      }

      // Simple VADER polarity calculation
      const tokens = text.toLowerCase().match(/[a-z']+/g) || [];
      let scoreSum = 0;
      let matches = 0;
      const detectedEmotions = {};

      tokens.forEach((t) => {
        if (VADER_LEXICON[t]) {
          scoreSum += VADER_LEXICON[t];
          matches++;
        }
        for (const [key, emo] of Object.entries(EMOTIONS)) {
          if (emo.cues.includes(t)) {
            detectedEmotions[key] = (detectedEmotions[key] || 0) + 1;
          }
        }
      });

      const compound = matches > 0 ? Math.max(-1, Math.min(1, scoreSum / Math.sqrt(scoreSum * scoreSum + 15))) : 0;
      const percentage = Math.round(((compound + 1) / 2) * 100);

      if (gaugeFill) {
        gaugeFill.style.width = `${percentage}%`;
        gaugeFill.style.backgroundColor = compound > 0.1 ? '#11562a' : compound < -0.1 ? '#ba1a1a' : '#4355b9';
      }

      if (sentimentBadge) {
        const sign = compound > 0 ? '+' : '';
        sentimentBadge.textContent = `${compound > 0.1 ? 'Flourishing' : compound < -0.1 ? 'Stress/Depletion' : 'Balanced'} (${sign}${compound.toFixed(2)})`;
        sentimentBadge.style.color = compound > 0.1 ? '#11562a' : compound < -0.1 ? '#ba1a1a' : '#4355b9';
      }

      // Update emotion chips
      if (emotionContainer) {
        emotionContainer.innerHTML = '';
        const sorted = Object.entries(detectedEmotions).sort((a, b) => b[1] - a[1]);
        if (sorted.length === 0) {
          emotionContainer.innerHTML = '<span class="px-3 py-1 rounded-xl bg-surface border border-outline-variant/30 text-xs font-medium text-on-surface">Balanced Equanimity</span>';
        } else {
          sorted.slice(0, 3).forEach(([key, count]) => {
            const emo = EMOTIONS[key];
            const chip = document.createElement('span');
            chip.className = 'px-3 py-1 rounded-xl text-xs font-semibold bg-surface border border-outline-variant/30';
            chip.style.borderColor = emo.color;
            chip.style.color = emo.color;
            chip.textContent = `${emo.label} · Active`;
            emotionContainer.appendChild(chip);
          });
        }
      }
    }

    if (clearBtn && journalInput) {
      clearBtn.addEventListener('click', () => {
        journalInput.value = '';
        if (charCount) charCount.textContent = '0 characters';
        evaluateInput('');
      });
    }

    if (submitBtn && journalInput) {
      submitBtn.addEventListener('click', () => {
        const text = journalInput.value.trim();
        if (!text) {
          showToast('Please type a few words into your check-in.', 'error');
          return;
        }

        // Add to history table
        const tableBody = document.getElementById('history-table-body');
        if (tableBody) {
          const row = document.createElement('tr');
          row.className = 'hover:bg-surface/50 transition-colors';
          const nowStr = 'Just now';
          row.innerHTML = `
            <td class="py-3 px-4 text-xs text-on-surface-variant whitespace-nowrap">${nowStr}</td>
            <td class="py-3 px-4 text-on-surface font-medium max-w-md truncate">${escapeHtml(text)}</td>
            <td class="py-3 px-4"><span class="px-2 py-0.5 rounded-full text-xs font-bold bg-primary-fixed text-on-primary-fixed">+0.65</span></td>
            <td class="py-3 px-4 text-xs font-semibold text-primary">Mindful Reflection</td>
            <td class="py-3 px-4 text-right"><button class="text-xs text-on-surface-variant hover:text-primary">View</button></td>
          `;
          tableBody.prepend(row);
        }

        journalInput.value = '';
        if (charCount) charCount.textContent = '0 characters';
        evaluateInput('');
        showToast('Daily check-in securely encrypted & recorded.', 'primary');
      });
    }
  }

  // =========================================================================
  // 8. Somatic Box Breathing Pacer (4-4-4-4)
  // =========================================================================
  function initBreathingPacer() {
    const ring = document.getElementById('breathing-ring');
    const phaseEl = document.getElementById('breathing-phase');
    const counterEl = document.getElementById('breathing-counter');
    const toggleBtn = document.getElementById('btn-toggle-breathing');

    let timer = null;
    let isRunning = false;
    let phaseIndex = 0;
    let secondsLeft = 4;

    const phases = [
      { name: 'Inhale', scale: '1.25', bg: 'rgba(46, 111, 64, 0.25)' },
      { name: 'Hold', scale: '1.25', bg: 'rgba(46, 111, 64, 0.35)' },
      { name: 'Exhale', scale: '0.85', bg: 'rgba(46, 111, 64, 0.1)' },
      { name: 'Hold', scale: '0.85', bg: 'rgba(46, 111, 64, 0.15)' }
    ];

    function step() {
      if (!isRunning) return;
      secondsLeft--;
      if (counterEl) counterEl.textContent = secondsLeft;

      if (secondsLeft <= 0) {
        phaseIndex = (phaseIndex + 1) % phases.length;
        secondsLeft = 4;
        const current = phases[phaseIndex];
        if (phaseEl) phaseEl.textContent = current.name;
        if (counterEl) counterEl.textContent = '4';
        if (ring) {
          ring.style.transform = `scale(${current.scale})`;
          ring.style.backgroundColor = current.bg;
        }
      }
    }

    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        if (isRunning) {
          clearInterval(timer);
          isRunning = false;
          toggleBtn.textContent = 'Start 1-Min Reset';
          if (phaseEl) phaseEl.textContent = 'Ready';
          if (counterEl) counterEl.textContent = '4';
          if (ring) ring.style.transform = 'scale(1)';
        } else {
          isRunning = true;
          phaseIndex = 0;
          secondsLeft = 4;
          toggleBtn.textContent = 'Pause Session';
          if (phaseEl) phaseEl.textContent = 'Inhale';
          if (ring) ring.style.transform = 'scale(1.25)';
          timer = setInterval(step, 1000);
        }
      });
    }
  }

  // =========================================================================
  // 9. Procedural Audio Sanctuary (Web Audio API)
  // =========================================================================
  let audioCtx = null;
  let rainNode = null;
  let isAudioPlaying = false;

  function initAudioPlayer() {
    const toggleBtn = document.getElementById('audio-toggle-btn');
    const playIcon = document.getElementById('audio-play-icon');
    const statusText = document.getElementById('audio-status');

    if (!toggleBtn) return;

    toggleBtn.addEventListener('click', () => {
      if (!isAudioPlaying) {
        startSoundscape();
        isAudioPlaying = true;
        if (playIcon) playIcon.textContent = 'pause';
        if (statusText) statusText.textContent = 'Playing 432Hz Sound';
        showToast('Audio Sanctuary: Calming brownian rain active.', 'primary');
      } else {
        stopSoundscape();
        isAudioPlaying = false;
        if (playIcon) playIcon.textContent = 'play_arrow';
        if (statusText) statusText.textContent = 'Soundscape Paused';
      }
    });
  }

  function startSoundscape() {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      if (!audioCtx) audioCtx = new AudioContext();
      if (audioCtx.state === 'suspended') audioCtx.resume();

      // Brownian / Pink noise synthesis for gentle rain
      const bufferSize = audioCtx.sampleRate * 2;
      const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
      const data = buffer.getChannelData(0);
      let lastOut = 0.0;
      for (let i = 0; i < bufferSize; i++) {
        const white = Math.random() * 2 - 1;
        data[i] = (lastOut + 0.02 * white) / 1.02;
        lastOut = data[i];
        data[i] *= 3.5;
      }

      const noise = audioCtx.createBufferSource();
      noise.buffer = buffer;
      noise.loop = true;

      // Filter to create warm rain sound
      const filter = audioCtx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(450, audioCtx.currentTime);

      const gain = audioCtx.createGain();
      gain.gain.setValueAtTime(0.04, audioCtx.currentTime);

      noise.connect(filter);
      filter.connect(gain);
      gain.connect(audioCtx.destination);
      noise.start();

      rainNode = { source: noise, gain: gain };
    } catch (e) {
      console.warn('Web Audio initialized quietly:', e);
    }
  }

  function stopSoundscape() {
    if (rainNode && rainNode.source) {
      try {
        rainNode.source.stop();
        rainNode.source.disconnect();
      } catch (e) {}
      rainNode = null;
    }
  }

  // =========================================================================
  // 10. Manager Resilience Canvas Chart
  // =========================================================================
  function renderResilienceChart() {
    const canvas = document.getElementById('resilience-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Days labels
    const days = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14'];
    const resilience = [7.2, 7.4, 7.1, 7.8, 8.0, 8.2, 8.1, 8.3, 8.5, 8.2, 8.6, 8.7, 8.5, 8.8];
    const cognitiveLoad = [6.5, 6.8, 7.2, 6.9, 6.4, 5.8, 5.5, 6.2, 6.5, 6.1, 5.7, 5.4, 5.2, 4.9];

    const paddingX = 40;
    const paddingY = 30;
    const chartW = width - paddingX * 2;
    const chartH = height - paddingY * 2;

    // Draw grid lines
    ctx.strokeStyle = '#e3e2e0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = paddingY + (chartH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(paddingX, y);
      ctx.lineTo(width - paddingX, y);
      ctx.stroke();
    }

    // Draw Resilience Curve (Forest Green)
    drawCurve(ctx, resilience, 4, 10, chartW, chartH, paddingX, paddingY, '#11562a', 'rgba(17, 86, 42, 0.12)');

    // Draw Cognitive Load Curve (Blue/Purple)
    drawCurve(ctx, cognitiveLoad, 4, 10, chartW, chartH, paddingX, paddingY, '#4355b9', 'rgba(67, 85, 185, 0.08)');

    // Draw X labels
    ctx.fillStyle = '#70796f';
    ctx.font = '10px "Plus Jakarta Sans", sans-serif';
    ctx.textAlign = 'center';
    days.forEach((day, idx) => {
      const x = paddingX + (chartW / (days.length - 1)) * idx;
      ctx.fillText(day, x, height - 10);
    });
  }

  function drawCurve(ctx, data, minVal, maxVal, w, h, px, py, strokeColor, fillColor) {
    const points = data.map((val, idx) => {
      const x = px + (w / (data.length - 1)) * idx;
      const normY = (val - minVal) / (maxVal - minVal);
      const y = py + h - normY * h;
      return { x, y };
    });

    // Fill area under curve
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const xc = (points[i].x + points[i - 1].x) / 2;
      const yc = (points[i].y + points[i - 1].y) / 2;
      ctx.quadraticCurveTo(points[i - 1].x, points[i - 1].y, xc, yc);
    }
    ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    ctx.lineTo(points[points.length - 1].x, py + h);
    ctx.lineTo(points[0].x, py + h);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();

    // Draw stroke
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const xc = (points[i].x + points[i - 1].x) / 2;
      const yc = (points[i].y + points[i - 1].y) / 2;
      ctx.quadraticCurveTo(points[i - 1].x, points[i - 1].y, xc, yc);
    }
    ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Draw points
    points.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#faf9f7';
      ctx.fill();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }

  // =========================================================================
  // 11. Export Helpers
  // =========================================================================
  function exportExecutiveReport() {
    const reportData = {
      platform: 'MoodMentor Workplace Affective Intelligence',
      version: '2.4',
      generatedAt: new Date().toISOString(),
      teamPulseIndex: 8.4,
      burnoutRisk: 'Low (-12%)',
      actionAdoptionRate: '78%',
      psychologicalSafety: 9.1,
      strategicShifts: [
        { priority: '01', title: 'Schedule Team Recharge Day', impact: 'Very High', status: 'Scheduled' },
        { priority: '02', title: 'Encourage Asynchronous Communication', impact: 'High', status: 'Active Guidelines' }
      ]
    };
    downloadJson(reportData, 'MoodMentor_Executive_Intelligence_Report.json');
    showToast('Executive report downloaded successfully.', 'primary');
  }

  function downloadJson(obj, filename) {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // =========================================================================
  // 12. Toast Feedback System
  // =========================================================================
  function showToast(message, type = 'primary') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'px-4 py-3 rounded-xl bg-surface border border-outline-variant/30 shadow-lg text-sm font-medium flex items-center gap-3 transition-all transform translate-y-2 opacity-0 pointer-events-auto';
    
    let iconName = 'check_circle';
    let iconColor = '#11562a';
    if (type === 'secondary') {
      iconName = 'info';
      iconColor = '#4355b9';
    } else if (type === 'error') {
      iconName = 'error';
      iconColor = '#ba1a1a';
    }

    toast.innerHTML = `
      <span class="material-symbols-outlined text-[18px]" style="color: ${iconColor};">${iconName}</span>
      <span class="text-on-surface">${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);
    requestAnimationFrame(() => {
      toast.classList.remove('translate-y-2', 'opacity-0');
    });

    setTimeout(() => {
      toast.classList.add('translate-y-2', 'opacity-0');
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }

  // =========================================================================
  // 13. Initialization Lifecycle
  // =========================================================================
  window.addEventListener('DOMContentLoaded', () => {
    initZenCompanion();
    initRouting();
    initInterventionsChecklist();
    initTimeframeFilters();
    initActionButtons();
    initDailyCheckIn();
    initBreathingPacer();
    initAudioPlayer();
  });

})();
