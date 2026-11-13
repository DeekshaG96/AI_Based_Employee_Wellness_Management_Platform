/**
 * MoodMentor | Enterprise Affective Wellness & Workplace Intelligence
 * Interactive Client Engine, NLP Analyzer, Procedural Web Audio & Mascot Animator
 */

(function () {
  'use strict';

  // =========================================================================
  // 1. Core Lexicons & NLP Telemetry Dictionaries
  // =========================================================================
  const VADER_LEXICON = {
    // Positive sentiment cues
    great: 3.1, fantastic: 3.4, excellent: 3.2, happy: 2.7, love: 3.2,
    joy: 2.8, grateful: 2.9, proud: 2.6, thrive: 2.8, flourishing: 2.9,
    accomplished: 2.4, calm: 2.1, energized: 2.7, inspired: 2.8, good: 1.9,
    wonderful: 3.1, serene: 2.4, peaceful: 2.3, relief: 2.2, success: 2.8,
    support: 2.1, teamwork: 2.3, valued: 2.5, meaningful: 2.4, confident: 2.3,
    // Negative sentiment cues
    exhausted: -2.8, burned: -2.9, burnout: -3.2, stressed: -2.6, stress: -2.4,
    anxious: -2.5, anxiety: -2.6, overwhelmed: -2.8, frustrated: -2.5, anger: -2.8,
    sad: -2.3, hopeless: -3.4, isolated: -2.4, terrified: -3.0, terrible: -3.0,
    depressed: -3.1, draining: -2.4, fatigue: -2.5, friction: -2.2, panic: -3.2,
    failure: -2.7, disrespected: -2.8, lonely: -2.4, miserable: -3.1, unappreciated: -2.5,
    bad: -2.1, hurt: -2.4, worried: -2.2, worry: -2.2, nervous: -2.1, afraid: -2.5
  };

  const BOOSTER_WORDS = {
    extremely: 0.293, very: 0.25, deeply: 0.27, highly: 0.24, completely: 0.28,
    totally: 0.26, absolutely: 0.29, truly: 0.22, really: 0.20, super: 0.25,
    barely: -0.22, slightly: -0.20, somewhat: -0.15, hardly: -0.25
  };

  const NEGATION_WORDS = new Set([
    'not', 'never', 'no', 'none', "don't", 'dont', "can't", 'cant',
    "won't", 'wont', "isn't", 'isnt', 'hardly', 'without', 'neither', 'barely'
  ]);

  const EMOTION_PROFILES = {
    joy: { name: 'Joy & Flourishing', color: '#10b981', cues: ['happy', 'delighted', 'joyful', 'ecstatic', 'thrilled', 'celebrate', 'smiled', 'fun', 'love', 'blessed', 'excited'] },
    gratitude: { name: 'Gratitude & Appreciation', color: '#0d9488', cues: ['grateful', 'thankful', 'appreciate', 'thank', 'thanks', 'valued', 'support', 'recognition', 'acknowledged'] },
    pride: { name: 'Pride & Achievement', color: '#3b82f6', cues: ['proud', 'accomplished', 'achieved', 'finished', 'shipped', 'released', 'success', 'milestone', 'won'] },
    stress: { name: 'Stress & Pressure', color: '#f59e0b', cues: ['stressed', 'pressure', 'deadline', 'urgent', 'anxious', 'tense', 'overwhelmed', 'rush', 'hectic', 'scared'] },
    fatigue: { name: 'Exhaustion & Fatigue', color: '#8b5cf6', cues: ['exhausted', 'tired', 'drained', 'burned', 'burnout', 'fatigue', 'sleepy', 'depleted', 'unfocused', 'weary'] },
    frustration: { name: 'Friction & Frustration', color: '#f43f5e', cues: ['frustrated', 'annoyed', 'irritated', 'angry', 'friction', 'blocked', 'unfair', 'bureaucracy', 'interrupted'] },
    sadness: { name: 'Low Mood & Isolation', color: '#64748b', cues: ['sad', 'isolated', 'lonely', 'disconnected', 'hopeless', 'down', 'gloomy', 'heavy', 'crying', 'unhappy'] },
    calm: { name: 'Calm & Equilibrium', color: '#06b6d4', cues: ['calm', 'peaceful', 'serene', 'balanced', 'grounded', 'centered', 'steady', 'relaxed', 'quiet'] },
    neutral: { name: 'Neutral Focus', color: '#94a3b8', cues: ['meeting', 'reviewed', 'code', 'standup', 'ticket', 'status', 'scheduled', 'document', 'updated', 'process'] }
  };

  const NUTRITION_DATABASE = {
    stress: {
      title: 'Cortisol Down-Regulation & GABA Balance Protocol',
      target: 'Inhibitory neurotransmitter modulation and adrenal axis recovery',
      nutrients: ['Magnesium Glycinate', 'L-Theanine', 'GABA Precursors', 'Vitamin B-Complex'],
      meals: {
        any: {
          breakfast: 'Warm steel-cut oats with crushed pumpkin seeds, wild blueberries, and organic kefir.',
          lunch: 'Wild Alaskan salmon bowl with avocado, wilted spinach, brown rice, and lemon-tahini dressing.',
          snack: 'Small handful of sprouted walnuts with chamomile-infused green tea.',
          dinner: 'Herb-roasted pasture-raised chicken breast with roasted asparagus and sweet potato wedges.'
        },
        vegetarian: {
          breakfast: 'Sprouted oat porridge with hemp seeds, chia, antioxidant berries, and organic greek yogurt.',
          lunch: 'Quinoa and black bean warm bowl with Hass avocado, pumpkin seeds, and steamed kale.',
          snack: 'Raw Brazil nuts (selenium rich) with a cup of ceremonial matcha.',
          dinner: 'Organic lentil curry with turmeric, spinach, coconut cream, and brown basmati rice.'
        },
        vegan: {
          breakfast: 'Chia-hemp seed pudding infused with almond milk, raw cacao nibs, and blueberries.',
          lunch: 'Tempeh Buddha bowl with roasted chickpeas, avocado, baby spinach, and sesame dressing.',
          snack: 'Sprouted sunflower seeds and pure chamomile lemon infusion.',
          dinner: 'Crispy baked organic tofu with sautéed bok choy, edamame, and forbidden black rice.'
        }
      },
      superfoods: [
        { food: 'Pumpkin Seeds', benefit: 'Highest natural source of bioavailable magnesium for neuromuscular relaxation.' },
        { food: 'Matcha / Green Tea', benefit: 'Concentrated L-Theanine induces calm alpha brainwaves without sedation.' },
        { food: 'Fermented Kefir', benefit: 'Microbial probiotics synthesize 90% of bodily serotonin precursors via vagus nerve.' }
      ],
      avoid: ['High-glycemic processed sugars that induce reactive hypoglycemia and cortisol spikes.', 'Excessive afternoon caffeine past 2 PM which elevates resting epinephrine.']
    },
    fatigue: {
      title: 'Mitochondrial Cellular Fuel & ATP Renewal Protocol',
      target: 'Cellular ATP production and slow-release low-glycemic glycemic stabilization',
      nutrients: ['CoQ10', 'Complex B12 & Folate', 'Iron', 'Electrolytes'],
      meals: {
        any: {
          breakfast: 'Poached free-range eggs on seeded sourdough toast with mashed avocado and microgreens.',
          lunch: 'Grilled lean turkey or chicken breast with roasted beets, goat cheese, and quinoa salad.',
          snack: 'Sliced green apple with organic almond butter and Ceylon cinnamon.',
          dinner: 'Pan-seared cod or salmon with steamed broccoli florets and roasted purple potatoes.'
        },
        vegetarian: {
          breakfast: 'Sprouted grain toast with pasture eggs or hummus, crushed pumpkin seeds, and cherry tomatoes.',
          lunch: 'Warm Moroccan spiced chickpea and roasted sweet potato bowl with tahini and baby spinach.',
          snack: 'Handful of roasted edamame beans and an organic cold-pressed ginger beetroot juice.',
          dinner: 'High-protein black bean and lentil chili with nutritional yeast and avocado slices.'
        },
        vegan: {
          breakfast: 'Scrambled organic firm tofu with baby spinach, nutritional yeast, and turmeric on sourdough.',
          lunch: 'Lentil and hemp-heart harvest bowl with roasted sweet potato, broccoli, and hemp pesto.',
          snack: 'Raw walnut halves and unsweetened cacao nibs.',
          dinner: 'Steamed edamame and pan-seared tempeh with sesame ginger dressing over quinoa.'
        }
      },
      superfoods: [
        { food: 'Beetroot & Greens', benefit: 'Dietary nitrates enhance nitric oxide vasodilation, improving mitochondrial oxygenation.' },
        { food: 'Chia & Hemp Seeds', benefit: 'Balanced 3:1 omega-3 to omega-6 ratio supports cellular membrane conductivity.' },
        { food: 'Raw Cacao', benefit: 'Rich in theobromine and bioflavonoids for sustained cerebral blood flow.' }
      ],
      avoid: ['Refined white flour pastries that trigger rapid mid-afternoon energy crashes.', 'Artificially sweetened energy drinks that disrupt gut microbiome equilibrium.']
    },
    joy: {
      title: 'Cognitive Flourishing & Neuro-Vitality Protocol',
      target: 'Brain-derived neurotrophic factor (BDNF) elevation and long-term neuroprotection',
      nutrients: ['Polyphenols', 'DHA Omega-3s', 'Choline', 'Curcumin'],
      meals: {
        any: {
          breakfast: 'Smoked salmon scramble with baby greens, cherry tomatoes, and fermented sourdough.',
          lunch: 'Mediterranean tuna or chicken salad with Kalamata olives, cucumbers, olive oil, and farro.',
          snack: 'Handful of organic blueberries and dark 85% single-origin dark chocolate.',
          dinner: 'Grilled rosemary trout or wild game with roasted Mediterranean vegetables and olive oil.'
        },
        vegetarian: {
          breakfast: 'Fluffy chia-seed pancakes with Greek yogurt, blackberry compote, and crushed pecans.',
          lunch: 'Mediterranean farro salad with feta, olives, cucumbers, artichoke hearts, and olive oil.',
          snack: 'Single-origin 85% dark cacao squares with antioxidant pomegranate arils.',
          dinner: 'Stuffed bell peppers with spiced lentils, toasted pine nuts, spinach, and goat cheese.'
        },
        vegan: {
          breakfast: 'Acai superfood bowl topped with flax seeds, raw pecans, fresh berries, and cacao nibs.',
          lunch: 'Mediterranean farro and white bean bowl with Kalamata olives, artichokes, and lemon vinaigrette.',
          snack: 'Fresh organic berries paired with raw unsalted macadamia nuts.',
          dinner: 'Roasted eggplant and lentil moussaka with cashew cream and fresh garden herbs.'
        }
      },
      superfoods: [
        { food: 'Wild Blueberries', benefit: 'Anthocyanins cross blood-brain barrier to accelerate synaptic neuroplasticity.' },
        { food: 'Extra Virgin Olive Oil', benefit: 'Oleocanthal delivers powerful neuro-protective anti-inflammatory signaling.' },
        { food: '85%+ Dark Cacao', benefit: 'Stimulates dopamine synthesis and enhances cognitive processing speed.' }
      ],
      avoid: ['Trans-fats and hydrogenated vegetable oils that impair neuronal membrane fluidity.', 'Excessive alcohol intake which suppresses REM sleep recovery.']
    }
  };

  // =========================================================================
  // 2. Animated Mascot ("Aura") Controller
  // =========================================================================
  const MascotController = {
    state: 'idle',
    svgElement: null,
    auraGlowElement: null,
    speechBubbleElement: null,
    moodTagElement: null,
    quoteElement: null,

    init() {
      this.svgElement = document.getElementById('mascotSvg');
      this.auraGlowElement = document.getElementById('mascotAuraGlow');
      this.speechBubbleElement = document.getElementById('mascotSpeechBubble');
      this.moodTagElement = document.getElementById('mascotMoodTag');
      this.quoteElement = document.getElementById('mascotQuote');

      if (this.svgElement) {
        this.svgElement.addEventListener('click', () => this.handleMascotClick());
      }
    },

    setMood(state, customQuote = null) {
      this.state = state;
      const eyeL = document.getElementById('mascotEyeL');
      const eyeR = document.getElementById('mascotEyeR');
      const mouth = document.getElementById('mascotMouth');
      const blushL = document.getElementById('mascotBlushL');
      const blushR = document.getElementById('mascotBlushR');

      if (!this.svgElement || !mouth) return;

      let quote = customQuote;
      let auraColor = 'rgba(45, 212, 191, 0.6)';
      let moodText = 'Calm & Present';

      switch (state) {
        case 'joy':
          moodText = 'Joyful & Energized';
          auraColor = 'radial-gradient(circle, #34d399 0%, rgba(52, 211, 153, 0.15) 60%, transparent 75%)';
          mouth.setAttribute('d', 'M 85 125 Q 100 148 115 125'); // Big wide smile
          if (blushL) blushL.style.opacity = '0.8';
          if (blushR) blushR.style.opacity = '0.8';
          quote = quote || 'Your momentum is radiating today. Let’s channel this flourishing focus!';
          break;

        case 'gratitude':
          moodText = 'Warm & Appreciative';
          auraColor = 'radial-gradient(circle, #2dd4bf 0%, rgba(45, 212, 191, 0.15) 60%, transparent 75%)';
          mouth.setAttribute('d', 'M 88 126 Q 100 140 112 126'); // Gentle warm smile
          if (blushL) blushL.style.opacity = '0.7';
          if (blushR) blushR.style.opacity = '0.7';
          quote = quote || 'Acknowledging support builds strong emotional resilience for the whole team.';
          break;

        case 'stress':
        case 'anxious':
          moodText = 'Holding Space for You';
          auraColor = 'radial-gradient(circle, #fbbf24 0%, rgba(251, 191, 36, 0.15) 60%, transparent 75%)';
          mouth.setAttribute('d', 'M 88 132 Q 100 126 112 132'); // Soft sympathetic line
          if (blushL) blushL.style.opacity = '0.3';
          if (blushR) blushR.style.opacity = '0.3';
          quote = quote || 'Take a deep breath with me right now. You are doing enough, and you are supported.';
          break;

        case 'fatigue':
        case 'exhausted':
          moodText = 'Gentle Rest Mode';
          auraColor = 'radial-gradient(circle, #a78bfa 0%, rgba(167, 139, 250, 0.15) 60%, transparent 75%)';
          mouth.setAttribute('d', 'M 90 128 Q 100 131 110 128'); // Soft neutral curve
          if (blushL) blushL.style.opacity = '0.4';
          if (blushR) blushR.style.opacity = '0.4';
          quote = quote || 'Exhaustion is your nervous system asking for a sacred reset. Listen to it today.';
          break;

        case 'frustrated':
          moodText = 'Grounding & Reset';
          auraColor = 'radial-gradient(circle, #f87171 0%, rgba(248, 113, 113, 0.15) 60%, transparent 75%)';
          mouth.setAttribute('d', 'M 88 133 L 112 131'); // Straight firm line
          if (blushL) blushL.style.opacity = '0.2';
          if (blushR) blushR.style.opacity = '0.2';
          quote = quote || 'Friction is natural in ambitious work. Step back for two minutes to regain perspective.';
          break;

        default: // 'idle' / 'calm'
          moodText = 'Centering & Focused';
          auraColor = 'radial-gradient(circle, #38bdf8 0%, rgba(56, 189, 248, 0.15) 60%, transparent 75%)';
          mouth.setAttribute('d', 'M 88 126 Q 100 136 112 126'); // Standard peaceful smile
          if (blushL) blushL.style.opacity = '0.5';
          if (blushR) blushR.style.opacity = '0.5';
          quote = quote || 'I am Aura, your wellness companion. How is your work and mind feeling right now?';
          break;
      }

      if (this.auraGlowElement) {
        this.auraGlowElement.style.background = auraColor;
      }
      if (this.moodTagElement) {
        this.moodTagElement.textContent = moodText;
      }
      if (this.quoteElement) {
        this.quoteElement.textContent = quote;
      }
    },

    handleMascotClick() {
      const phrases = [
        'Notice your shoulders right now. Drop them an inch away from your ears.',
        'Unclench your jaw and take one slow, smooth inhale.',
        'Whatever is on your desk can wait 60 seconds while you check in with yourself.',
        'Celebrate even the micro-wins today. They compound into greatness.'
      ];
      const pick = phrases[Math.floor(Math.random() * phrases.length)];
      this.setMood(this.state, pick);
    }
  };

  // =========================================================================
  // 3. Client-Side Affective NLP Engine
  // =========================================================================
  const AffectiveEngine = {
    analyze(text) {
      if (!text || !text.trim()) {
        return {
          compound: 0.0,
          label: 'neutral',
          dominantEmotion: 'neutral',
          emotionScores: { neutral: 0.95 },
          tokens: [],
          actionTitle: 'Gentle Reflection',
          actionSteps: ['Write a few words to explore your feelings.', 'Take one conscious breath.', 'Keep suggestions simple.']
        };
      }

      const words = text.toLowerCase().match(/\b[a-z'-]+\b/g) || [];
      let totalScore = 0;
      let matchedCount = 0;

      // Detect negations and boosters
      for (let i = 0; i < words.length; i++) {
        const word = words[i];
        let val = VADER_LEXICON[word];

        if (val !== undefined) {
          // Check preceding words for negations
          let isNegated = false;
          if (i > 0 && NEGATION_WORDS.has(words[i - 1])) isNegated = true;
          if (i > 1 && NEGATION_WORDS.has(words[i - 2])) isNegated = true;

          // Check boosters
          let boost = 0;
          if (i > 0 && BOOSTER_WORDS[words[i - 1]]) boost += BOOSTER_WORDS[words[i - 1]];

          if (val > 0) val += boost;
          else val -= boost;

          if (isNegated) {
            val = -val * 0.74; // Invert sentiment with attenuation
          }

          totalScore += val;
          matchedCount++;
        }
      }

      // Normalization to [-1.0, 1.0] VADER compound
      const alpha = 15;
      const compound = totalScore === 0 ? 0 : totalScore / Math.sqrt((totalScore * totalScore) + alpha);
      const roundedCompound = parseFloat(compound.toFixed(3));

      let label = 'neutral';
      if (roundedCompound >= 0.05) label = 'positive';
      else if (roundedCompound <= -0.05) label = 'negative';

      // Emotion scoring
      const emotionScores = {};
      let highestScore = 0;
      let dominantEmotion = 'neutral';

      for (const [key, profile] of Object.entries(EMOTION_PROFILES)) {
        let count = 0;
        for (const cue of profile.cues) {
          if (text.toLowerCase().includes(cue)) count += 1;
        }
        // Base probability calculation
        const score = Math.min(0.96, Math.max(0.04, count > 0 ? (count * 0.28) + 0.15 : 0.05));
        emotionScores[key] = parseFloat(score.toFixed(2));
        if (score > highestScore) {
          highestScore = score;
          dominantEmotion = key;
        }
      }

      // If negative but no emotion matched, default to stress or sadness
      if (label === 'negative' && dominantEmotion === 'neutral') {
        dominantEmotion = 'stress';
        emotionScores['stress'] = 0.62;
      } else if (label === 'positive' && dominantEmotion === 'neutral') {
        dominantEmotion = 'joy';
        emotionScores['joy'] = 0.68;
      }

      // Determine personalized action pathway
      let actionTitle = 'Self-Paced Re-Centering';
      let actionSteps = [
        'Acknowledge how your body is holding this moment.',
        'Sip a glass of cool mineral water to reset autonomic tone.',
        'Identify one micro-task you can finish or gently park for tomorrow.'
      ];

      if (dominantEmotion === 'stress' || dominantEmotion === 'frustration') {
        actionTitle = 'De-escalation & Nervous System Reset';
        actionSteps = [
          'Practice two rounds of the 60-Second Physiological Sigh in our Somatic tab.',
          'Write down the single most friction-heavy roadblock on a notepad to clear cognitive RAM.',
          'Schedule a 5-minute physical walk away from screens to lower cortisol.'
        ];
      } else if (dominantEmotion === 'fatigue') {
        actionTitle = 'Mitochondrial Recharge & Sensory Rest';
        actionSteps = [
          'Turn off high-contrast monitors for 5 minutes of ambient sound immersion.',
          'Nourish yourself with complex electrolyte hydration and bioavailable magnesium.',
          'Block the final 30 minutes of your workday for zero-meeting wrap-up.'
        ];
      } else if (dominantEmotion === 'joy' || dominantEmotion === 'gratitude') {
        actionTitle = 'Morale Momentum & Collaborative Flow';
        actionSteps = [
          'Send a 1-sentence gratitude note to an employee or peer who helped unblock you.',
          'Jot down the creative insight sparked today while it is still fresh in mind.',
          'Anchor this sense of accomplishment before diving into the next sprint.'
        ];
      }

      return {
        compound: roundedCompound,
        label,
        dominantEmotion,
        emotionScores,
        tokens: words.slice(0, 15),
        actionTitle,
        actionSteps
      };
    }
  };

  // =========================================================================
  // 4. Procedural Web Audio Synthesizer (Zero-Network Offline Sanctuary)
  // =========================================================================
  const AudioSanctuary = {
    ctx: null,
    currentSound: null,
    isPlaying: false,
    nodes: [],
    gainNode: null,
    volume: 0.5,

    init() {
      // Lazy init AudioContext on first user interaction
      const startCtx = () => {
        if (!this.ctx) {
          const AudioContext = window.AudioContext || window.webkitAudioContext;
          if (AudioContext) {
            this.ctx = new AudioContext();
            this.gainNode = this.ctx.createGain();
            this.gainNode.gain.setValueAtTime(this.volume, this.ctx.currentTime);
            this.gainNode.connect(this.ctx.destination);
          }
        }
      };
      document.body.addEventListener('click', startCtx, { once: true });
    },

    setVolume(vol) {
      this.volume = vol;
      if (this.gainNode && this.ctx) {
        this.gainNode.gain.setTargetAtTime(vol, this.ctx.currentTime, 0.05);
      }
    },

    stopAll() {
      this.nodes.forEach(n => {
        try { n.stop(); } catch (e) {}
        try { n.disconnect(); } catch (e) {}
      });
      this.nodes = [];
      this.isPlaying = false;
      this.currentSound = null;
      this.updateUI();
    },

    play(soundType) {
      if (this.currentSound === soundType && this.isPlaying) {
        this.stopAll();
        return;
      }
      this.stopAll();

      if (!this.ctx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioContext();
        this.gainNode = this.ctx.createGain();
        this.gainNode.gain.setValueAtTime(this.volume, this.ctx.currentTime);
        this.gainNode.connect(this.ctx.destination);
      }

      if (this.ctx.state === 'suspended') {
        this.ctx.resume();
      }

      this.currentSound = soundType;
      this.isPlaying = true;

      switch (soundType) {
        case 'rain':
          this.synthesizeRain();
          break;
        case 'waves':
          this.synthesizeOceanWaves();
          break;
        case 'bowl':
          this.synthesizeSingingBowl();
          break;
        case 'drone':
          this.synthesizeBinauralZen();
          break;
      }

      this.updateUI();
    },

    synthesizeRain() {
      const bufferSize = this.ctx.sampleRate * 2;
      const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
      const data = buffer.getChannelData(0);
      let b0 = 0, b1 = 0, b2 = 0;

      // Pink noise filter algorithm
      for (let i = 0; i < bufferSize; i++) {
        const white = Math.random() * 2 - 1;
        b0 = 0.99886 * b0 + white * 0.0555179;
        b1 = 0.99332 * b1 + white * 0.0750759;
        b2 = 0.96900 * b2 + white * 0.1538520;
        data[i] = (b0 + b1 + b2) * 0.11;
      }

      const noise = this.ctx.createBufferSource();
      noise.buffer = buffer;
      noise.loop = true;

      const filter = this.ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(1100, this.ctx.currentTime);

      noise.connect(filter);
      filter.connect(this.gainNode);
      noise.start();
      this.nodes.push(noise);
    },

    synthesizeOceanWaves() {
      const bufferSize = this.ctx.sampleRate * 4;
      const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * 0.2;
      }

      const noise = this.ctx.createBufferSource();
      noise.buffer = buffer;
      noise.loop = true;

      const filter = this.ctx.createBiquadFilter();
      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(450, this.ctx.currentTime);
      filter.Q.setValueAtTime(1.8, this.ctx.currentTime);

      // Low frequency oscillator for wave swells (in and out)
      const lfo = this.ctx.createOscillator();
      lfo.frequency.setValueAtTime(0.12, this.ctx.currentTime); // ~8 second wave cycle
      const lfoGain = this.ctx.createGain();
      lfoGain.gain.setValueAtTime(320, this.ctx.currentTime);

      lfo.connect(lfoGain);
      lfoGain.connect(filter.frequency);

      noise.connect(filter);
      filter.connect(this.gainNode);

      lfo.start();
      noise.start();
      this.nodes.push(noise, lfo);
    },

    synthesizeSingingBowl() {
      const freqs = [432, 864, 1296, 1728];
      const gains = [0.4, 0.2, 0.08, 0.04];

      freqs.forEach((f, idx) => {
        const osc = this.ctx.createOscillator();
        const g = this.ctx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(f + (Math.random() * 1.5 - 0.75), this.ctx.currentTime);

        g.gain.setValueAtTime(gains[idx], this.ctx.currentTime);
        // Periodic gentle pulse
        osc.connect(g);
        g.connect(this.gainNode);
        osc.start();
        this.nodes.push(osc);
      });
    },

    synthesizeBinauralZen() {
      // 432Hz harmonic theta binaural drone
      const oscL = this.ctx.createOscillator();
      const oscR = this.ctx.createOscillator();
      const pannerL = this.ctx.createStereoPanner ? this.ctx.createStereoPanner() : null;
      const pannerR = this.ctx.createStereoPanner ? this.ctx.createStereoPanner() : null;

      oscL.type = 'sine';
      oscR.type = 'sine';
      oscL.frequency.setValueAtTime(136.1, this.ctx.currentTime); // Om tone
      oscR.frequency.setValueAtTime(142.1, this.ctx.currentTime); // +6Hz theta wave beat

      if (pannerL && pannerR) {
        pannerL.pan.setValueAtTime(-0.8, this.ctx.currentTime);
        pannerR.pan.setValueAtTime(0.8, this.ctx.currentTime);
        oscL.connect(pannerL);
        pannerL.connect(this.gainNode);
        oscR.connect(pannerR);
        pannerR.connect(this.gainNode);
      } else {
        oscL.connect(this.gainNode);
        oscR.connect(this.gainNode);
      }

      oscL.start();
      oscR.start();
      this.nodes.push(oscL, oscR);
    },

    updateUI() {
      const quickToggle = document.getElementById('soundQuickToggle');
      if (quickToggle) {
        if (this.isPlaying) {
          quickToggle.classList.add('playing');
        } else {
          quickToggle.classList.remove('playing');
        }
      }

      document.querySelectorAll('.soundscape-tile').forEach(tile => {
        const type = tile.getAttribute('data-sound');
        if (type === this.currentSound && this.isPlaying) {
          tile.classList.add('active');
        } else {
          tile.classList.remove('active');
        }
      });
    }
  };

  // =========================================================================
  // 5. Somatic Breathwork Pacer (Interactive Loop)
  // =========================================================================
  const BreathPacer = {
    isRunning: false,
    timer: null,
    phaseIndex: 0,
    secondsLeft: 4,
    phases: [
      { name: 'Inhale Slowly', class: 'inhale', duration: 4 },
      { name: 'Gently Hold', class: 'hold', duration: 4 },
      { name: 'Exhale Smoothly', class: 'exhale', duration: 4 },
      { name: 'Rest & Pause', class: 'hold', duration: 4 }
    ],

    init() {
      const startBtn = document.getElementById('startPacerBtn');
      if (startBtn) {
        startBtn.addEventListener('click', () => this.toggle());
      }
    },

    toggle() {
      if (this.isRunning) {
        this.stop();
      } else {
        this.start();
      }
    },

    start() {
      this.isRunning = true;
      this.phaseIndex = 0;
      this.secondsLeft = this.phases[0].duration;

      const btn = document.getElementById('startPacerBtn');
      if (btn) btn.textContent = 'Pause Breathing';

      this.step();
      this.timer = setInterval(() => this.tick(), 1000);
    },

    stop() {
      this.isRunning = false;
      clearInterval(this.timer);
      const btn = document.getElementById('startPacerBtn');
      if (btn) btn.textContent = 'Start Guided Breathwork';

      const orb = document.getElementById('pacerOrb');
      const label = document.getElementById('pacerPhaseLabel');
      if (orb) {
        orb.className = 'pacer-orb';
        orb.textContent = '4s';
      }
      if (label) label.textContent = 'Ready to begin';
    },

    tick() {
      this.secondsLeft--;
      const orb = document.getElementById('pacerOrb');
      if (orb) orb.textContent = `${this.secondsLeft}s`;

      if (this.secondsLeft <= 0) {
        this.phaseIndex = (this.phaseIndex + 1) % this.phases.length;
        this.secondsLeft = this.phases[this.phaseIndex].duration;
        this.step();
      }
    },

    step() {
      const current = this.phases[this.phaseIndex];
      const orb = document.getElementById('pacerOrb');
      const label = document.getElementById('pacerPhaseLabel');

      if (orb) {
        orb.className = `pacer-orb ${current.class}`;
        orb.textContent = `${this.secondsLeft}s`;
      }
      if (label) {
        label.textContent = current.name;
      }
    }
  };

  // =========================================================================
  // 6. Team Morale & Burnout Radar Simulator Canvas Chart
  // =========================================================================
  const TeamSimulator = {
    canvas: null,
    ctx: null,
    scenario: 'balanced',

    init() {
      this.canvas = document.getElementById('teamChartCanvas');
      if (this.canvas) {
        this.ctx = this.canvas.getContext('2d');
        window.addEventListener('resize', () => this.drawChart());
        this.setupScenarioButtons();
        this.drawChart();
      }
    },

    setupScenarioButtons() {
      document.querySelectorAll('.team-scenario-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          document.querySelectorAll('.team-scenario-btn').forEach(b => b.classList.remove('active'));
          e.currentTarget.classList.add('active');
          this.scenario = e.currentTarget.getAttribute('data-scenario');
          this.updateMetrics();
          this.drawChart();
        });
      });
    },

    updateMetrics() {
      const riskEl = document.getElementById('kpiBurnoutRisk');
      const resEl = document.getElementById('kpiResilienceIndex');
      const compEl = document.getElementById('kpiSentimentComp');
      const distEl = document.getElementById('kpiDistressRatio');

      if (this.scenario === 'thriving') {
        if (riskEl) riskEl.textContent = '14/100';
        if (resEl) resEl.textContent = '91/100';
        if (compEl) compEl.textContent = '+0.48';
        if (distEl) distEl.textContent = '6% / 94%';
      } else if (this.scenario === 'burnout') {
        if (riskEl) riskEl.textContent = '68/100';
        if (resEl) resEl.textContent = '38/100';
        if (compEl) compEl.textContent = '-0.36';
        if (distEl) distEl.textContent = '54% / 46%';
      } else {
        // Balanced
        if (riskEl) riskEl.textContent = '32/100';
        if (resEl) resEl.textContent = '74/100';
        if (compEl) compEl.textContent = '+0.18';
        if (distEl) distEl.textContent = '18% / 82%';
      }
    },

    drawChart() {
      if (!this.canvas || !this.ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = this.canvas.getBoundingClientRect();

      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = rect.height;
      this.ctx.clearRect(0, 0, w, h);

      // 14 days of synthetic data based on scenario
      let points = [];
      if (this.scenario === 'thriving') {
        points = [0.25, 0.32, 0.40, 0.38, 0.50, 0.45, 0.48, 0.52, 0.58, 0.50, 0.62, 0.55, 0.58, 0.64];
      } else if (this.scenario === 'burnout') {
        points = [0.10, -0.05, -0.15, -0.22, -0.18, -0.32, -0.28, -0.35, -0.42, -0.38, -0.48, -0.42, -0.52, -0.58];
      } else {
        points = [0.08, 0.15, 0.22, 0.12, 0.18, 0.09, 0.14, 0.25, 0.20, 0.18, 0.22, 0.16, 0.20, 0.24];
      }

      // Draw zero line
      const zeroY = h * 0.5;
      this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
      this.ctx.lineWidth = 1;
      this.ctx.setLineDash([4, 4]);
      this.ctx.beginPath();
      this.ctx.moveTo(40, zeroY);
      this.ctx.lineTo(w - 20, zeroY);
      this.ctx.stroke();
      this.ctx.setLineDash([]);

      // Baseline labels
      this.ctx.fillStyle = '#94a3b8';
      this.ctx.font = '11px JetBrains Mono, monospace';
      this.ctx.fillText('0.0 (Neutral Baseline)', 45, zeroY - 6);

      // Danger boundary line (-0.3)
      const dangerY = zeroY + (h * 0.3);
      this.ctx.strokeStyle = 'rgba(244, 63, 94, 0.25)';
      this.ctx.beginPath();
      this.ctx.moveTo(40, dangerY);
      this.ctx.lineTo(w - 20, dangerY);
      this.ctx.stroke();
      this.ctx.fillStyle = '#f43f5e';
      this.ctx.fillText('-0.30 (Strain Boundary)', 45, dangerY - 4);

      // Plot curve
      const stepX = (w - 70) / (points.length - 1);
      const coords = points.map((p, idx) => {
        const x = 50 + (idx * stepX);
        const y = zeroY - (p * (h * 0.42));
        return { x, y };
      });

      // Gradient Fill
      const grad = this.ctx.createLinearGradient(0, 0, 0, h);
      if (this.scenario === 'burnout') {
        grad.addColorStop(0, 'rgba(244, 63, 94, 0.35)');
        grad.addColorStop(1, 'rgba(244, 63, 94, 0.02)');
      } else {
        grad.addColorStop(0, 'rgba(15, 118, 110, 0.35)');
        grad.addColorStop(1, 'rgba(15, 118, 110, 0.02)');
      }

      this.ctx.beginPath();
      this.ctx.moveTo(coords[0].x, zeroY);
      this.ctx.lineTo(coords[0].x, coords[0].y);
      for (let i = 1; i < coords.length; i++) {
        const midX = (coords[i - 1].x + coords[i].x) / 2;
        this.ctx.quadraticCurveTo(coords[i - 1].x, coords[i - 1].y, midX, (coords[i - 1].y + coords[i].y) / 2);
      }
      this.ctx.lineTo(coords[coords.length - 1].x, coords[coords.length - 1].y);
      this.ctx.lineTo(coords[coords.length - 1].x, zeroY);
      this.ctx.closePath();
      this.ctx.fillStyle = grad;
      this.ctx.fill();

      // Stroke Line
      this.ctx.beginPath();
      this.ctx.moveTo(coords[0].x, coords[0].y);
      for (let i = 1; i < coords.length; i++) {
        this.ctx.lineTo(coords[i].x, coords[i].y);
      }
      this.ctx.strokeStyle = this.scenario === 'burnout' ? '#f43f5e' : '#0f766e';
      this.ctx.lineWidth = 3;
      this.ctx.stroke();

      // Draw points
      coords.forEach(pt => {
        this.ctx.beginPath();
        this.ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();
        this.ctx.strokeStyle = this.scenario === 'burnout' ? '#f43f5e' : '#0f766e';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
      });
    }
  };

  // =========================================================================
  // 7. Confidential Reflection Journal Storage & Export Manager
  // =========================================================================
  const JournalManager = {
    storageKey: 'moodmentor_confidential_journal',

    getEntries() {
      try {
        const raw = localStorage.getItem(this.storageKey);
        return raw ? JSON.parse(raw) : [];
      } catch (e) {
        return [];
      }
    },

    saveEntry(text, result) {
      const entries = this.getEntries();
      const newEntry = {
        id: Date.now(),
        date: new Date().toISOString(),
        text,
        compound: result.compound,
        sentiment: result.label,
        emotion: result.dominantEmotion
      };
      entries.unshift(newEntry);
      try {
        localStorage.setItem(this.storageKey, JSON.stringify(entries));
      } catch (e) {}
      this.renderEntries();
    },

    deleteEntry(id) {
      let entries = this.getEntries();
      entries = entries.filter(e => e.id !== id);
      localStorage.setItem(this.storageKey, JSON.stringify(entries));
      this.renderEntries();
    },

    renderEntries() {
      const container = document.getElementById('journalEntriesList');
      if (!container) return;

      const entries = this.getEntries();
      if (entries.length === 0) {
        container.innerHTML = `
          <div style="text-align: center; padding: 32px; color: var(--text-muted);">
            <p>No journal reflections saved yet. Analyze your thoughts above and tap "Save to Confidential Journal" to begin your log.</p>
          </div>
        `;
        return;
      }

      container.innerHTML = entries.map(item => {
        const d = new Date(item.date);
        const dateStr = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        const badgeClass = item.sentiment === 'positive' ? 'polarity-positive' : (item.sentiment === 'negative' ? 'polarity-negative' : 'polarity-neutral');

        return `
          <div class="journal-entry-card" id="entry-${item.id}">
            <div>
              <div class="entry-text-excerpt">"${this.escapeHtml(item.text.slice(0, 140))}${item.text.length > 140 ? '...' : ''}"</div>
              <div class="entry-meta-row">
                <span>${dateStr}</span>
                <span>Emotion: <strong>${item.emotion.toUpperCase()}</strong></span>
                <span>Compound: <strong>${item.compound > 0 ? '+' : ''}${item.compound.toFixed(2)}</strong></span>
              </div>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
              <span class="polarity-pill ${badgeClass}">${item.sentiment}</span>
              <button class="btn-icon" onclick="window.MoodMentorApp.deleteJournalItem(${item.id})" title="Delete record">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
              </button>
            </div>
          </div>
        `;
      }).join('');
    },

    exportMarkdown() {
      const entries = this.getEntries();
      if (!entries.length) {
        alert('No saved entries to export yet.');
        return;
      }
      let md = '# MoodMentor Employee Reflection Journal\n\n*Strictly Confidential Workplace Wellness Log*\n\n---\n\n';
      entries.forEach(item => {
        md += `### ${new Date(item.date).toLocaleString()}\n`;
        md += `- **Polarity:** ${item.sentiment.toUpperCase()} (Compound: ${item.compound})\n`;
        md += `- **Dominant Emotion:** ${item.emotion}\n`;
        md += `\n> ${item.text}\n\n---\n\n`;
      });

      this.downloadFile(md, 'moodmentor-journal-export.md', 'text/markdown');
    },

    exportCSV() {
      const entries = this.getEntries();
      if (!entries.length) {
        alert('No saved entries to export yet.');
        return;
      }
      let csv = 'Timestamp,Sentiment,CompoundScore,DominantEmotion,OriginalText\n';
      entries.forEach(e => {
        const clean = `"${e.text.replace(/"/g, '""')}"`;
        csv += `${e.date},${e.sentiment},${e.compound},${e.emotion},${clean}\n`;
      });

      this.downloadFile(csv, 'moodmentor-journal-export.csv', 'text/csv');
    },

    downloadFile(content, filename, type) {
      const blob = new Blob([content], { type });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },

    escapeHtml(str) {
      return str.replace(/[&<>'"]/g, tag => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
      }[tag] || tag));
    }
  };

  // =========================================================================
  // 8. Main Application Controller & Event Wiring
  // =========================================================================
  const App = {
    lastAnalysisResult: null,

    init() {
      MascotController.init();
      AudioSanctuary.init();
      BreathPacer.init();
      TeamSimulator.init();
      JournalManager.renderEntries();

      this.bindEvents();
    },

    bindEvents() {
      // Theme Toggle
      const themeBtn = document.getElementById('themeToggleBtn');
      if (themeBtn) {
        themeBtn.addEventListener('click', () => {
          document.body.classList.toggle('theme-dark');
          const isDark = document.body.classList.contains('theme-dark');
          localStorage.setItem('moodmentor_theme', isDark ? 'dark' : 'light');
        });

        // Restore theme
        if (localStorage.getItem('moodmentor_theme') === 'dark') {
          document.body.classList.add('theme-dark');
        }
      }

      // Quick Audio Bar Toggle
      const quickSound = document.getElementById('soundQuickToggle');
      if (quickSound) {
        quickSound.addEventListener('click', () => {
          if (AudioSanctuary.isPlaying) {
            AudioSanctuary.stopAll();
          } else {
            AudioSanctuary.play('rain');
          }
        });
      }

      // Soundscape Tiles
      document.querySelectorAll('.soundscape-tile').forEach(tile => {
        tile.addEventListener('click', (e) => {
          const sound = e.currentTarget.getAttribute('data-sound');
          AudioSanctuary.play(sound);
        });
      });

      // Volume Slider
      const volSlider = document.getElementById('soundVolumeSlider');
      if (volSlider) {
        volSlider.addEventListener('input', (e) => {
          AudioSanctuary.setVolume(parseFloat(e.target.value));
        });
      }

      // Reflection Textarea & Character Count
      const textarea = document.getElementById('analyzerTextarea');
      const charCount = document.getElementById('analyzerCharCount');
      if (textarea && charCount) {
        textarea.addEventListener('input', () => {
          charCount.textContent = `${textarea.value.length} characters`;
        });
      }

      // Preset Scenarios
      const presetSelect = document.getElementById('presetSelect');
      if (presetSelect && textarea) {
        presetSelect.addEventListener('change', () => {
          if (presetSelect.value) {
            textarea.value = presetSelect.value;
            textarea.dispatchEvent(new Event('input'));
            this.runAnalysis();
          }
        });
      }

      // Analyze Button
      const analyzeBtn = document.getElementById('runAnalyzeBtn');
      if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => this.runAnalysis());
      }

      // Save to Journal Button
      const saveJournalBtn = document.getElementById('saveToJournalBtn');
      if (saveJournalBtn && textarea) {
        saveJournalBtn.addEventListener('click', () => {
          const text = textarea.value.trim();
          if (!text) {
            alert('Please compose or analyze a reflection note first.');
            return;
          }
          const result = this.lastAnalysisResult || AffectiveEngine.analyze(text);
          JournalManager.saveEntry(text, result);
          alert('Saved reflection to your private local journal.');
        });
      }

      // Journal Export Buttons
      const exportMdBtn = document.getElementById('exportMdBtn');
      const exportCsvBtn = document.getElementById('exportCsvBtn');
      if (exportMdBtn) exportMdBtn.addEventListener('click', () => JournalManager.exportMarkdown());
      if (exportCsvBtn) exportCsvBtn.addEventListener('click', () => JournalManager.exportCSV());

      // Nutrition Selectors
      document.querySelectorAll('.seg-diet-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          document.querySelectorAll('.seg-diet-btn').forEach(b => b.classList.remove('active'));
          e.currentTarget.classList.add('active');
          this.updateNutritionView();
        });
      });

      const moodSelect = document.getElementById('nutritionMoodSelect');
      if (moodSelect) {
        moodSelect.addEventListener('change', () => this.updateNutritionView());
      }

      // Crisis Modal
      const openCrisisBtn = document.getElementById('crisisModalOpenBtn');
      const closeCrisisBtn = document.getElementById('crisisModalCloseBtn');
      const modal = document.getElementById('crisisModalBackdrop');

      if (openCrisisBtn && modal) {
        openCrisisBtn.addEventListener('click', () => modal.classList.add('open'));
      }
      if (closeCrisisBtn && modal) {
        closeCrisisBtn.addEventListener('click', () => modal.classList.remove('open'));
      }
      if (modal) {
        modal.addEventListener('click', (e) => {
          if (e.target === modal) modal.classList.remove('open');
        });
      }
    },

    runAnalysis() {
      const textarea = document.getElementById('analyzerTextarea');
      if (!textarea) return;

      const text = textarea.value.trim();
      const resultsContainer = document.getElementById('analysisResultsGrid');

      const result = AffectiveEngine.analyze(text);
      this.lastAnalysisResult = result;

      // Update Mascot
      MascotController.setMood(result.dominantEmotion);

      // Render Results
      if (resultsContainer) {
        resultsContainer.style.display = 'grid';

        // Gauge values
        const gaugeVal = document.getElementById('gaugeCompoundVal');
        const gaugeFill = document.getElementById('gaugeNeedleFill');
        const polarityPill = document.getElementById('gaugePolarityPill');

        if (gaugeVal) gaugeVal.textContent = (result.compound > 0 ? '+' : '') + result.compound.toFixed(2);
        if (gaugeFill) {
          // Normalize [-1, 1] to [0%, 100%]
          const pct = Math.max(4, Math.min(96, ((result.compound + 1) / 2) * 100));
          gaugeFill.style.width = `${pct}%`;
          gaugeFill.style.background = result.compound > 0.05 ? '#10b981' : (result.compound < -0.05 ? '#f43f5e' : '#94a3b8');
        }
        if (polarityPill) {
          polarityPill.className = `polarity-pill polarity-${result.label}`;
          polarityPill.textContent = result.label.toUpperCase();
        }

        // Emotion Progress Bars
        const emotionBarsContainer = document.getElementById('emotionSignalsList');
        if (emotionBarsContainer) {
          const sorted = Object.entries(result.emotionScores)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);

          emotionBarsContainer.innerHTML = sorted.map(([k, score]) => {
            const prof = EMOTION_PROFILES[k] || { name: k, color: '#0d9488' };
            const pct = Math.round(score * 100);
            return `
              <div class="emotion-bar-item">
                <span class="emotion-name">${prof.name.split(' ')[0]}</span>
                <div class="emotion-progress-track">
                  <div class="emotion-progress-fill" style="width: ${pct}%; background: ${prof.color};"></div>
                </div>
                <span class="emotion-pct">${pct}%</span>
              </div>
            `;
          }).join('');
        }

        // Action Pathways
        const pathwayTitle = document.getElementById('actionPathwayTitle');
        const pathwaySteps = document.getElementById('actionPathwaySteps');
        if (pathwayTitle) pathwayTitle.textContent = result.actionTitle;
        if (pathwaySteps) {
          pathwaySteps.innerHTML = result.actionSteps.map((s, idx) => `
            <li class="action-step-item">
              <span class="action-step-num">STEP 0${idx + 1}</span>
              <p>${s}</p>
            </li>
          `).join('');
        }
      }
    },

    updateNutritionView() {
      const moodSelect = document.getElementById('nutritionMoodSelect');
      const activeDietBtn = document.querySelector('.seg-diet-btn.active');
      if (!moodSelect || !activeDietBtn) return;

      const moodKey = moodSelect.value || 'stress';
      const dietKey = activeDietBtn.getAttribute('data-diet') || 'any';
      const plan = NUTRITION_DATABASE[moodKey] || NUTRITION_DATABASE.stress;

      const titleEl = document.getElementById('nutritionPlanTitle');
      const targetEl = document.getElementById('nutritionPlanTarget');
      const meals = plan.meals[dietKey] || plan.meals.any;

      if (titleEl) titleEl.textContent = plan.title;
      if (targetEl) targetEl.textContent = plan.target;

      const bEl = document.getElementById('mealBreakfastDesc');
      const lEl = document.getElementById('mealLunchDesc');
      const sEl = document.getElementById('mealSnackDesc');
      const dEl = document.getElementById('mealDinnerDesc');

      if (bEl) bEl.textContent = meals.breakfast;
      if (lEl) lEl.textContent = meals.lunch;
      if (sEl) sEl.textContent = meals.snack;
      if (dEl) dEl.textContent = meals.dinner;

      const superList = document.getElementById('superfoodsList');
      if (superList) {
        superList.innerHTML = plan.superfoods.map(item => `
          <div class="superfood-item">
            <strong>${item.food}:</strong> <span>${item.benefit}</span>
          </div>
        `).join('');
      }

      const avoidList = document.getElementById('moderateList');
      if (avoidList) {
        avoidList.innerHTML = plan.avoid.map(av => `
          <div class="superfood-item">
            <span style="color:#f43f5e;font-weight:700;">Moderate:</span> <span>${av}</span>
          </div>
        `).join('');
      }
    }
  };

  // Expose global methods
  window.MoodMentorApp = {
    deleteJournalItem(id) {
      JournalManager.deleteEntry(id);
    }
  };

  // Initialize on DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => App.init());
  } else {
    App.init();
  }
})();
