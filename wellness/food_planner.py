"""Nutritional Psychiatry & Mood-Food Planning Engine.

Evidence-based nutritional recommendations tailored to emotional states,
neurochemical pathways (gut-brain axis, cortisol modulation, serotonin/GABA synthesis),
and dietary preferences for employee well-being.
"""

from __future__ import annotations

from typing import Any


def _filter_meal_for_diet(meal_desc: str, dietary_pref: str) -> str:
    """Adjust meal description for vegetarian or vegan preferences if needed."""
    pref = dietary_pref.lower().strip()
    if pref in ("omnivore", "any"):
        return meal_desc

    text = meal_desc
    if pref == "vegan":
        text = text.replace("Greek yogurt (or coconut yogurt)", "Coconut yogurt")
        text = text.replace("Greek yogurt", "Coconut probiotic yogurt")
        text = text.replace("Poached eggs", "Scrambled spiced organic tofu")
        text = text.replace("eggs", "silken tofu or chia seed pudding")
        text = text.replace("egg", "tofu slice")
        text = text.replace("Pan-seared wild salmon (or crispy spiced organic tofu)", "Crispy spiced organic tofu")
        text = text.replace("wild salmon", "marinated tempeh with seaweed flakes (algal DHA)")
        text = text.replace("salmon", "hemp-crusted tempeh")
        text = text.replace("Baked lemon-herb chicken breast (or grilled lentil patties)", "Grilled herb-lentil patties")
        text = text.replace("chicken breast", "crispy chickpea cakes")
        text = text.replace("chicken", "herb-roasted tempeh")
        text = text.replace("cod", "pan-seared marinated king oyster mushrooms")
        text = text.replace("bone broth", "mineral-rich shiitake and kelp broth")
    elif pref == "vegetarian":
        text = text.replace("Pan-seared wild salmon (or crispy spiced organic tofu)", "Crispy spiced organic tofu or paneer")
        text = text.replace("wild salmon", "grilled organic paneer or spiced tempeh")
        text = text.replace("salmon", "baked tofu or paneer")
        text = text.replace("Baked lemon-herb chicken breast (or grilled lentil patties)", "Grilled lentil patties or halloumi")
        text = text.replace("chicken breast", "grilled lentil-quinoa cutlets")
        text = text.replace("chicken", "seasoned plant protein")
        text = text.replace("cod", "pan-seared tempeh or paneer")
        text = text.replace("bone broth", "warm vegetable and shiitake broth")
    return text


def get_mood_food_plan(emotion: str, dietary_pref: str = "any") -> dict[str, Any]:
    """Return tailored nutritional psychiatry recommendations for the detected emotion.
    
    Parameters
    ----------
    emotion : str
        Detected emotion label (e.g. fear, anger, sadness, joy, fatigue, neutral).
    dietary_pref : str
        Dietary preference filter: 'any', 'omnivore', 'vegetarian', or 'vegan'.
    """
    emo = emotion.lower().strip()
    pref = dietary_pref.lower().strip()
    if pref not in ("any", "omnivore", "vegetarian", "vegan"):
        pref = "any"

    # Map fine-grained GoEmotions into nutritional psychiatry profiles
    if emo in ("fear", "nervousness", "anxiety"):
        category = "stress_anxiety"
        title = "CortisolShield Plate (Acute Stress & Nervousness)"
        objective = "Down-regulate adrenal cortisol, stimulate inhibitory GABA synthesis, and restore autonomic calm."
        key_nutrients = ["Magnesium Glycinate / Citrate", "L-Theanine", "Omega-3 (EPA/DHA)", "Vitamin B6", "Zinc"]
        superfoods = [
            {"food": "Pumpkin Seeds", "benefit": "Highest natural dietary magnesium; relaxes tense vascular & skeletal muscle tissue."},
            {"food": "Dark Chocolate (85%+ Cacao)", "benefit": "Flavanols reduce salivary cortisol and lower acute cardiovascular stress reactivity."},
            {"food": "Matcha or Organic Green Tea", "benefit": "High L-Theanine crosses the blood-brain barrier to induce calming alpha brain waves."},
            {"food": "Wild Salmon / Chia Seeds", "benefit": "Rich in neuroprotective Omega-3 fatty acids that combat stress-induced neuroinflammation."},
            {"food": "Chamomile & Holy Basil (Tulsi) Tea", "benefit": "Contains apigenin, an antioxidant that binds calming GABA-A receptors."},
        ]
        meals = {
            "breakfast": "Warm rolled oats cooked with almond milk, topped with crushed walnuts, chia seeds, fresh blueberries, and a pinch of cinnamon.",
            "lunch": "Quinoa vitality bowl layered with fresh baby spinach, roasted sweet potatoes, pumpkin seeds, and creamy lemon-tahini dressing.",
            "snack": "Two squares of 85% dark chocolate paired with raw almonds and a warm cup of chamomile-tulsi tea.",
            "dinner": "Pan-seared wild salmon (or crispy spiced organic tofu) served over steamed broccoli florets and turmeric brown rice.",
        }
        avoid = [
            "Excessive caffeine (>200 mg), which elevates baseline catecholamines and triggers jitteriness",
            "High-fructose corn syrup & sugary energy sodas that spike and crash blood glucose",
            "Deep-fried fast foods high in trans-fats that increase neuroinflammatory markers",
        ]
        hydration_tip = "Sip room-temperature water infused with fresh lemon and a pinch of Celtic sea salt for cellular electrolyte balance."
        scientific_basis = "Supported by Harvard Nutritional Psychiatry and the SMILES trial: magnesium and omega-3 polyunsaturated fats down-regulate hypothalamic-pituitary-adrenal (HPA) axis overdrive."

    elif emo in ("anger", "annoyance", "disapproval", "disgust"):
        category = "irritability_anger"
        title = "GlycoCalm Nourish Bowl (Irritability & Frustration)"
        objective = "Stabilize glycemic fluctuations, attenuate systemic vascular tension, and cool physiological irritability."
        key_nutrients = ["Complex Slow-Release Carbohydrates", "Potassium", "Polyphenols", "Vitamin C", "L-Tryptophan"]
        superfoods = [
            {"food": "Coconut Water", "benefit": "Abundant bioavailable potassium and electrolytes that buffer vascular adrenaline constriction."},
            {"food": "Rolled Oats & Sweet Potatoes", "benefit": "Low-glycemic complex carbs prevent hypoglycemia-induced irritability and anger spikes ('hangry' state)."},
            {"food": "Crisp Apples & Blueberries", "benefit": "Rich in quercetin and Vitamin C to blunt adrenal hyper-secretion under interpersonal tension."},
            {"food": "Peppermint & Hibiscus Tea", "benefit": "Natural smooth-muscle relaxant that relieves gastrointestinal and jaw clenching."},
            {"food": "Fermented Greek Yogurt / Kefir", "benefit": "Gut microbiome nourishment to support peripheral serotonin and GABA production."},
        ]
        meals = {
            "breakfast": "Greek yogurt (or coconut yogurt) with ground flaxseed, sliced kiwi, fresh blueberries, and raw pumpkin seeds.",
            "lunch": "Mediterranean chickpea salad with crisp Persian cucumbers, cherry tomatoes, kalamata olives, and cold-pressed extra virgin olive oil.",
            "snack": "Crisp organic apple slices paired with natural unsalted almond butter or chilled coconut water.",
            "dinner": "Baked lemon-herb chicken breast (or grilled lentil patties) alongside roasted rosemary asparagus and mashed sweet potatoes.",
        }
        avoid = [
            "Hard alcohol or cocktails, which disrupt prefrontal cortex emotional impulse inhibition",
            "Energy drinks with synthetic taurine and guarana that prolong irritability",
            "Ultra-processed salty chips that trigger dehydration and increase vascular resistance",
        ]
        hydration_tip = "Drink cool water infused with fresh cucumber slices and fresh crushed mint leaves."
        scientific_basis = "Clinical research in Psychoneuroendocrinology proves rapid glycemic drops trigger epinephrine and cortisol surges, directly degrading emotional tolerance."

    elif emo in ("sadness", "grief", "disappointment", "remorse"):
        category = "low_mood_sadness"
        title = "NeuroGenesis Serotonin Bowl (Low Mood & Melancholy)"
        objective = "Provide amino acid precursors (tryptophan, tyrosine) for serotonin/dopamine synthesis and support the gut-brain axis."
        key_nutrients = ["L-Tryptophan", "L-Tyrosine", "Folate (Vitamin B9)", "Vitamin D3", "Probiotics & Prebiotics"]
        superfoods = [
            {"food": "Pasture-Raised Eggs or Edamame", "benefit": "Rich in tryptophan, choline, and bioavailable protein essential for neurotransmitter synthesis."},
            {"food": "Ripe Bananas", "benefit": "Supplies Vitamin B6 (pyridoxine), a vital enzymatic cofactor required to convert tryptophan into serotonin."},
            {"food": "Fermented Kimchi, Kefir, or Miso", "benefit": "90% of serotonin is synthesized in the digestive tract; probiotics directly nourish the enteric nervous system."},
            {"food": "Dark Leafy Greens (Spinach/Kale)", "benefit": "Abundant in natural folate (B9); clinical studies link low folate with persistent depressive moods."},
            {"food": "Golden Turmeric Latte with Black Pepper", "benefit": "Curcumin crosses the blood-brain barrier, providing potent neuro-anti-inflammatory protection."},
        ]
        meals = {
            "breakfast": "Poached eggs on sprouted whole-grain sourdough toast with gently sautéed baby spinach and sliced avocado.",
            "lunch": "Warm golden lentil and turmeric dal soup accompanied by a side of fermented kraut and roasted sunflower seeds.",
            "snack": "Sliced ripe banana with natural peanut butter and a cup of roasted dandelion or rooibos tea.",
            "dinner": "Comforting slow-cooked vegetable and white bean stew with a drizzle of extra virgin olive oil and brown basmati rice.",
        }
        avoid = [
            "Refined sugar 'comfort foods' (pastries, candy), which trigger a temporary dopamine spike followed by a severe neurochemical drop within 45 minutes",
            "Skipping meals or fasting during emotional low points, which compounds mental lethargy",
            "Processed industrial seed oils with high omega-6 to omega-3 ratios",
        ]
        hydration_tip = "Sip warm water, herbal ginger tea, or comforting warm mineral broth to stimulate vagal nerve warmth."
        scientific_basis = "Lancet Psychiatry Nutritional Guidelines highlight that gut-derived short-chain fatty acids (SCFAs) signal the central nervous system via the vagus nerve to elevate brain-derived neurotrophic factor (BDNF)."

    elif emo in ("joy", "gratitude", "admiration", "amusement", "love", "optimism", "pride", "excitement"):
        category = "positive_flourishing"
        title = "SynapseFlow Mezze Plate (Flourishing & Morale)"
        objective = "Sustain high cognitive vitality, enhance mitochondrial ATP generation, and reinforce neural longevity."
        key_nutrients = ["Polyphenols & Anthocyanins", "Monounsaturated Fatty Acids", "Coenzyme Q10", "Clean Protein", "Choline"]
        superfoods = [
            {"food": "Fresh Avocados", "benefit": "Oleic acid and healthy fats enhance neuronal membrane fluidity and steady cerebral blood flow."},
            {"food": "Blackberries & Pomegranate Seeds", "benefit": "Anthocyanin flavonoids improve synaptic plasticity, executive function, and working memory."},
            {"food": "Raw Walnuts", "benefit": "Rich in plant-based ALA omega-3 and Vitamin E, which shield cerebral neurons from oxidative strain."},
            {"food": "High-Grade Green Tea (Sencha/Gyokuro)", "benefit": "Gentle balanced caffeine combined with EGCG antioxidants sustains effortless cognitive flow state."},
            {"food": "Sprouted Lentils & Seeds", "benefit": "Clean bioavailable plant proteins that fuel sustained daily neurotransmitter balance."},
        ]
        meals = {
            "breakfast": "Vibrant antioxidant smoothie bowl with wild berries, baby spinach, unsweetened almond milk, chia seeds, and walnut halves.",
            "lunch": "Rainbow Mediterranean grain bowl with cooked farro or quinoa, roasted golden beets, wild arugula, steamed edamame, and tahini drizzle.",
            "snack": "Handful of raw walnuts and dried figs enjoyed with a cup of fresh Japanese green tea.",
            "dinner": "Pan-seared wild salmon (or crispy spiced organic tofu) over roasted kabocha squash, sauteed swiss chard, and a squeeze of fresh lemon.",
        }
        avoid = [
            "Heavy deep-fried fast food combos that induce post-prandial somnolence (food coma) and blunt creative focus",
            "Excessive sugary treats that disrupt an already well-regulated neurochemical balance",
        ]
        hydration_tip = "Sparkling spring water garnished with crushed fresh rosemary and a slice of grapefruit or lime."
        scientific_basis = "Adherence to polyphenol-dense Mediterranean dietary patterns is clinically documented to protect sustained positive affect and executive resilience."

    elif emo in ("fatigue", "burnout"):
        category = "fatigue_burnout"
        title = "MitoChondria Fuel Plate (Burnout Recovery & ATP)"
        objective = "Restore depleted cellular ATP energy, replenish micronutrient stores, and ease chronic allostatic load."
        key_nutrients = ["Magnesium Malate", "Iron & Vitamin C", "B-Complex (B1, B5, B12)", "CoQ10", "Adaptogens"]
        superfoods = [
            {"food": "Overnight Chia & Hemp Seed Oats", "benefit": "Sustained cellular glycogen replenishment without triggering reactive hypoglycemia."},
            {"food": "Raw Brazil Nuts", "benefit": "Just 2 nuts supply 100% RDA of selenium, crucial for thyroid hormone conversion and metabolic energy."},
            {"food": "Steamed Beets & Beet Greens", "benefit": "Natural dietary nitrates dilate micro-vasculature, optimizing oxygen delivery to the cerebral cortex."},
            {"food": "Ashwagandha / Rhodiola Root Tea", "benefit": "Clinically proven adaptogenic herbs that normalize chronic cortisol dysregulation."},
            {"food": "Spirulina / Chlorella", "benefit": "Dense in bioavailable chlorophyll, iron, and trace minerals to overcome cellular fatigue."},
        ]
        meals = {
            "breakfast": "Overnight oats made with chia seeds, hemp hearts, oat milk, sliced strawberries, and raw pumpkin seeds.",
            "lunch": "Warm quinoa and lentil bowl topped with roasted butternut squash, steamed curly kale, and pumpkin seed pesto.",
            "snack": "Cold-pressed beet, apple, and ginger juice paired with 2 Brazil nuts and raw pumpkin seeds.",
            "dinner": "Slow-cooked vegetable and white bean stew with a drizzle of extra virgin olive oil, garlic, and fresh parsley.",
        }
        avoid = [
            "Repeated high-dose coffee/energy drink cycling, which further taxes exhausted adrenal receptors",
            "Skipping protein at breakfast, which guarantees an afternoon energy crash",
            "Artificial sweeteners which disrupt metabolic signaling and microbiota diversity",
        ]
        hydration_tip = "Pure coconut water blended with filtered water and a pinch of pink Himalayan rock salt for adrenal hydration."
        scientific_basis = "Chronic workplace burnout depletes intracellular magnesium and B-vitamin cofactors, starving mitochondrial electron transport chains."

    else:  # Neutral, unclassified, confusion, curious, etc.
        category = "balanced_neutral"
        title = "Equilibrium Baseline Plate (Balanced Homeostasis & Focus)"
        objective = "Maintain stable glycemic baseline, steady microvascular circulation, and neurochemical equilibrium."
        key_nutrients = ["Balanced Complete Protein", "Dietary Prebiotic Fiber", "Omega-3", "Trace Minerals"]
        superfoods = [
            {"food": "Mixed Raw Nuts (Almonds, Cashews, Walnuts)", "benefit": "Balanced blend of monounsaturated fats, protein, and minerals for steady cognition."},
            {"food": "Crisp Apples & Pears", "benefit": "Rich in pectin prebiotic fiber to feed beneficial Bifidobacteria and Lactobacilli in the gut."},
            {"food": "Extra Virgin Olive Oil", "benefit": "Rich in oleocanthal, a natural anti-inflammatory compound that protects cerebrovascular health."},
            {"food": "Fresh Herbal Mint or Rooibos Tea", "benefit": "Caffeine-free refreshing hydration that sustains mindful alertness without jitteriness."},
        ]
        meals = {
            "breakfast": "Whole-grain sprouted toast with mashed avocado, hemp hearts, cherry tomatoes, and a soft-boiled egg (or marinated tofu).",
            "lunch": "Garden vitality salad with mixed baby greens, steamed edamame, shredded carrots, cucumber, and apple-cider vinaigrette.",
            "snack": "Crisp green apple wedges served with raw creamy almond butter.",
            "dinner": "Herbed grilled chicken breast (or pan-seared tempeh) with roasted zucchini, bell peppers, and fluffy quinoa.",
        }
        avoid = [
            "Mindless grazing or desk snacking while distracted by emails",
            "Ultra-processed packaged snacks loaded with emulsifiers and artificial preservatives",
        ]
        hydration_tip = "Fresh water with sliced cucumber, fresh lemon, and a sprig of fresh garden mint."
        scientific_basis = "Nutritional psychiatry emphasizes that dietary stability and balanced macronutrient distribution prevent acute affective volatility."

    # Apply dietary preference filter to meal recommendations
    meals_filtered = {
        key: _filter_meal_for_diet(value, pref) for key, value in meals.items()
    }

    # Generate shopping list from superfoods and meals
    shopping_items = [s["food"].split("(")[0].strip() for s in superfoods]
    shopping_items.extend(["Extra Virgin Olive Oil", "Fresh Lemons", "Herbal Tea"])

    return {
        "emotion_detected": emo,
        "dietary_preference": pref,
        "mood_category": category,
        "title": title,
        "neurochemical_objective": objective,
        "key_nutrients": key_nutrients,
        "superfoods": superfoods,
        "curated_meals": meals_filtered,
        "foods_to_avoid": avoid,
        "hydration_tip": hydration_tip,
        "scientific_basis": scientific_basis,
        "shopping_essentials": sorted(set(shopping_items)),
    }


def get_supported_mood_categories() -> list[dict[str, str]]:
    """Return all supported emotional profiles for nutritional planning."""
    return [
        {"id": "fear", "label": "Stress & Nervousness (Cortisol Balance)"},
        {"id": "anger", "label": "Irritability & Frustration (Blood Sugar Stability)"},
        {"id": "sadness", "label": "Low Mood & Melancholy (Serotonin & Dopamine Support)"},
        {"id": "joy", "label": "Joy & Flourishing (Cognitive Vitality & Endurance)"},
        {"id": "fatigue", "label": "Burnout & Exhaustion (Mitochondrial ATP Restoration)"},
        {"id": "neutral", "label": "Balanced Focus (Equilibrium & Clarity)"},
    ]

