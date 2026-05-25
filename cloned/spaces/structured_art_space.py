from __future__ import annotations
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Sequence, Optional

from .base import Candidate, SearchSpace

art_data = {
    "Style": [
        "surrealist dreamscape",
        "high renaissance realism",
        "art nouveau organic",
        "cyberpunk noir",
        "solarpunk utopia",
        "vaporwave aesthetic",
        "Giger biomechanical",
        "Bauhaus geometric",
        "ukiyo-e woodblock",
        "baroque theatrical",
        "afrofuturism",
        "wabi-sabi imperfect",
        "brutalist architecture",
        "psychedelic",
        "retrofuturism",
        "abstract expressionism",
        "hyperrealism",
        "pop art",
        "neoclassical",
        "street art graffiti",
        "impressionist painterly",
        "cubist fragmented",
        "minimalist clean",
        "maximalist ornate",
        "gothic dark",
        "romantic sublime",
        "constructivist bold",
        "dadaist absurd",
        "futurist dynamic",
        "lowbrow pop surrealism",
        "expressionist painting",
        "film noir",
        "dark romanticism",
        "memento mori",
        "vanitas painting",
        "naive folk art",
        "outsider art",
        "brutalist graphic design",
        "swiss graphic design",
        "bauhaus poster",
        "art brut",
        "lyrical abstraction",
        "hard edge painting",
        "color field painting",
        "heavy impasto oil paint",
        "fluid watercolor wash",
        "sumi-e ink brush",
        "graphite pencil realism",
        "charcoal on parchment",
        "etched copperplate engraving",
        "stencil street art",
        "vitreous enamel",
        "gold leaf inlay",
        "cyanotype photogram",
        "risograph print",
        "sculpted marble",
        "spray paint",
        "colored pencil",
        "woodblock print",
        "linocut print",
        "acrylic paint",
        "oil pastel",
        "gouache",
        "fresco wall painting",
        "mosaic tiles",
        "stained glass",
        "encaustic wax",
        "screen print",
        "torn paper collage",
        "embroidery textile",
        "ceramic glaze",
        "sand art",
        "neon light",
        "digital matte painting",
        "chalk pavement",
        "graffiti spray",
        "wax crayon",
        "ink stippling",
        "monotype print",
        "lino etching",
        "batik fabric",
        "cut paper stencil",
        "bronze casting",
        "laser engraving",
        "",
    ],
    # ── SubjectMatter (was Subject — biological/social content) ───────────────
    "SubjectMatter": [
        # Faces and expressions
        "portrait face",
        "direct eye contact",
        "eyes averted downward",
        "smiling face",
        "laughing open mouth",
        "crying tears",
        "fearful wide eyes",
        "angry bared teeth",
        "disgusted face",
        "surprised face",
        "neutral blank expression",
        "elderly weathered face",
        "infant baby face",
        "uncanny CGI face",
        "human skull",
        "screaming face",
        "sleeping face",
        "face behind glass",
        "face underwater",
        "grief stricken face",
        "disfigured face",
        "masked face",
        # Social dynamics
        "embracing couple",
        "mother and newborn",
        "hostile threatening crowd",
        "lone figure in crowd",
        "two figures fighting",
        "figures kissing",
        "caregiver and patient",
        "religious ceremony crowd",
        "protest march",
        "funeral procession",
        # Bodies and action
        "full body portrait",
        "dancer mid-motion",
        "sprinting athlete",
        "hands reaching toward viewer",
        "finger pointing at viewer",
        "weapon aimed at viewer",
        "falling figure",
        "nude figure",
        "body silhouette",
        "bound figure",
        "kneeling figure",
        "figure in water",
        "body outline chalk",
        # Animals — threatening
        "predator eyes close-up",
        "snake striking",
        "spider macro",
        "wolf baring teeth",
        "bird of prey diving",
        "shark silhouette",
        "scorpion",
        "crocodile jaws",
        "wasp swarm",
        "bear charging",
        # Animals — neutral/gentle
        "newborn animal",
        "deep sea creature",
        "insect macro",
        "jellyfish bioluminescent",
        "butterfly wings",
        "sleeping cat",
        "newborn fawn",
        "whale",
        "elephant",
        "dolphin",
        "",
    ],

    "AbstractContent": [
        # Architecture and interior spaces
        "long dark corridor",
        "dark doorway entrance",
        "abandoned cathedral interior",
        "spiral staircase",
        "flooded interior room",
        "vast empty plain",
        "skyscraper edge looking down",
        "prison cell",
        "asylum ward",
        "burning room",
        "graveyard",
        "underground tunnel",
        "empty parking garage",
        "room of mirrors",
        "backrooms liminal",
        # Natural phenomena
        "crashing ocean wave",
        "lightning strike",
        "tornado funnel cloud",
        "volcanic eruption",
        "aurora borealis",
        "sandstorm",
        "avalanche",
        "solar eclipse",
        "meteor shower",
        # Flora and organic
        "blooming flower macro",
        "ancient tree canopy",
        "mushroom colony",
        "dripping ripe fruit",
        "feast table food spread",
        "rotting fruit",
        "carnivorous plant",
        "tangled thorns",
        "strangler fig roots",
        # Biological macro
        "eye iris extreme macro",
        "cell division microscope",
        "virus particle structure",
        "neural pathways brain",
        "mycelium network",
        "blood cells flowing",
        "DNA strand",
        # Objects and symbols
        "sacred geometry mandala",
        "burning fire",
        "pile of gold coins",
        "surgical medical tools",
        "religious icon",
        "broken mirror shards",
        "space nebula",
        "black hole",
        "gravestone",
        "coffin",
        "hourglass",
        "locked door",
        "shattered glass",
        "burning letters",
        "abandoned toys",
        "",
    ],

    "SceneContext": [
        "plain white background",
        "bokeh blurred background",
        "color gradient background",
        "thick fog atmosphere",
        "stormy dramatic sky",
        "misty morning light",
        "rainy wet street",
        "snowy winter landscape",
        "golden hour sunlight",
        "moonlit night",
        "underwater caustic light",
        "futuristic neon cityscape",
        "overgrown jungle ruins",
        "crowded market bazaar",
        "empty vast desert",
        "dense dark forest",
        "arctic frozen tundra",
        "volcanic wasteland",
        "coral reef underwater",
        "medieval stone village",
        "space station interior",
        "crystal cave interior",
        "cherry blossom garden",
        "industrial factory floor",
        "hospital sterile corridor",
        "library book stacks",
        "rooftop city panorama",
        "subway tunnel",
        "rocky mountain summit",
        "open ocean horizon",
        # new
        "childhood bedroom",
        "warm bakery interior",
        "brutalist megastructure",
        "abandoned theme park",
        "flooded city streets",
        "infinite server room",
        "salt flat infinite",
        "mangrove swamp",
        "bamboo forest",
        "lavender field",
        "canyon depth",
        "wildfire front",
        "",
    ],

    "Framing": [
        "centered subject",
        "rule of thirds",
        "perfect bilateral symmetry",
        "phi grid composition",
        "negative space dominant",
        "frame within a frame",
        "wide establishing shot",
        "extreme close-up",
        "dutch angle tilt",
        "vanishing point perspective",
        "diagonal leading lines",
        "panoramic wide",
        "one-point perspective",
        "two-point perspective",
        "isometric view",
        "fisheye distortion",
        "vertiginous height view",
        "worm's eye view",
        "bird's eye view",
        "first-person subjective",
        "impossible Escher geometry",
        "flat 2D overhead",
        "forced perspective",
        "split screen diptych",
        "circular vignette",
        "over the shoulder",
        "through window frame",
        "reflection in water",
        "mirror reflection",
        "triptych three panels",
        # new
        "through keyhole",
        "subject at edge",
        "tiny figure vast space",
        "radial outward burst",
        "",
    ],

    "DepthFocus": [
        "razor-thin depth of field",
        "deep infinite focus",
        "strong bokeh blur",
        "tilt-shift miniature",
        "anamorphic lens flare",
        "peripheral motion blur",
        "atmospheric haze depth",
        "tack-sharp full detail",
        "selective focus pull",
        "rack focus mid-ground",
        "defocused foreground",
        "macro focus stack",
        "zoom burst radial blur",
        "panning motion blur",
        "lens diffraction star",
        "chromatic aberration",
        "flat no depth cues",
        "aerial perspective fade",
        "lens distortion barrel",
        "soft dream focus",
        # new
        "vaseline soft focus",
        "underwater refraction",
        "heat haze shimmer",
        "",
    ],

    "Lighting": [
        "dramatic chiaroscuro",
        "golden hour warm",
        "blue hour cool",
        "harsh midday shadow",
        "rim lighting edge",
        "backlit silhouette",
        "bioluminescent glow",
        "volumetric god rays",
        "neon light flicker",
        "candlelight warm",
        "strobe freeze frame",
        "softbox studio even",
        "reflected bounce light",
        "single spotlight",
        "starlight minimal",
        "lightning flash",
        "UV blacklight glow",
        "firelight orange",
        "moonlight cold blue",
        "overcast flat diffuse",
        "colored gel lighting",
        "underlighting eerie",
        "natural window light",
        "dappled forest light",
        "underwater caustic light",
        "lava glow from below",
        "neon sign reflection",
        "fluorescent harsh",
        "infrared glowing",
        "emergency red light",
        # new
        "total darkness",
        "screen glow blue",
        "stage spotlight",
        "hospital fluorescent",
        "torch light caves",
        "light painting long",
        "contre-jour backlight",
        "",
    ],

    "Color": [
        "maximum vivid saturation",
        "muted desaturated tones",
        "monochromatic single hue",
        "sepia aged warm",
        "teal and orange cinematic",
        "complementary color contrast",
        "triadic color harmony",
        "ultra-black deep shadows",
        "pearlescent iridescent",
        "neon electric palette",
        "earthy ochre brown",
        "glowing luminous",
        "black and white red accent",
        "faded washed film look",
        "pastel soft",
        "jewel tones rich",
        "dominant red",
        "dominant orange",
        "dominant yellow",
        "dominant green",
        "dominant cyan",
        "dominant blue",
        "dominant purple",
        "dominant pink",
        "dominant gold",
        "dominant white",
        "dominant black",
        "rainbow full spectrum",
        "duotone two color",
        "infrared false color",
        # new
        "blood red only",
        "acid neon yellow",
        "toxic waste green",
        "deep bruise purple",
        "flesh tones warm",
        "bioluminescent blue",
        "rust and decay",
        "noir black white",
        "thermal heat map",
        "oil slick iridescent",
        "",
    ],

    "Surface": [
        "wet glistening skin",
        "cold brushed metal",
        "rough weathered stone",
        "soft velvet fabric",
        "crystalline lattice",
        "cracked dry earth",
        "polished obsidian",
        "tactile wood grain",
        "gossamer silk sheer",
        "translucent membrane",
        "heavy film grain",
        "halftone dots",
        "embossed raised relief",
        "frosted glass",
        "liquid mercury chrome",
        "translucent jade",
        "raw poured concrete",
        "oxidized copper verdigris",
        "flowing molten lava",
        "wispy smoke ethereal",
        "sculpted glacier ice",
        "woven carbon fiber",
        "subsurface scatter skin",
        "polished white bone",
        "rusted corroded metal",
        "veined marble",
        "cracked aged leather",
        "raw flesh meat",
        "fine beach sand",
        "rich dark soil",
        # new
        "peeling paint layers",
        "hammered beaten metal",
        "latex rubber shiny",
        "deep fur dense",
        "reptile scales",
        "soft moss",
        "barnacle encrusted",
        "burned charred",
        "foam sea froth",
        "spider web threads",
        "",
    ],

    "Pattern": [
        "tessellated hexagons",
        "recursive mandala",
        "voronoi cell diagram",
        "circuitry PCB traces",
        "leopard animal print",
        "concentric ripple rings",
        "Fibonacci golden spiral",
        "fractal tree branching",
        "floral repeat wallpaper",
        "topographic contour lines",
        "sine wave interference pattern",
        "geometric grid tiling",
        "camouflage disruptive",
        "radial spoke symmetry",
        "high contrast checkerboard",
        "vertical bar grating",
        "horizontal bar grating",
        "diagonal stripe pattern",
        "random dot noise",
        "Penrose aperiodic tiling",
        "Islamic geometric star",
        "Celtic knotwork",
        "herringbone weave",
        "polka dot grid",
        "Perlin noise texture",
        "DNA helix",
        "chain link mesh",
        "fish scale repeat",
        "op art illusion",
        "Moire interference",
        # moved from Subject
        "sine wave grating",
        "radial starburst pattern",
        "random white noise",
        # new
        "interference fringes",
        "cracked glaze pattern",
        "brick wall repeat",
        "water ripple rings",
        "fingerprint whorls",
        "",
    ],

    "ShapeLanguage": [
        "aggressive jagged spikes",
        "soothing circular forms",
        "stable pyramid triangles",
        "infinite Mobius loop",
        "sharp triangular wedges",
        "flowing parabolic curves",
        "unsettling asymmetry",
        "biomorphic organic blobs",
        "interlocking gear geometry",
        "concentric nested forms",
        "explosive radial burst",
        "drooping melting forms",
        "rigid orthogonal grid",
        "tangled knotted lines",
        "branching dendritic forms",
        "crystalline faceted forms",
        "undulating wave forms",
        "compressed crushed forms",
        "stretched elongated forms",
        "spiral helical",
        "fragmented shattered",
        "perfectly spherical",
        "hollow empty vessel",
        "dense packed forms",
        "cantilevered overhanging",
        # new
        "inflated bulging",
        "recursive nested",
        "mirrored bilateral",
        "chaotic entangled",
        "skeletal wireframe",
        "",
    ],

    "Mood": [
        "sublime overwhelming awe",
        "uncanny valley dread",
        "liminal eerie unease",
        "euphoric transcendent joy",
        "primal visceral fear",
        "serene meditative calm",
        "nostalgic bittersweet",
        "cozy intimate warmth",
        "epic cinematic grandeur",
        "deep melancholy sorrow",
        "romantic tender",
        "gritty raw brutal",
        "dreamlike dissociative",
        "ominous threatening",
        "hopeful uplifting",
        "suspenseful tense",
        "disgust revulsion",
        "acute danger urgent",
        "erotic charged",
        "violent aggressive rage",
        "hypnotic trance",
        "existential void",
        "childlike wonder",
        "sacred spiritual reverence",
        "grotesque body horror",
        "claustrophobic trapped",
        "vertiginous height fear",
        "feverish delirious",
        "sterile clinical cold",
        "overwhelming sensory overload",
        # new
        "wistful longing",
        "detached dissociation",
        "paranoid dread",
        "triumphant glory",
        "playful mischief",
        "solemn grief",
        "nihilistic void",
        "frantic panic",
        "hypnagogic trance",
        "cathartic release",
        "",
    ],

    "ScenePhysics": [
        "completely frozen still",
        "subtle breath movement",
        "slow drift motion",
        "walking steady pace",
        "running full speed",
        "explosive sudden burst",
        "slow motion",
        "frozen mid-air suspended",
        "swirling fluid vortex",
        "impact collision moment",
        "falling descent",
        "rising ascending",
        "oscillating rhythm",
        "chaotic turbulence",
        "rhythmic pulsing",
        "microscopic scale",
        "intimate human scale",
        "architectural scale",
        "citywide scale",
        "planetary scale",
        "cosmic galactic scale",
        "time lapse compressed",
        "shockwave expanding",
        "gravity defying float",
        # new
        "molecular scale",
        "cellular scale",
        "stellar scale",
        "shattering breaking",
        "melting dissolving",
        "growing sprouting",
        "burning consuming",
        "",
    ],

    # ── NEW: Arousal ──────────────────────────────────────────────────────────
    # Physiological activation level — orthogonal to valence
    # EEG: broadband power, alpha suppression, pupil dilation
    "Arousal": [
        # Near zero
        "comatose",
        "flatlined",
        "stone statue",
        "time has stopped",
        "body in still water",
        # Very low
        "deeply sedated",
        "cave silence",
        "lotus pond dawn",
        "barely breathing",
        "meditative trance",
        "hypnotic drift",
        # Low
        "deeply calm",
        "sleeping village",
        "slow tide",
        "drowsy",
        "peaceful",
        "lethargic",
        # Mild
        "relaxed",
        "quietly aware",
        "mildly alert",
        "gently curious",
        "softly focused",
        # Moderate
        "alert",
        "focused",
        "engaged",
        "anticipating",
        "tense",
        # High
        "excited",
        "intense",
        "agitated",
        "urgent",
        "alarmed",
        # Very high
        "breathless",
        "frantic",
        "heart pounding",
        "panicking",
        "chase happening",
        # Maximum
        "sensory overload",
        "explosion moment",
        "strobe maximum",
        "electrifying",
        "seizure edge",
        "",
    ],

    # ── NEW: Valence ──────────────────────────────────────────────────────────
    # Affective value — good to bad axis
    # EEG: frontal alpha asymmetry, LPP, vmPFC
    "Valence": [
        # Extremely positive
        "ecstatic bliss",
        "transcendent joy",
        "triumphant",
        "golden light pouring",
        "rapturous",
        # Strongly positive
        "joyful",
        "euphoric",
        "loving tender",
        "grateful",
        "celebratory",
        # Moderately positive
        "hopeful",
        "playful",
        "content",
        "warm pleasant",
        "nostalgic warm",
        # Mildly positive
        "cozy",
        "charming",
        "romantic",
        "quietly happy",
        "gentle comfort",
        # Neutral
        "neutral",
        "ambivalent",
        "contemplative",
        "curious",
        "detached",
        # Mildly negative
        "melancholic",
        "wistful",
        "bittersweet",
        "lonely",
        "pensive",
        # Moderately negative
        "sorrowful",
        "anxious",
        "troubled",
        "regretful",
        "despairing",
        # Strongly negative
        "horrifying",
        "anguished",
        "revolting",
        "terrifying",
        "traumatic",
        # Extremely negative
        "nihilistic",
        "void consuming",
        "abject terror",
        "all hope gone",
        "",
    ],

    # ── NEW: ThreatProximity ──────────────────────────────────────────────────
    # Safety to lethal danger imminence
    # EEG: P300, LPP, sustained amygdala slow wave
    "ThreatProximity": [
        # Completely safe
        "completely safe",
        "nurturing sanctuary",
        "protected warmth",
        "healing comfort",
        "fortress walls",
        # Benign
        "neutral",
        "inviting",
        "open path",
        "harmless",
        "daylight street",
        # Slightly unsettling
        "unsettling quiet",
        "wrong somehow",
        "door ajar darkness",
        "footsteps behind",
        "uncanny stillness",
        # Low threat
        "being watched",
        "followed",
        "something hidden",
        "window watcher",
        "sinister undercurrent",
        # Moderate threat
        "cornered",
        "surrounded",
        "predator visible",
        "fire spreading",
        "water rising",
        # Active danger
        "weapon drawn",
        "attack beginning",
        "building collapsing",
        "flames surrounding",
        "hostile advance",
        # Imminent death
        "gun at head",
        "jaws closing",
        "impact half second",
        "cliff crumbling",
        "inescapable",
        # Catastrophic
        "mass casualty",
        "city burning",
        "apocalyptic",
        "extinction event",
        "",
    ],
}

def flatten_art_data(ad: Dict) -> tuple[list[str], list[list[str]]]:
    categories = list(ad.keys())
    options: list[list[str]] = []
    for k in categories:
        v = ad[k]
        if isinstance(v, dict):
            merged = []
            for s in v.values():
                merged.extend(s)
            options.append(list(merged))
        else:
            options.append(list(v))
    return categories, options

class StructuredArtPromptSpace(SearchSpace):
    """
    SearchSpace for prompts built by picking one option per category.
    semantic mutation using precomputed embeddings (e.g. CLIP text).
    """

    def __init__(
        self,
        art_data: Dict,
        option_embeddings: Optional[list[torch.Tensor]] = None,
        semantic_temperature: float = 0.2,
    ):
        self.categories, self.options = flatten_art_data(art_data)
        self.option_embeddings = option_embeddings
        self.semantic_temperature = semantic_temperature
        self.sizes = np.array([len(o) for o in self.options], dtype=np.int32)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int32)
        self.feat_dim = int(np.sum(self.sizes))

    def random_candidate(self) -> Candidate:
        genes = [random.randrange(len(self.options[i])) for i in range(len(self.options))]
        return Candidate(genes=tuple(genes))

    def decode(self, cand: Candidate) -> str:
        parts = [self.options[i][cand.genes[i]] for i in range(len(self.options))]
        parts = [p for p in parts if p and p.strip()]
        return ", ".join(parts)
    
    def encode(self, prompt: str) -> Optional[List[int]]:
        """
        Reverse of decode(): map a prompt string back to a gene list.

        The prompt is split on ", " and each token is matched against the
        options for each category in order, skipping empty-string slots.
        Returns None if the prompt cannot be faithfully reconstructed
        (e.g. it came from outside this space).
        """
        tokens = [t.strip() for t in prompt.split(", ") if t.strip()]
        genes  = [len(self.options[i]) - 1 for i in range(len(self.options))]

        token_idx = 0
        for cat_idx, opts in enumerate(self.options):
            if token_idx >= len(tokens):
                break
            tok = tokens[token_idx]
            if tok in opts:
                genes[cat_idx] = opts.index(tok)
                token_idx += 1

        if token_idx < len(tokens):
            return None

        return genes

    def mutate(self, cand: Candidate, mutation_rate: float) -> Candidate:
        genes = list(cand.genes)
        for i in range(len(genes)):
            if random.random() < mutation_rate:
                genes[i] = random.randrange(len(self.options[i]))
        return Candidate(genes=tuple(genes))

    def semantic_mutate(self, cand: Candidate, mutation_rate: float) -> Candidate:
        if self.option_embeddings is None:
            return self.mutate(cand, mutation_rate)

        genes = list(cand.genes)
        for i in range(len(genes)):
            if random.random() < mutation_rate:
                current_idx = genes[i]
                emb = self.option_embeddings[i] 
                sims = emb @ emb[current_idx]   
                probs = torch.softmax(sims / self.semantic_temperature, dim=0).cpu().numpy()
                genes[i] = int(np.random.choice(len(probs), p=probs))
        return Candidate(genes=tuple(genes))

    def crossover(self, a: Candidate, b: Candidate, crossover_rate: float) -> tuple[Candidate, Candidate]:
        if random.random() > crossover_rate:
            return a, b
        point = random.randrange(1, len(a.genes))
        c1 = a.genes[:point] + b.genes[point:]
        c2 = b.genes[:point] + a.genes[point:]
        return Candidate(tuple(c1)), Candidate(tuple(c2))
    
    def hamming(self, a, b) -> int:
        return int(sum(int(x != y) for x, y in zip(a, b)))

    def featurize(self, ind) -> np.ndarray:
        x = np.zeros((self.feat_dim,), dtype=np.float32)
        for ci, opt_idx in enumerate(ind):
            x[int(self.offsets[ci] + opt_idx)] = 1.0
        return x