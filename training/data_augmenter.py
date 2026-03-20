"""Data augmenter for intent classification training.

Generates 500+ samples per intent class using:
1. Template-based sentence generation
2. Synonym substitution
3. Sentence structure variation

No LLM API required — pure programmatic generation.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

# ── Synonym dictionaries ─────────────────────────────────
_SYNONYMS: Dict[str, List[str]] = {
    # Action verbs
    "attack": ["strike", "hit", "assault", "charge", "slash", "lunge", "pounce", "ambush"],
    "fight": ["battle", "combat", "engage", "duel", "clash", "grapple", "spar"],
    "defend": ["block", "parry", "shield", "guard", "protect", "ward", "dodge"],
    "shoot": ["fire", "loose", "aim", "hurl", "throw", "launch"],
    "kill": ["slay", "vanquish", "defeat", "destroy", "eliminate", "overpower"],
    # Dialogue verbs
    "talk": ["speak", "chat", "converse", "discourse", "discuss", "address"],
    "ask": ["inquire", "query", "question", "interrogate", "request"],
    "tell": ["inform", "explain", "narrate", "relate", "describe", "reveal"],
    "greet": ["salute", "welcome", "hail", "bow", "nod", "wave"],
    "say": ["declare", "announce", "proclaim", "state", "mutter", "whisper", "shout"],
    # Exploration verbs
    "explore": ["survey", "scout", "roam", "traverse", "venture", "prowl"],
    "search": ["hunt", "seek", "rummage", "scour", "comb", "investigate"],
    "look": ["gaze", "glance", "peer", "stare", "observe", "examine", "inspect"],
    "examine": ["study", "inspect", "scrutinize", "analyze", "investigate", "check"],
    # Use item verbs
    "use": ["apply", "utilize", "employ", "activate", "deploy", "wield"],
    "drink": ["sip", "gulp", "quaff", "imbibe", "swallow", "consume"],
    "eat": ["consume", "devour", "munch", "nibble", "feast", "taste"],
    "equip": ["wear", "don", "put on", "strap on", "arm with", "outfit"],
    "open": ["unlock", "unseal", "unbar", "unlatch", "pry open", "force open"],
    # Rest verbs
    "rest": ["relax", "unwind", "recuperate", "recover", "repose", "pause"],
    "sleep": ["slumber", "nap", "doze", "snooze", "dream", "hibernate"],
    "camp": ["bivouac", "encamp", "settle", "make camp", "pitch tent"],
    # Trade verbs
    "buy": ["purchase", "acquire", "procure", "obtain", "invest in"],
    "sell": ["trade", "vend", "auction", "barter", "exchange"],
    "trade": ["barter", "exchange", "swap", "negotiate", "haggle"],
}

# ── Target/recipient templates ───────────────────────────
_ENEMIES = [
    "goblin", "troll", "orc", "dragon", "skeleton", "wolf", "bandit", "thief",
    "guard", "soldier", "warrior", "knight", "mage", "demon", "spider", "rat",
    "zombie", "vampire", "ogre", "giant", "wyvern", "assassin", "beast",
]
_NPCS = [
    "merchant", "blacksmith", "elder", "guard", "innkeeper", "priest", "wizard",
    "farmer", "king", "queen", "princess", "noble", "bard", "healer", "captain",
    "stranger", "traveler", "villager", "hermit", "sage", "alchemist", "nurse",
]
_LOCATIONS = [
    "forest", "cave", "castle", "village", "dungeon", "tower", "temple", "tavern",
    "market", "shrine", "ruins", "palace", "bridge", "river", "mountain", "camp",
    "lair", "sanctuary", "library", "garden", "cemetery", "harbor", "sewer",
]
_ITEMS = [
    "sword", "shield", "potion", "scroll", "key", "map", "book", "gem", "ring",
    "staff", "bow", "dagger", "armor", "helmet", "cloak", "amulet", "wand",
    "torch", "rope", "food", "water", "bread", "gold", "crystal", "orb",
]

# ── Template-based generators ────────────────────────────

def _gen_action(n: int) -> List[str]:
    """Generate action intent examples."""
    templates = [
        "{verb} the {enemy}",
        "{verb} the {enemy} with my {item}",
        "I {verb} the {enemy}",
        "I will {verb} the {enemy}",
        "{verb} at the {enemy} with all my might",
        "Let me {verb} that {enemy}",
        "I must {verb} the {enemy} before it's too late",
        "{verb}! {verb} the {enemy} now!",
        "I charge at the {enemy} and {verb} it",
        "I {verb} the {enemy} from behind",
        "Time to {verb} this {enemy}",
        "I ready my {item} and {verb} the {enemy}",
        "{verb} the {enemy} with a powerful blow",
        "I try to {verb} the {enemy} but miss",
        "Carefully, I {verb} the {enemy}",
    ]
    verbs = ["attack", "fight", "strike", "defend", "shoot", "kill", "slash", "hit"]
    results = []
    for _ in range(n):
        t = random.choice(templates)
        verb = random.choice(verbs)
        synonym = random.choice(_SYNONYMS.get(verb, [verb]))
        results.append(t.format(
            verb=synonym,
            enemy=random.choice(_ENEMIES),
            item=random.choice(_ITEMS),
        ))
    return results


def _gen_dialogue(n: int) -> List[str]:
    """Generate dialogue intent examples."""
    templates = [
        "{verb} to the {npc}",
        "{verb} with the {npc}",
        "I {verb} to the {npc}",
        "{verb} the {npc} about the quest",
        "I want to {verb} to the {npc}",
        "Let me {verb} with the {npc}",
        "{verb} hello to the {npc}",
        "I {verb} the {npc} for information",
        "Can I {verb} to the {npc}?",
        "I approach the {npc} and {verb}",
        "{verb} quietly to the {npc}",
        "I need to {verb} with the {npc} urgently",
        "{verb} the {npc} what happened here",
        "I politely {verb} the {npc}",
    ]
    verbs = ["talk", "speak", "ask", "tell", "greet", "say", "chat", "discuss"]
    results = []
    for _ in range(n):
        t = random.choice(templates)
        verb = random.choice(verbs)
        synonym = random.choice(_SYNONYMS.get(verb, [verb]))
        results.append(t.format(verb=synonym, npc=random.choice(_NPCS)))
    return results


def _gen_explore(n: int) -> List[str]:
    """Generate explore intent examples."""
    templates = [
        "{verb} the {location}",
        "{verb} around the {location}",
        "I {verb} the {location}",
        "Let me {verb} this {location}",
        "{verb} the area for clues",
        "I {verb} deeper into the {location}",
        "{verb} the {location} carefully",
        "I want to {verb} the {location}",
        "{verb} ahead for danger",
        "I {verb} the {location} for hidden passages",
        "Let's {verb} the {location}",
        "I decide to {verb} the {location}",
        "{verb} the nearby {location}",
        "I {verb} every corner of the {location}",
    ]
    verbs = ["explore", "search", "look", "investigate", "examine", "survey", "scout"]
    results = []
    for _ in range(n):
        t = random.choice(templates)
        verb = random.choice(verbs)
        synonym = random.choice(_SYNONYMS.get(verb, [verb]))
        results.append(t.format(verb=synonym, location=random.choice(_LOCATIONS)))
    return results


def _gen_use_item(n: int) -> List[str]:
    """Generate use_item intent examples."""
    templates = [
        "{verb} the {item}",
        "{verb} my {item}",
        "I {verb} the {item}",
        "{verb} the {item} on the {enemy}",
        "Let me {verb} this {item}",
        "I {verb} the {item} to heal",
        "Time to {verb} the {item}",
        "{verb} the {item} immediately",
        "I {verb} the {item} from my bag",
        "I should {verb} the {item} now",
        "{verb} the {item} and hope for the best",
        "I carefully {verb} the {item}",
    ]
    verbs = ["use", "drink", "eat", "equip", "open", "apply", "activate", "throw"]
    results = []
    for _ in range(n):
        t = random.choice(templates)
        verb = random.choice(verbs)
        synonym = random.choice(_SYNONYMS.get(verb, [verb]))
        results.append(t.format(verb=synonym, item=random.choice(_ITEMS), enemy=random.choice(_ENEMIES)))
    return results


def _gen_ask_info(n: int) -> List[str]:
    """Generate ask_info intent examples."""
    templates = [
        "What is this {location}?",
        "Where can I find the {npc}?",
        "Who is the {npc}?",
        "How do I get to the {location}?",
        "Why is the {location} abandoned?",
        "What happened to the {npc}?",
        "Tell me about the {item}",
        "Describe the {enemy}",
        "What is this {item} for?",
        "Where does the {npc} live?",
        "How dangerous is the {location}?",
        "Who built this {location}?",
        "What creatures live in the {location}?",
        "Can you explain the map?",
        "What does this inscription say?",
        "Where am I?",
        "What is the meaning of this?",
        "How did the {npc} get here?",
        "Why are the {enemy}s attacking?",
        "What lies beyond the {location}?",
        "Tell me the history of this {location}",
        "What kind of {item} is this?",
        "How old is this {location}?",
        "Who rules this {location}?",
    ]
    results = []
    for _ in range(n):
        t = random.choice(templates)
        results.append(t.format(
            location=random.choice(_LOCATIONS),
            npc=random.choice(_NPCS),
            item=random.choice(_ITEMS),
            enemy=random.choice(_ENEMIES),
        ))
    return results


def _gen_rest(n: int) -> List[str]:
    """Generate rest intent examples."""
    templates = [
        "{verb} by the campfire",
        "{verb} at the inn",
        "I need to {verb}",
        "Let me {verb} for a while",
        "{verb} until morning",
        "I want to {verb}",
        "Time to {verb}",
        "{verb} here for the night",
        "I should {verb} and recover",
        "{verb} to restore my strength",
        "Let's set up camp and {verb}",
        "I find a safe spot to {verb}",
        "{verb} before continuing",
        "I sit down and {verb}",
        "I need a moment to {verb}",
    ]
    verbs = ["rest", "sleep", "wait", "camp", "relax", "meditate", "nap", "heal"]
    results = []
    for _ in range(n):
        t = random.choice(templates)
        verb = random.choice(verbs)
        synonym = random.choice(_SYNONYMS.get(verb, [verb]))
        results.append(t.format(verb=synonym))
    return results


def _gen_trade(n: int) -> List[str]:
    """Generate trade intent examples."""
    templates = [
        "{verb} with the {npc}",
        "{verb} a {item} from the {npc}",
        "I want to {verb} my {item}",
        "Let me {verb} for supplies",
        "{verb} my {item} for gold",
        "Can I {verb} here?",
        "I need to {verb} some {item}s",
        "{verb} with the {npc} for a better price",
        "I {verb} my old {item}",
        "Time to {verb} at the market",
        "I offer to {verb} the {item}",
        "{verb} and get better equipment",
    ]
    verbs = ["trade", "buy", "sell", "barter", "exchange", "shop", "purchase", "bargain"]
    results = []
    for _ in range(n):
        t = random.choice(templates)
        verb = random.choice(verbs)
        synonym = random.choice(_SYNONYMS.get(verb, [verb]))
        results.append(t.format(verb=synonym, npc=random.choice(_NPCS), item=random.choice(_ITEMS)))
    return results


def _gen_other(n: int) -> List[str]:
    """Generate other/miscellaneous intent examples."""
    templates = [
        "Sing a song to pass the time",
        "Write a letter home",
        "Pray at the shrine",
        "Craft a {item} from materials",
        "Build a shelter",
        "Start a campfire",
        "Dance around the bonfire",
        "Do nothing for now",
        "Draw a picture in the dirt",
        "Meditate on my journey",
        "Count my {item}s",
        "Polish my {item}",
        "Practice my fighting stance",
        "Whistle a tune",
        "Watch the sunset",
        "Stare at the stars",
        "Clean my equipment",
        "Organize my inventory",
        "Stretch and warm up",
        "Reflect on past battles",
        "Sketch a map of where I've been",
        "Talk to myself about the quest",
        "Juggle some rocks",
        "Practice my magic",
        "Sharpen my {item}",
        "Study the sky for weather",
        "Smell the flowers nearby",
        "Pick some herbs",
        "Climb a tree to get a better view",
        "Sit and think",
    ]
    results = []
    for _ in range(n):
        t = random.choice(templates)
        results.append(t.format(item=random.choice(_ITEMS)))
    return results


# ── Public API ───────────────────────────────────────────

_GENERATORS = {
    "action": _gen_action,
    "dialogue": _gen_dialogue,
    "explore": _gen_explore,
    "use_item": _gen_use_item,
    "ask_info": _gen_ask_info,
    "rest": _gen_rest,
    "trade": _gen_trade,
    "other": _gen_other,
}


def generate_dataset(
    num_per_class: int = 500,
    output_path: str = "training/data/intent_train.jsonl",
    seed: int = 42,
) -> str:
    """Generate a balanced intent classification dataset.

    Returns the output file path.
    """
    random.seed(seed)

    all_samples = []
    for intent, generator in _GENERATORS.items():
        samples = generator(num_per_class)
        for text in samples:
            all_samples.append({"text": text, "intent": intent})

    random.shuffle(all_samples)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Generated {len(all_samples)} samples ({num_per_class}/class) → {output}")
    return str(output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate intent training data")
    parser.add_argument("--num_per_class", type=int, default=500)
    parser.add_argument("--output", default="training/data/intent_train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.num_per_class, args.output, args.seed)
