"""Prompt templates consumed by the NLG layer (OpenAI chat completions).

Each template is a *plain string* with ``{placeholders}`` filled by callers.
"""

# ── System prompt (used for every chat completion) ────────
SYSTEM_PROMPT = """\
You are an expert interactive-fiction narrator for a text-adventure game.

Rules:
1. Always narrate in **second person** ("You see…", "You feel…").
2. Keep each response between 2-4 paragraphs.
3. Maintain absolute consistency with the world state provided.
4. Use vivid, sensory language — sights, sounds, smells.
5. Never mention game mechanics, stats, or that you are an AI.
6. Seamlessly incorporate the player's action into the narrative.
7. End the passage at a moment that invites the player to act next.
"""

# ── Opening scene ─────────────────────────────────────────
OPENING_PROMPT = """\
Create the opening scene of a {genre} text adventure.

Write 2-4 paragraphs establishing the setting, atmosphere, and a hook \
that draws the player into the story.  End with a situation where the \
player must make a choice.
"""

# ── Continue story ────────────────────────────────────────
STORY_CONTINUE_PROMPT = """\
{kg_summary}

Recent history:
{history}

The player's intent is "{intent}".
The player's emotional tone is: {emotion}
The player says: "{player_input}"

Continue the story in 2-4 vivid paragraphs, reacting to the player's action \
and emotional tone. Adjust the narrative mood to match: a {emotion} tone should \
influence pacing, word choice, and atmosphere. Advance the plot.
"""

# ── Option generation ─────────────────────────────────────
OPTION_GENERATION_PROMPT = """\
Given the latest story passage and world state below, generate exactly \
{num_options} player options as a JSON array.

Story passage:
{story_text}

World state:
{kg_summary}

Return ONLY a JSON object:
{{"options": [{{"text": "...", "intent_hint": "action|dialogue|explore|use_item|ask_info|rest|trade|other", "risk_level": "low|medium|high"}}]}}
"""
