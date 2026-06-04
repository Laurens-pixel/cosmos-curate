# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared judge prompts.

All prompts use ``{gt_action_text}`` and ``{caption_text}`` placeholders. Plugins call
``get_prompt(name).format(...)`` rather than hard-coding any prompt themselves.
"""

# Lenient binary judge — best-performing on AgiBotWorld (Gemma4 F1=0.859 on 30-vid manual).
LENIENT_BINARY = """\
You are checking whether a robot manipulation caption and a ground-truth action phrase
describe the same overall goal. Be lenient: wording and specificity can differ, but the
core intent must match.

Ground truth (what actually happened):
"{gt_action_text}"

Caption (what the model said):
"{caption_text}"

Judge as CORRECT if:
- The caption mentions the same object as the GT, even if phrased differently
  ("purple vegetable" for "onion", "mushroom" for "shiitake mushroom", "bottle" for
  "green tea bottle", "item" or "object" when context is clear — all fine)
- The caption conveys the same goal as the GT, even if it also describes extra steps
  (e.g. GT = "Retrieve onion" and caption says "picks up onion and places it in cart"
  — the retrieval goal is captured, so this is CORRECT)
- Minor differences in phrasing, perspective, or level of detail are acceptable

Judge as INCORRECT only if:
- The caption clearly names a different object (e.g. "cucumber" when GT is "mushroom")
- The caption describes the opposite action to the GT with no mention of the GT action
  (e.g. GT = "Retrieve X" but caption only says "places X down" without any picking)

Respond with exactly one word on the first line — CORRECT or INCORRECT — then one
sentence explaining why."""


# Strict 4-verdict prompt (kept for parity with earlier experiments). Returned verdict
# is one of: CORRECT, INCORRECT_OBJECT, INCORRECT_ACTION, INCORRECT.
STRICT_4VERDICT = """\
You are evaluating whether a robot manipulation caption correctly describes what happened.

Ground truth — what actually happened:
"{gt_action_text}"

Generated caption:
"{caption_text}"

Respond with exactly one of these four verdicts on the first line:
  CORRECT           — object and action are both right
  INCORRECT_OBJECT  — object is wrong, action is right
  INCORRECT_ACTION  — action is wrong, object is right
  INCORRECT         — both object and action are wrong

Then on the second line give a single sentence explaining your reasoning."""


# Lenient binary judge for cooking videos (YouCook2 domain). Action correctness is
# primary; ingredient naming is secondary and treated very leniently.
YOUCOOK2_BINARY = """\
You are checking whether a cooking video caption and a ground-truth description
describe the same cooking step. The action being performed is the most important
factor; ingredient identification is secondary.

Ground truth (what actually happened):
"{gt_action_text}"

Caption (what the model said):
"{caption_text}"

Judge as CORRECT if:
- The caption describes the same cooking action as the GT, even if phrased differently
  ("slicing" for "cutting", "adding" for "placing", etc.)
- Ingredient naming can differ widely — a vague description, a visual description
  ("the green vegetable", "a dark sauce"), a synonym, or even omitting the ingredient
  entirely is acceptable as long as the action is right
- The caption includes extra steps or context beyond the GT step — still CORRECT

Judge as INCORRECT only if:
- The caption clearly describes a different cooking action from the GT
  (e.g. "frying" when GT is "boiling", "removing from pan" when GT is "adding to pan")

Respond with exactly one word on the first line — CORRECT or INCORRECT — then one
sentence explaining why."""


# Lenient binary judge for autonomous driving / multi-camera AV footage (nuScenes domain).
# GT is a scene-level description tag string (e.g. "Parked truck, construction, intersection").
# Lenient on camera angle, perspective and temporal emphasis; strict on clearly wrong objects
# or scene types (e.g. indoor scene described as outdoor highway).
AV_BINARY = """\
You are checking whether a camera caption from an autonomous vehicle and a ground-truth
scene description refer to the same driving scenario. Be lenient: the caption may focus
on a subset of what the GT tags describe, may come from a different camera angle, or may
differ in emphasis — that is fine as long as it describes the same scene.

Ground truth scene tags (what was observed):
"{gt_action_text}"

Caption (what the model said):
"{caption_text}"

Judge as CORRECT if:
- The caption mentions at least one key element from the GT tags (road type, named objects,
  manoeuvre, or environmental condition such as night / rain)
- The caption is consistent with the overall driving context described by the GT, even if
  it focuses on a different detail
- Minor differences in emphasis, perspective, or level of detail are acceptable

Judge as INCORRECT only if:
- The caption clearly contradicts the GT (e.g. "highway at high speed" when GT is "parking lot",
  or "daytime sunny" when GT is "night")
- The caption describes a completely different scene type or set of objects with no overlap

Respond with exactly one word on the first line — CORRECT or INCORRECT — then one
sentence explaining why."""


# Lenient binary judge for industrial assembly footage (InHARD domain).
# GT is the action class label from the dataset (e.g. "Take screwdriver", "Assemble system").
# Labels are coarse class names — accept any caption that describes the right action + tool
# category, even if it uses different wording or describes additional context.
INHARD_BINARY = """\
You are checking whether an industrial assembly video caption and a ground-truth action
label describe the same worker action. Be lenient: the caption may use different wording
or describe additional context, but the core action and the tool/component must match.

Ground truth action label:
"{gt_action_text}"

Caption (what the model said):
"{caption_text}"

Judge as CORRECT if:
- The caption describes the same type of action as the GT label
  ("picks up" for "Take", "sets down" or "places" for "Put down", "reads" or "looks at"
  for "Consult", "fastens" or "tightens" for "Assemble system" — all acceptable)
- The caption mentions the same tool or component category as the GT, even if vaguely
  ("a tool" for "screwdriver", "a part" for "component", "a rod" for "measuring rod" — fine)
- The caption includes additional motion or context beyond the GT label — still CORRECT

Judge as INCORRECT only if:
- The caption describes a clearly different action type (e.g. "picks up" when GT is "Put down")
- The caption mentions a tool/component that clearly contradicts the GT
  (e.g. "screwdriver" when GT is "measuring rod")
- The caption describes a completely unrelated activity with no overlap

Respond with exactly one word on the first line — CORRECT or INCORRECT — then one
sentence explaining why."""


_PROMPTS: dict[str, str] = {
    "av_binary": AV_BINARY,
    "inhard_binary": INHARD_BINARY,
    "lenient_binary": LENIENT_BINARY,
    "strict_4verdict": STRICT_4VERDICT,
    "youcook2_binary": YOUCOOK2_BINARY,
}


def get_prompt(name: str) -> str:
    """Return the prompt template for ``name``. Raises if unknown."""
    if name not in _PROMPTS:
        msg = f"Unknown judge prompt: {name!r}. Registered: {sorted(_PROMPTS)}"
        raise ValueError(msg)
    return _PROMPTS[name]


def list_prompts() -> list[str]:
    """Return the sorted list of registered prompt names."""
    return sorted(_PROMPTS)
