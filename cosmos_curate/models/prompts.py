# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prompts for captioning."""

from loguru import logger

_PROMPTS = {
    "default": """
        Elaborate on the visual and narrative elements of the video in detail.
    """,
    "av": """
        The video depicts the view from a camera mounted on a car as it is driving.
        Pay special attention to the motion of the cars, including the primary car
        whose point-of-view we observe in the video. Also note important factors
        that would relate to driving safety like the relative positions of pedestrians,
        lane markers, road signs, traffic signals, and any aggressive driving behavior
        of other vehicles. Also pay attention to interesting landmarks and describe
        them in detail.
    """,
    "av-surveillance": """
        The video depicts the view from a surveillance camera. Pay special attention
        to the motion of the cars and other important factors that would relate to
        driving safety like the relative positions of pedestrians, lane markers,
        road signs, traffic signals, and any aggressive driving behavior of vehicles.
        Also pay attention to interesting landmarks and describe them in detail.
    """,
    "av-multiview": """
        You are observing a driving scene from three synchronized cameras mounted on a car.
        The first video is from CAM_FRONT_LEFT (left-facing camera).
        The second video is from CAM_FRONT (forward-facing camera, primary view).
        The third video is from CAM_FRONT_RIGHT (right-facing camera).
        Describe the complete scene using information from all three views. Pay special
        attention to the motion of vehicles, pedestrians crossing or approaching from any
        direction, lane markings, road signs, traffic signals, and any driving safety
        factors visible across all cameras. Note objects that appear in multiple views
        and describe how the scene evolves over time.
    """,
    "agibot_old_50vids": """
        You will be given a video trajectory showing robotic arms performing a task. Your goal is to generate detailed, step-by-step captions that explain what is happening and why at each moment.

        Your captions should describe the task, the environment, the robot's actions, and the reasoning behind each movement. Focus on clear, visual, and action-oriented descriptions that match what is visible in the video.

        First, give a comprehensive description of the task. Describe the overall activity, the main objects and their positions, how the robot interacts with them, the workspace layout, any obstacles, and the sequence of major actions. Then list the high-level movements that were executed and explain why each movement was necessary.

        For each step in the trajectory, generate a detailed caption explaining what remains to be done, what has already been completed, which objects are currently relevant, what the robot is preparing to do next, and why the current action is appropriate.

        All descriptions must match the visual evidence in the video. Do not invent objects or actions. Be precise, visual, and consistent with the trajectory. Include every step and do not group steps.
    """,
    "youcook2": """
    You are a cooking video captioner. You will receive short clips from cooking tutorial videos
    and must produce a concise, accurate caption describing the cooking step shown.

    ## TASK

    Watch the clip and write a single caption (1–2 sentences) describing:
    1. The specific cooking action being performed (e.g., stir, chop, pour, add, simmer, fry)
    2. The key ingredient(s) or item(s) involved
    3. Any important detail about technique or quantity, if clearly visible

    ## GUIDELINES

    - Be concise and literal — describe only what is visible in the clip
    - Name the action plainly: add, stir, chop, pour, mix, sauté, boil, season, drain, etc.
    - Name specific ingredients when identifiable; use a descriptive term if unsure
      (e.g. "a dark sauce", "the leafy greens", "the dough")
    - Do NOT describe the kitchen, the cook's clothing, lighting, or background decor
    - Do NOT elaborate on atmosphere or narrative — focus only on the cooking action
    - If nothing cooking-related happens, describe the food item visible

    ## EXAMPLES

    Clip: person pours olive oil into a hot pan
    Caption: "Pour olive oil into a heated pan."

    Clip: hands chop a handful of parsley on a cutting board
    Caption: "Finely chop the fresh parsley on a cutting board."

    Clip: stir a thick red sauce in a pot with a wooden spoon
    Caption: "Stir the tomato sauce in the pot over medium heat."
    """,
    "agibot": """
    You are a robotic manipulation video captioner. You will receive short POV video clips (~8.5 seconds) from a robot and must produce a concise, accurate caption describing what happens.
    
    ---
    
    ## TASK
    
    Watch the clip and write a single caption (1–3 sentences) describing:
    1. What the robot does (motion, direction, action)
    2. What object is involved, if any
    3. The outcome or end state, if visible
    ---
    
    ## GUIDELINES
    
    - Be concise and literal — describe only what is visible
    - Use simple spatial language: left, right, forward, above, toward, away
    - Name the action plainly: moves, reaches, grasps, places, releases, slides, rotates
    - If nothing notable happens, say so (e.g., "The arm moves from left to right without interacting with any object.")
    - Do not infer intent or over-interpret — stick to what is observable
    - Avoid jargon unless clearly applicable (e.g., "gripper" is fine)
    ## EXAMPLES
    
    Clip: arm swings from left side of frame to right, no object contact
    ```json
    {
    "caption": "The robotic arm sweeps from left to right across the workspace without contacting any object.",
    "action": "move",
    "object": null
    }
    ```
    
    Clip: gripper descends, closes around a small red block, lifts slightly
    ```json
    {
    "caption": "The gripper lowers toward a red block on the surface, closes around it, and lifts it a few centimeters.",
    "action": "grasp",
    "object": "red block"
    }
    ```
    """,
    "robot_reason": """
    Describe what happens in this short robot point-of-view clip in detail, getting the object
    identity and the action exactly right.

    First reason about what the object is from its visible features; then write a detailed,
    factual caption. Cover the specific object being manipulated, the action and its phases, the
    spatial context, and the outcome. Rich, concrete visual detail is welcome — it is useful for
    downstream tasks — provided every detail is actually visible in the clip.

    Output a single JSON object and nothing else:
    {"caption": "<a detailed, factual description naming the specific object, the action, and the outcome>", "action": "<the main action verb>", "object": "<the specific object being manipulated, or null if none>"}

    Rules:
    - Name the object as specifically as the visual evidence supports. If the exact type is
      unclear, describe its distinguishing visible features (shape, colour, size, texture)
      rather than guessing a common object.
    - Include only detail you can actually see. Do NOT invent objects, attributes, or actions,
      and do not infer intent — everything written must be grounded in the video.
    """,
    "inhard": """
    You are an industrial assembly video captioner. You will receive short clips from a
    factory assembly task recorded from a fixed overhead camera and must produce a concise,
    accurate caption describing what the worker is doing.

    ## TASK

    Watch the clip and write a single caption (1–2 sentences) describing:
    1. The specific action being performed (e.g., pick up, place, turn, fasten, consult)
    2. The tool or component involved (e.g., screwdriver, measuring rod, subsystem part, sheet)
    3. The direction or hand used if clearly visible (left hand, right side, in front)

    ## GUIDELINES

    - Be concise and literal — describe only what is visible
    - Name the action plainly: picks up, places, fastens, turns, consults, assembles
    - Name the object specifically when identifiable; use a descriptive term if unsure
      (e.g. "a small component", "a metal rod", "a sheet of paper")
    - If the clip is very short (under 1 second), describe the dominant action or object
      visible even if the motion is not fully shown
    - Do NOT describe the background, lighting, camera angle, or clothing
    - Do NOT infer intent beyond what is visible

    ## EXAMPLES

    Clip: worker's right hand reaches forward and picks up a screwdriver from the table
    Caption: "The worker picks up a screwdriver with their right hand."

    Clip: worker places a small metal component into a slot on the assembly board
    Caption: "The worker places a small metal component into the assembly board."

    Clip: worker glances down at a paper sheet on the left side of the workspace
    Caption: "The worker consults a paper sheet on the left side of the workspace."
    """,
}


_ENHANCE_PROMPTS = {
    "default": """
        You are a chatbot that enhances video caption inputs, adding more color and details to the text.
        The output should be longer than the provided input caption.
        Respond only with the enhanced caption; do not ask follow-up questions or offer additional assistance.
    """,
    "av-surveillance": """
        You are a chatbot that enhances video captions from vehicle dashboard cameras or surveillance cameras.
        Add more details and generate a summary from the original text.
        The output should be longer than the provided input caption.
    """,
}


_DEFAULT_STAGE2_PROMPT = """
Improve and refine following video description. Focus on highlighting the key visual and sensory elements.
Ensure the description is clear, precise, and paints a compelling picture of the scene.
"""


# Named stage-2 prompts, selectable by name via ``--qwen-stage2-prompt-text <name>``.
#
# ``verify`` implements self-verification (a "Self-Refine"-style second pass): the reasoning
# model re-watches the same clip together with its own first-pass description and is asked to
# critically re-examine the object identity and action, correcting any error. Unlike the default
# "make it more vivid" refinement (which inflates verbosity and can add ungrounded detail), this
# is designed to raise factual accuracy — it is the intended way to use a reasoning VLM's second
# pass. It is domain-general: it names no specific objects, it instructs the model to reason from
# visible physical features, so it transfers to any manipulation domain unchanged.
_STAGE2_PROMPTS: dict[str, str] = {
    "verify": """
Re-watch the clip and critically verify the description below before finalising it. Do not simply
restate it — check it against what is actually visible:
- Object: look again at the manipulated object's shape, colour, size, texture, and any markings.
  Is it named correctly and as specifically as the evidence supports? If the evidence points to a
  different object, correct it. If the exact type cannot be determined, describe its distinguishing
  features instead of guessing.
- Action: confirm the action and its phase (approach, reach, grasp, lift, move, place, release,
  idle) are correct.
- Remove anything not actually visible.
Output only a single JSON object: {"caption": "<corrected concise caption>", "action": "<verb>", "object": "<specific object or null>"}.

Description to verify:
""",
}


def get_prompt(
    prompt_variant: str,
    prompt_text: str | None,
    *,
    verbose: bool = False,
) -> str:
    """Get the captioning prompt.

    Args:
        prompt_variant: The variant of the prompt.
        prompt_text: The text of the prompt.
        verbose: Whether to print the prompt.

    Returns:
        The captioning prompt.

    Raises:
        ValueError: If the prompt variant is invalid.

    """
    if prompt_text is not None:
        prompt = prompt_text
    else:
        if prompt_variant not in _PROMPTS:
            error_msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error_msg)
        prompt = _PROMPTS[prompt_variant]
    if verbose:
        logger.debug(f"Captioning prompt: {prompt}")
    return prompt


def get_enhance_prompt(prompt_variant: str, prompt_text: str | None, *, verbose: bool = False) -> str:
    """Get the enhance captioning prompt.

    Args:
        prompt_variant: The variant of the prompt.
        prompt_text: The text of the prompt.
        verbose: Whether to print the prompt.

    Returns:
        The enhance captioning prompt.

    Raises:
        ValueError: If the prompt variant is invalid.

    """
    if prompt_text is not None:
        prompt = prompt_text
    else:
        if prompt_variant not in _ENHANCE_PROMPTS:
            error_msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error_msg)
        prompt = _ENHANCE_PROMPTS[prompt_variant]
    if verbose:
        logger.debug(f"Enhance Captioning prompt: {prompt}")
    return prompt


def get_stage2_prompt(prompt: str | None) -> str:
    """Get the stage 2 prompt.

    Args:
        prompt: The stage-2 prompt selector. If None, the default refinement prompt is used.
            If it matches a key in ``_STAGE2_PROMPTS`` (e.g. ``"verify"``), that named prompt is
            returned. Otherwise it is treated as literal prompt text.

    Returns:
        The stage 2 prompt.

    """
    if prompt is None:
        return _DEFAULT_STAGE2_PROMPT.strip() + "\n"
    if prompt in _STAGE2_PROMPTS:
        return _STAGE2_PROMPTS[prompt].strip() + "\n"
    return prompt
