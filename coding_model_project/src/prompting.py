from __future__ import annotations


SYSTEM_PROMPT = """You are an expert Python programmer.

Output rules:
1. Output Python code only.
2. Include necessary imports only if needed.
3. Wrap the entire code in <code> and </code>.
4. Do not write anything outside the <code> tags.
5. Follow dataset-specific constraints given by the user prompt (function-only vs full program)."""


PROMPT_TEMPLATES = {
    "humaneval": """Complete the following Python function.

Rules:
- Keep the function name, parameters, and docstring unchanged.
- Output a complete, executable Python code snippet that defines the function.
- Use only Python standard library (no pip packages).
- Do NOT read from stdin and do NOT print anything.
- Do NOT include "if __name__ == '__main__':" or any top-level execution.
- Do NOT define a function named "check" (it is reserved for tests).

{prompt}

Output ONLY:
<code>
# python code
</code>""",
    "mbpp_reg": """Implement a Python function for the following task.

Task:
{prompt}

Rules:
- The function name MUST be: {entry_point}
- Your function will be called like: {example_call}
- Use only Python standard library (no pip packages).
- Do NOT read from stdin and do NOT print anything.
- Do NOT include "if __name__ == '__main__':" or any top-level execution.

Output ONLY:
<code>
# python code
</code>""",
    "codecontests_train": """Solve the following competitive programming problem in Python.

Rules:
- Read from stdin and write to stdout.
- Your program MUST produce output when executed (call solve() under main guard, or execute at top-level).
- Use fast I/O if needed (sys.stdin.buffer).
- Do NOT print anything except the required output.

{prompt}

Output ONLY:
<code>
# python code
</code>""",
    "codecontests_valid": """Solve the following competitive programming problem in Python.

Rules:
- Read from stdin and write to stdout.
- Your program MUST produce output when executed (call solve() under main guard, or execute at top-level).
- Use fast I/O if needed (sys.stdin.buffer).
- Do NOT print anything except the required output.

{prompt}

Output ONLY:
<code>
# python code
</code>""",
    "codecontests_valid_big": """Solve the following competitive programming problem in Python.

Rules:
- Read from stdin and write to stdout.
- Your program MUST produce output when executed (call solve() under main guard, or execute at top-level).
- Use fast I/O if needed (sys.stdin.buffer).
- Do NOT print anything except the required output.

{prompt}

Output ONLY:
<code>
# python code
</code>""",
    "codecontests_test": """Solve the following competitive programming problem in Python.

Rules:
- Read from stdin and write to stdout.
- Your program MUST produce output when executed (call solve() under main guard, or execute at top-level).
- Use fast I/O if needed (sys.stdin.buffer).
- Do NOT print anything except the required output.

{prompt}

Output ONLY:
<code>
# python code
</code>""",
}


def format_prompt(raw_prompt: str, dataset_key: str, entry_point: str = "", example_call: str = "") -> str:
    template = PROMPT_TEMPLATES.get(dataset_key)
    if not template:
        return f"""Solve the following problem in Python.

{raw_prompt}

Output ONLY:
<code>
# python code
</code>"""

    if dataset_key == "mbpp_reg":
        if not entry_point:
            raise ValueError(f"MBPP entry_point is empty for prompt: {raw_prompt[:50]}...")
        if not example_call:
            raise ValueError(f"MBPP example_call is empty for prompt: {raw_prompt[:50]}...")
        return template.format(prompt=raw_prompt, entry_point=entry_point, example_call=example_call)

    return template.format(prompt=raw_prompt)
