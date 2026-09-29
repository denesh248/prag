from llm_guard.input_scanners import PromptInjection
from llm_guard.output_scanners import Sensitive

from llm_guard.input_scanners import Scanner as InputScanner
from llm_guard.output_scanners import Scanner as OutputScanner


input_scanner = InputScanner(
    scanners=[
        PromptInjection()
    ]
)


output_scanner = OutputScanner(
    scanners=[
        Sensitive()
    ]
)


def check_input(prompt):

    sanitized_prompt, is_valid, score = input_scanner.scan(
        prompt
    )

    return {
        "prompt": sanitized_prompt,
        "is_safe": is_valid,
        "score": score
    }


def check_output(prompt, answer):

    sanitized_answer, is_valid, score = output_scanner.scan(
        prompt,
        answer
    )

    return {
        "answer": sanitized_answer,
        "is_safe": is_valid,
        "score": score
    }