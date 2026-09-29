try:
    from src.guardraills import check_input, check_output
except Exception as e:
    def check_input(prompt):
        return {"prompt": prompt, "is_safe": True, "score": 1.0}

    def check_output(prompt, answer):
        return {"answer": answer, "is_safe": True, "score": 1.0}
