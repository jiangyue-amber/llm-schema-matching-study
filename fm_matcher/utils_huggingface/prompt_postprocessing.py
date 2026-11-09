import itertools
from typing import List, Dict, Any

from .models import (
    Answer,
    AttributePair,
    Decision,
    Parameters,
    Result,
    ResultPair,
    Vote,
)


def extract_json(answer: Answer) -> Dict[str, Any]:
    """Extract the JSON formatted summary from an Answer retrieved from Hugging Face model."""
    import json, re
    text = answer.answer

    # Strip <think>...</think> blocks produced by reasoning models like Qwen3
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try to find JSON block in markdown code blocks first (``` json ... ```)
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except Exception:
            pass

    # Scan for ALL JSON objects and return the first one that looks like a decision
    # (has "yes" or "no" top-level keys) — handles cases where the last { is not the right one
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            candidate = json.loads(match.group().replace("'", '"'))
            keys = {k.lower() for k in candidate.keys()}
            if "yes" in keys or "no" in keys:
                return candidate
        except Exception:
            continue

    # Last resort: try the last { ... } block (original behaviour)
    try:
        start_decision = text.rindex("{")
        end_decision = text.index("}", start_decision)
        raw_json = text[start_decision : end_decision + 1].replace("'", '"')
        return json.loads(raw_json)
    except ValueError:
        raise ValueError(f"Could not extract JSON from answer: {text[:200]}...")


def postprocess_answers(parameters: Parameters, answers: List[Answer]) -> Result:
    """Postprocess all answers into structured Results. At this point, we will assume that the answers are validated."""
    result = _generate_empty_result(parameters)
    for answer in answers:
        if not answer.valid:
            continue
            
        try:
            extracted_data = extract_json(answer)
        except:
            continue
            
        # Handle different potential JSON structures (yes/no keys)
        reversed_json = {}
        for decision_key, attribute_list in extracted_data.items():
            if isinstance(attribute_list, list):
                for attribute in attribute_list:
                    reversed_json[str(attribute)] = decision_key

        for src, trgt in itertools.product(
            answer.attributes.sources,
            answer.attributes.targets,
        ):
            if len(answer.attributes.sources) == len(answer.attributes.targets) == 1:
                # by convention, I use the target attribute name for 1 to 1
                look_for = trgt.name
            elif len(answer.attributes.sources) == 1:
                look_for = trgt.name
            elif len(answer.attributes.targets) == 1:
                look_for = src.name
            else:
                look_for = f"{src.name},{trgt.name}"
                
            if look_for in reversed_json:
                try:
                    vote_str = reversed_json[look_for].upper()
                    # Map potentially different output strings to Vote enum
                    if vote_str == "YES": vote = Vote.YES
                    elif vote_str == "NO": vote = Vote.NO
                    else: vote = Vote.UNKNOWN
                except KeyError:
                    vote = Vote.UNKNOWN
            else:
                vote = Vote.UNKNOWN
                
            result_pair = result.pairs[AttributePair(src, trgt)]
            decision = Decision(vote=vote, explanation=answer.answer, answer=answer)
            result_pair.votes.append(decision)
    return result


def _generate_empty_result(parameters: Parameters) -> Result:
    """Helper method that generates an empty, yet structured (i.e. filled with all attribute combinations) Result object."""
    attribute_combinations = list(
        itertools.product(
            parameters.source_relation.attributes, parameters.target_relation.attributes
        )
    )
    return Result(
        name="test",
        parameters=parameters,
        pairs={
            AttributePair(src, trgt): ResultPair(attributes=AttributePair(src, trgt))
            for (src, trgt) in attribute_combinations
        },
    )
