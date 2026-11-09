import asyncio
import copy
import json
import logging
import math
import time
from collections import deque
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.api_core.exceptions import TooManyRequests
import tenacity

from .config import config
from .errors import NotDoneException
from .models import Answer, Parameters, Prompt
from .storage import store_answer, store_gemini_completion, get_answers_by_prompt

# Rate-limit configuration.
_RAW_REQUESTS_PER_MINUTE: int = int(config.get("GEMINI_REQUESTS_PER_MINUTE", 1000))
_RAW_TOKENS_PER_MINUTE: int = int(config.get("GEMINI_TOKENS_PER_MINUTE", 1000000))
_RATE_LIMIT_BUFFER: float = float(config.get("GEMINI_RATE_LIMIT_BUFFER", 0.9))
_REQUEST_WINDOW_SECONDS: float = 60.0

_REQUESTS_PER_MINUTE: int = max(1, int(_RAW_REQUESTS_PER_MINUTE * _RATE_LIMIT_BUFFER))
_TOKENS_PER_MINUTE: int = max(1, int(_RAW_TOKENS_PER_MINUTE * _RATE_LIMIT_BUFFER))
_MIN_REQUEST_INTERVAL: float = float(
    config.get(
        "GEMINI_REQUEST_INTERVAL",
        max(0.0, 60.0 / max(1, _REQUESTS_PER_MINUTE)),
    )
)

_REQUEST_TIMESTAMPS = deque()
_TOKEN_BUCKET: float = float(_TOKENS_PER_MINUTE)
_LAST_TOKEN_REFILL: float = time.monotonic()
_last_request_time: float = 0.0
_request_lock = asyncio.Lock()  # module-level lock (created at import time in sync context)



def send_prompts(parameters: Parameters, prompts: List[Prompt]) -> List[Answer]:
    """Generate a result from parameters and prompts."""
    return asyncio.run(process_prompt_list(parameters, prompts))


@tenacity.retry(
    stop=tenacity.stop_after_attempt(6),
    wait=tenacity.wait_exponential_jitter(initial=10, max=120, jitter=10),
    retry=(
        tenacity.retry_if_exception_type(google_exceptions.DeadlineExceeded)
        | tenacity.retry_if_exception_type(google_exceptions.ResourceExhausted)
        | tenacity.retry_if_exception_type(google_exceptions.InternalServerError)
        | tenacity.retry_if_exception_type(google_exceptions.ServiceUnavailable)
        | tenacity.retry_if_exception_type(TooManyRequests)
    ),
    before_sleep=tenacity.before_sleep_log(
        logging.getLogger(__name__),
        logging.WARNING,
    ),
)
async def ask_gemini(prompt_dict: Dict[str, Any]) -> Any:
    """Send one request to Gemini with rate-limit–aware throttling and robust retries.

    Retried automatically on 429 (ResourceExhausted / TooManyRequests),
    503 (ServiceUnavailable), 500 (InternalServerError), and timeouts.
    A module-level lock + minimum interval ensures we never fire requests
    faster than _MIN_REQUEST_INTERVAL seconds apart.
    """
    global _last_request_time

    model = genai.GenerativeModel(prompt_dict["model"])
    generation_config = prompt_dict.get("generation_config", {})
    contents = prompt_dict["contents"]

    def _extract_text() -> str:
        """Flatten the contents list into a single string."""
        if isinstance(contents, list) and contents:
            if isinstance(contents[0], dict) and "parts" in contents[0]:
                text_parts = [
                    part["text"]
                    for item in contents
                    for part in item.get("parts", [])
                    if isinstance(part, dict) and "text" in part
                ]
                return "\n\n".join(text_parts) or str(contents)
        return str(contents)

    def _generate():
        return model.generate_content(
            _extract_text(),
            generation_config=genai.types.GenerationConfig(
                temperature=generation_config.get("temperature", 1.0),
                candidate_count=generation_config.get("candidate_count", 1),
            ),
        )

    def _estimate_tokens() -> int:
        text = _extract_text()
        candidate_count = max(1, generation_config.get("candidate_count", 1))
        # Approximate conservative token usage using 4 characters per token,
        # then add a small output overhead for response tokens.
        return max(1, math.ceil(len(text) / 4)) * candidate_count + 100

    # --- Rate-limit throttle: enforce both request and token budgets ---
    async with _request_lock:
        global _last_request_time, _TOKEN_BUCKET, _LAST_TOKEN_REFILL

        now = time.monotonic()
        elapsed = now - _LAST_TOKEN_REFILL
        _TOKEN_BUCKET = min(
            float(_TOKENS_PER_MINUTE),
            _TOKEN_BUCKET + elapsed * (_TOKENS_PER_MINUTE / _REQUEST_WINDOW_SECONDS),
        )
        _LAST_TOKEN_REFILL = now

        estimated_tokens = _estimate_tokens()
        while True:
            now = time.monotonic()
            while _REQUEST_TIMESTAMPS and now - _REQUEST_TIMESTAMPS[0] >= _REQUEST_WINDOW_SECONDS:
                _REQUEST_TIMESTAMPS.popleft()

            request_count = len(_REQUEST_TIMESTAMPS)
            time_since_last_request = now - _last_request_time
            wait_for_request = max(0.0, _MIN_REQUEST_INTERVAL - time_since_last_request)
            wait_for_rpm = 0.0
            if request_count >= _REQUESTS_PER_MINUTE:
                oldest = _REQUEST_TIMESTAMPS[0]
                wait_for_rpm = max(0.0, _REQUEST_WINDOW_SECONDS - (now - oldest))

            token_refill_rate = _TOKENS_PER_MINUTE / _REQUEST_WINDOW_SECONDS
            if _TOKEN_BUCKET < estimated_tokens:
                wait_for_tokens = (estimated_tokens - _TOKEN_BUCKET) / token_refill_rate
            else:
                wait_for_tokens = 0.0

            wait = max(wait_for_request, wait_for_rpm, wait_for_tokens)
            if wait <= 0:
                break
            await asyncio.sleep(wait)

            now = time.monotonic()
            elapsed = now - _LAST_TOKEN_REFILL
            _TOKEN_BUCKET = min(
                float(_TOKENS_PER_MINUTE),
                _TOKEN_BUCKET + elapsed * token_refill_rate,
            )
            _LAST_TOKEN_REFILL = now

        _REQUEST_TIMESTAMPS.append(now)
        _TOKEN_BUCKET -= estimated_tokens
        _last_request_time = now

    return await asyncio.to_thread(_generate)


def result_into_answers(result: Any, prompt: Prompt) -> List[Answer]:
    """Convert Gemini response to Answer objects."""
    answers = []
    # Gemini returns candidates in result.candidates
    if hasattr(result, 'candidates') and result.candidates:
        for idx, candidate in enumerate(result.candidates):
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    text_content = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                else:
                    # Fallback if structure is different
                    text_content = str(candidate.content)
            else:
                text_content = str(candidate)
            answers.append(
                Answer(
                    prompt.attributes,
                    index=idx,
                    answer=text_content,
                )
            )
    return answers


async def process_and_store_prompt(
    parameters: Parameters,
    prompt: Prompt,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[Answer]:
    """Process a prompt and store the result.

    Uses a semaphore to cap concurrency and avoid RateLimitErrors.
    NotDoneException (logic retry) is separated from API-error retries so
    that transient 429/503 responses are handled entirely inside ask_gemini.
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    valid_answers = get_answers_by_prompt(prompt, filter_valid=True)
    _completion_prompt = copy.deepcopy(prompt.prompt)
    candidate_count = _completion_prompt["generation_config"].get("candidate_count", 1)
    _completion_prompt["generation_config"]["candidate_count"] = candidate_count - len(valid_answers)

    # Retry only for NotDoneException (i.e., not enough valid JSON answers).
    # API-level errors (429, 503, …) are retried inside ask_gemini itself.
    for attempt in tenacity.Retrying(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=2, min=5, max=30),
        retry=tenacity.retry_if_exception_type(NotDoneException),
        reraise=False,
    ):
        with attempt:
            if _completion_prompt["generation_config"]["candidate_count"] <= 0:
                break
            async with semaphore:
                result = await ask_gemini(_completion_prompt)
            store_gemini_completion(result, prompt.meta["path"])
            answers = result_into_answers(result, prompt)
            for answer in answers:
                completion_id = getattr(result, "response_id", str(id(result)))
                store_answer(answer, prompt.meta["path"], completion_id)
                if is_valid_answer(answer):
                    _completion_prompt["generation_config"]["candidate_count"] -= 1
                    answer.valid = True
                    valid_answers.append(answer)
            if _completion_prompt["generation_config"]["candidate_count"] > 0:
                raise NotDoneException("Not enough valid answers provided.")

    # NOTE: restricting the return to GEMINI_N elements
    return valid_answers[: config["GEMINI_N"]]


async def process_prompt_list(
    parameters: Parameters, prompts: List[Prompt]
) -> List[Answer]:
    """Process a list of prompts sequentially (or with bounded concurrency).

    PARALLEL_GEMINI_REQUESTS controls the semaphore width.  The default in
    config is 1 (fully sequential), which is the safest setting for the
    Gemini free tier.  Increase it only if you have a paid quota.
    """
    max_parallel = int(config.get("PARALLEL_GEMINI_REQUESTS", 1))
    semaphore = asyncio.Semaphore(max_parallel)
    tasks = []
    async with asyncio.TaskGroup() as tg:
        for prompt in prompts:
            tasks.append(
                tg.create_task(process_and_store_prompt(parameters, prompt, semaphore))
            )
    return [result for task in tasks for result in task.result()]


def extract_json(answer: Answer) -> Dict[str, Any]:
    """Extract the JSON formatted summary from an Answer retrieved from Gemini."""
    # checks whether there is a proper JSON structure in the response
    start_decision = answer.answer.rindex("{")
    end_decision = answer.answer.index("}", start_decision)
    # cleanup step
    raw_json = answer.answer[start_decision : end_decision + 1].replace("'", '"')
    return json.loads(raw_json)


def is_valid_answer(answer: Answer) -> bool:
    """Check whether an Answer is valid."""
    try:
        extract_json(answer)
    except:  # noqa: E722
        return False
    return True
