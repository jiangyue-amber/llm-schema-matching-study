import asyncio
import copy
import json
import logging
import torch
import accelerate
from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import config
from .errors import NotDoneException
from .models import Answer, Parameters, Prompt
from .storage import store_answer, store_huggingface_completion, get_answers_by_prompt
from .prompt_postprocessing import extract_json

# Global variable to cache the model and tokenizer
_MODEL = None
_TOKENIZER = None

def load_model_if_needed():
    """Loads the model and tokenizer if they haven't been loaded yet."""
    global _MODEL, _TOKENIZER, _PIPELINE
    if _MODEL is not None:
        return _MODEL

    model_name = config["HUGGINGFACE_MODEL"]
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Loading model {model_name} on {device}...")
    
    # Load tokenizer with padding side set to left (better for batch generation)
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _TOKENIZER.padding_side = "left"
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
    
    # Load model with GPU memory optimizations
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,  # Use float16 for 2x GPU memory efficiency
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if device == "cuda" else None,  # Faster attention
    )
    
    # Disable gradient computation for inference only
    _MODEL.eval()
    torch.set_grad_enabled(False)
    
    print("Model loaded successfully.")
    return _MODEL

def send_prompts(parameters: Parameters, prompts: List[Prompt]) -> List[Answer]:
    """Generate a result from parameters and prompts."""
    # Since we are running locally, asyncio.run might block, but we keep the interface.
    return asyncio.run(process_prompt_list(parameters, prompts))


def run_inference_synchronously(prompt: Prompt, num_sequences: int) -> List[Answer]:
    """Runs the inference synchronously using direct model generation."""
    model = load_model_if_needed()
    
    messages = prompt.prompt["messages"]
    generation_config = prompt.prompt.get("generation_config", {})
    
    # Apply chat template
    try:
        formatted_prompt = _TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        formatted_prompt = _TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Tokenize and move to GPU - this is more efficient than using pipeline
    inputs = _TOKENIZER(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)
    
    # Generate with GPU optimization
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            num_return_sequences=num_sequences,
            do_sample=True,
            temperature=generation_config.get("temperature", config["HUGGINGFACE_TEMPERATURE"]),
            top_p=0.9,  # Added nucleus sampling for better GPU utilization
            pad_token_id=_TOKENIZER.eos_token_id,
        )
    
    # Decode outputs
    generated_texts = _TOKENIZER.batch_decode(outputs, skip_special_tokens=True)
    
    # Extract only the new tokens (remove input prompt)
    input_len = inputs["input_ids"].shape[1]
    answers = []
    for idx in range(len(generated_texts)):
        # Re-tokenize to get only new part
        full_text = generated_texts[idx]
        generated_text = _TOKENIZER.decode(outputs[idx][input_len:], skip_special_tokens=True)
        
        answers.append(
            Answer(
                prompt.attributes,
                index=idx,
                answer=generated_text,
            )
        )
    
    # Store results
    try:
        completion_id = store_huggingface_completion(
            [{"generated_text": ans.answer} for ans in answers], 
            prompt.meta["path"]
        )
        
        for answer in answers:
            if is_valid_answer(answer):
                answer.valid = True
            store_answer(answer, prompt.meta["path"], completion_id)
    except Exception as e:
        print(f"Error storing results: {e}")
            
    return answers

def run_batch_inference(prompts: List[Prompt]) -> List[List[Answer]]:
    """Runs batch inference for multiple prompts with GPU-optimized batching."""
    num_sequences = config["HUGGINGFACE_N"]
    model = load_model_if_needed()
    
    # 1. Format all prompts efficiently
    formatted_prompts = []
    for prompt in prompts:
        messages = prompt.prompt["messages"]
        try:
            formatted = _TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            formatted = _TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        formatted_prompts.append(formatted)
    
    # 2. Tokenize with padding for efficient GPU batching
    # NOTE: This tokenizes the batch on GPU for better performance
    inputs = _TOKENIZER(
        formatted_prompts,
        return_tensors="pt",
        padding=True,  # Pad to longest sequence in batch
        truncation=True,
        max_length=4096,
    ).to(model.device)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Track input lengths for each sequence (needed to extract only new tokens)
    input_lengths = attention_mask.sum(dim=1)
    
    # 3. Generate with optimizations
    generation_config = prompts[0].prompt.get("generation_config", {})
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
            num_return_sequences=num_sequences,
            do_sample=True,
            temperature=generation_config.get("temperature", config["HUGGINGFACE_TEMPERATURE"]),
            top_p=0.9,  # Nucleus sampling
            top_k=50,   # Top-K sampling for better quality
            pad_token_id=_TOKENIZER.eos_token_id,
            eos_token_id=_TOKENIZER.eos_token_id,
            # These help GPU: use cache, no repeat penalty computation
            output_scores=False,  # Don't compute scores to save GPU memory
            use_cache=True,  # Use KV cache for faster generation
        )
    
    # 4. Decode and extract only new tokens
    # Due to num_return_sequences, outputs shape is (batch_size * num_sequences, max_len)
    all_generated_texts = _TOKENIZER.batch_decode(outputs, skip_special_tokens=True)
    
    # 5. Reorganize and store results
    answers_batch = []
    for prompt_idx, prompt in enumerate(prompts):
        answers = []
        for seq_idx in range(num_sequences):
            output_idx = prompt_idx * num_sequences + seq_idx
            
            # Extract only the new tokens
            new_tokens_start = input_lengths[prompt_idx].item()
            generated_text = _TOKENIZER.decode(
                outputs[output_idx][new_tokens_start:], 
                skip_special_tokens=True
            )
            
            answer = Answer(
                prompt.attributes,
                index=seq_idx,
                answer=generated_text,
            )
            answers.append(answer)
        
        answers_batch.append(answers)
    
    # 6. Store results in database
    for prompt_idx, prompt in enumerate(prompts):
        output_list = [{"generated_text": ans.answer} for ans in answers_batch[prompt_idx]]
        try:
            completion_id = store_huggingface_completion(output_list, prompt.meta["path"])
            for answer in answers_batch[prompt_idx]:
                if is_valid_answer(answer):
                    answer.valid = True
                store_answer(answer, prompt.meta["path"], completion_id)
        except Exception as e:
            print(f"Error storing batch results for prompt {prompt_idx}: {e}")
    
    return answers_batch

def is_valid_answer(answer: Answer) -> bool:
    """Check whether an Answer is valid."""
    try:
        extract_json(answer)
    except:
        return False
    return True

async def process_prompt_list(
    parameters: Parameters, prompts: List[Prompt]
) -> List[Answer]:
    """Process a list of prompts with aggressive concurrent batching to maximize GPU utilization.
    
    Config parameters:
    - HUGGINGFACE_MAX_CONCURRENT: max concurrent GPU batch operations (default 16, increase for more utilization)
    - HUGGINGFACE_BATCH_SIZE: prompts per batch (default 64, increase for larger batches)
    
    Strategy: Process many batches concurrently to keep GPU fully saturated.
    Higher values = higher GPU utilization but more VRAM usage.
    """
    # Collect all prompts that need processing
    prompts_to_process = []
    for prompt in prompts:
        valid_answers = get_answers_by_prompt(prompt, filter_valid=True)
        needed = config["HUGGINGFACE_N"] - len(valid_answers)
        if needed > 0:
            prompts_to_process.append(prompt)
    
    if not prompts_to_process:
        return []
    
    # Use larger semaphore to allow more concurrent GPU operations
    # This is the KEY to better GPU utilization!
    max_concurrent = config.get("HUGGINGFACE_MAX_CONCURRENT", 16)  # Increased from default
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Split into chunks for concurrent batch inference
    # Larger batches = better GPU utilization (if VRAM allows)
    chunk_size = config.get("HUGGINGFACE_BATCH_SIZE", 64)  # Increased from default
    chunks = [
        prompts_to_process[i:i + chunk_size] 
        for i in range(0, len(prompts_to_process), chunk_size)
    ]
    
    print(f"Processing {len(prompts_to_process)} prompts in {len(chunks)} chunks of size {chunk_size}")
    print(f"Max concurrent batch operations: {max_concurrent}")
    
    async def process_chunk(chunk: List[Prompt]) -> List[List[Answer]]:
        """Process a single chunk of prompts with semaphore-gated concurrency."""
        async with semaphore:
            return await asyncio.to_thread(run_batch_inference, chunk)
    
    # Process all chunks concurrently
    # The semaphore ensures we don't overload GPU memory
    chunk_results = await asyncio.gather(
        *[process_chunk(chunk) for chunk in chunks],
        return_exceptions=False
    )
    
    # Flatten and return all answers
    all_answers = []
    for chunk_result in chunk_results:
        for sublist in chunk_result:
            all_answers.extend(sublist)
    
    return all_answers
