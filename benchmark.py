# benchmark.py

import time
import openai
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
# We might use accelerate or not, depending on your preference:
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Local modules
from prompts import CHAT_CONVERSATIONS, QA_PROMPTS

###############################################################################
# 1. Configuration
###############################################################################

# GPT-4 API Key (replace with your actual key)
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly: "sk-xxx..."

# Model names on Hugging Face
LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"       # Example only; may require license acceptance
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Example; check Hugging Face for exact name

# If your GPU VRAM is very limited, set this to True to load models in 8-bit
LOAD_IN_8BIT = True

# Max new tokens for generation
MAX_NEW_TOKENS = 128

# Generation config
GENERATION_KWARGS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "max_new_tokens": MAX_NEW_TOKENS,
}

###############################################################################
# 2. Utility Functions
###############################################################################

def load_model_and_tokenizer(model_name, load_in_8bit=False):
    """
    Loads a Hugging Face model and tokenizer.
    """
    print(f"Loading model: {model_name} (8-bit={load_in_8bit})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",         # automatically map layers to GPU
            load_in_8bit=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return tokenizer, model

def measure_latency(func):
    """
    Decorator to measure the time a function call takes.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time
        return result, latency
    return wrapper

@measure_latency
def generate_response_hf(model, tokenizer, prompt):
    """
    Generate a response using a Hugging Face model (LLaMA or Mistral).
    Returns the model output text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, **GENERATION_KWARGS)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@measure_latency
def generate_response_gpt4(system_prompt, user_prompt):
    """
    Generate a response using OpenAI GPT-4 API in ChatCompletion mode.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        n=1
    )
    return response.choices[0].message["content"]

###############################################################################
# 3. Chat Benchmarking
###############################################################################

def run_chat_benchmark(model, tokenizer, model_name):
    """
    Test the model on multi-turn conversations.
    """
    print(f"\n{'='*40}")
    print(f" Chat Benchmarking: {model_name}")
    print(f"{'='*40}")

    for i, conversation in enumerate(CHAT_CONVERSATIONS, start=1):
        print(f"\n--- Conversation {i} ---")
        conversation_history = ""
        for turn_index, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                # Add user input to conversation history
                conversation_history += f"User: {content}\n"
                # We won't measure latency on user input alone
                print(f"[User] {content}")
            else:
                # Generate assistant response
                full_prompt = (
                    "A conversation between a user and an AI assistant. "
                    "The assistant should be helpful, polite, and accurate.\n\n"
                    + conversation_history
                    + "Assistant:"
                )
                response, latency = generate_response_hf(model, tokenizer, full_prompt)
                # Clean up the response by removing any leftover prompt text
                # We'll just print the entire response as is for now
                print(f"[Assistant] {response}")
                print(f"(Latency: {latency:.2f} seconds)")
                
                # Update conversation history
                conversation_history += f"Assistant: {response}\n"

def run_chat_benchmark_gpt4():
    """
    Test GPT-4 on multi-turn conversations (separate test).
    """
    print(f"\n{'='*40}")
    print(" Chat Benchmarking: GPT-4")
    print(f"{'='*40}")
    
    for i, conversation in enumerate(CHAT_CONVERSATIONS, start=1):
        print(f"\n--- Conversation {i} ---")
        conversation_history = ""
        for turn_index, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                # Just show user input
                print(f"[User] {content}")
                conversation_history += f"User: {content}\n"
            else:
                # Generate using GPT-4
                system_prompt = (
                    "You are a helpful AI assistant. "
                    "Provide accurate and concise answers."
                )
                # We'll pass the entire conversation so far as context
                user_prompt = conversation_history + "Assistant:"
                response, latency = generate_response_gpt4(system_prompt, user_prompt)
                print(f"[Assistant] {response}")
                print(f"(Latency: {latency:.2f} seconds)")
                # Append to conversation history
                conversation_history += f"Assistant: {response}\n"

###############################################################################
# 4. QA Benchmarking
###############################################################################

def run_qa_benchmark(model, tokenizer, model_name):
    """
    Test the model on single-turn QA prompts.
    """
    print(f"\n{'='*40}")
    print(f" QA Benchmarking: {model_name}")
    print(f"{'='*40}")

    for i, qa_item in enumerate(QA_PROMPTS, start=1):
        question = qa_item["question"]
        reference_answer = qa_item["reference_answer"]

        prompt = (
            "You are a helpful AI assistant. "
            "Answer the following question concisely and accurately:\n\n"
            f"Question: {question}\nAnswer:"
        )
        response, latency = generate_response_hf(model, tokenizer, prompt)
        
        print(f"\nQ{i}: {question}")
        print(f"Model Answer: {response}")
        print(f"Reference:   {reference_answer}")
        print(f"(Latency: {latency:.2f} seconds)")

def run_qa_benchmark_gpt4():
    """
    Test GPT-4 on single-turn QA prompts.
    """
    print(f"\n{'='*40}")
    print(" QA Benchmarking: GPT-4")
    print(f"{'='*40}")
    
    for i, qa_item in enumerate(QA_PROMPTS, start=1):
        question = qa_item["question"]
        reference_answer = qa_item["reference_answer"]

        system_prompt = "You are a helpful AI assistant. Provide concise, accurate answers."
        user_prompt = f"Question: {question}\nAnswer:"
        response, latency = generate_response_gpt4(system_prompt, user_prompt)
        
        print(f"\nQ{i}: {question}")
        print(f"Model Answer: {response}")
        print(f"Reference:   {reference_answer}")
        print(f"(Latency: {latency:.2f} seconds)")

###############################################################################
# 5. Main
###############################################################################

def main():
    # 1. Load LLaMA
    llama_tokenizer, llama_model = load_model_and_tokenizer(
        model_name=LLAMA_MODEL_NAME,
        load_in_8bit=LOAD_IN_8BIT
    )

    # 2. Load Mistral
    mistral_tokenizer, mistral_model = load_model_and_tokenizer(
        model_name=MISTRAL_MODEL_NAME,
        load_in_8bit=LOAD_IN_8BIT
    )

    # 3. Run Chat Benchmark for LLaMA
    run_chat_benchmark(llama_model, llama_tokenizer, "LLaMA")

    # 4. Run Chat Benchmark for Mistral
    run_chat_benchmark(mistral_model, mistral_tokenizer, "Mistral")

    # 5. Run QA Benchmark for LLaMA
    run_qa_benchmark(llama_model, llama_tokenizer, "LLaMA")

    # 6. Run QA Benchmark for Mistral
    run_qa_benchmark(mistral_model, mistral_tokenizer, "Mistral")

    # 7. GPT-4 Chat Benchmark (separate)
    run_chat_benchmark_gpt4()

    # 8. GPT-4 QA Benchmark (separate)
    run_qa_benchmark_gpt4()


if __name__ == "__main__":
    main()
