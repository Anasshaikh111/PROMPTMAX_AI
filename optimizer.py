from transformers import pipeline
import torch
import re

# 🧠 Load facebook/bart-large (optimized usage)
def load_generator():
    print("🔹 Loading facebook/bart-large (optimized mode)...")
    return pipeline(
        "text2text-generation",
        model="facebook/bart-large",
        device=0 if torch.cuda.is_available() else -1
    )

generator = load_generator()


def clean_text(text):
    """Clean and normalize input text."""
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def optimize_prompt(user_prompt):
    """Optimize a given user prompt for AI models using bart-large."""
    user_prompt = clean_text(user_prompt)
    if not user_prompt:
        return "Please enter a valid prompt."

    # 🧩 Remove excessive repetition
    words = user_prompt.split()
    if len(set(words)) < len(words) / 2:
        user_prompt = " ".join(sorted(set(words), key=words.index))

    # ⚙️ Instruction pattern that works best for non-instruction models like BART
    instruction = (
        f"Input: {user_prompt}\n\n"
        
    )

    try:
        result = generator(
            instruction,
            max_length=180,
            temperature=0.9,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )

        optimized = result[0]['generated_text'].strip()

        # 🧠 Clean up any leftover tokens or repeated context
        optimized = optimized.replace("Input:", "").replace("Output:", "").strip()

        # ✨ If model just copied the input, add small fallback rewording
        if optimized.lower() == user_prompt.lower() or len(optimized.split()) < 4:
            optimized = f"A clear and detailed rewrite of your prompt: {user_prompt}"

        return optimized

    except Exception as e:
        return f"⚠️ Optimization failed: {str(e)}"

