from openai import OpenAI
import json
from pathlib import Path
from datetime import datetime

# TODO: Setze hier deinen API-Key ein
API_KEY = "sk-GsX9Ax0dzOKS8gAqJaPKkQ"  # Hier deinen vollständigen API-Key eintragen

# Initialize the client
client = OpenAI(
    api_key=API_KEY,
    base_url="https://adesso-ai-hub.3asabc.de/v1"
)

def load_questions(file_path):
    """Load questions from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('fragen', [])

def load_context(file_path):
    """Load context from text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def query_llm(question, context):
    """Query the LLM with a question and context"""
    prompt_template = """Kontext:
{context}

Frage:
{question}

Antworte kurz basierend auf den Informationen im obigen Kontext."""
    
    prompt = prompt_template.format(context=context, question=question)
    
    max_tokens = 10000
    temperature = 0.7
    
    response = client.chat.completions.create(
        model="gpt-oss-120b-sovereign",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.choices[0].message.content, prompt_template, max_tokens, temperature

def save_documentation(current_dir, questions, context, prompt_template, results, max_tokens, temperature):
    """Save complete documentation with timestamp versioning"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results to JSON file
    output_file = current_dir / "LLM_Antworten.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"antworten": results}, f, ensure_ascii=False, indent=2)
    
    # Save complete documentation with timestamp
    documentation_file = current_dir / f"Dokumentation_Komplett_{timestamp}.json"
    with open(documentation_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "fragen": questions,
            "kontext": context,
            "prompt": prompt_template,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "antworten": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Alle Antworten wurden in '{output_file.name}' gespeichert.")
    print(f"✓ Vollständige Dokumentation wurde in '{documentation_file.name}' gespeichert.")

def main():
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Load questions and context
    questions_file = current_dir / "Fragen neu"
    context_file = current_dir / "Kontext.txt"
    
    questions = load_questions(questions_file)
    context = load_context(context_file)
    
    print(f"Kontext geladen: {len(context)} Zeichen")
    print(f"Anzahl Fragen: {len(questions)}\n")
    print("=" * 80)
    
    # Process each question
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\nFrage {i}/{len(questions)}:")
        print(f"{question}")
        print("-" * 80)
        
        # Query the LLM
        answer, prompt_template, max_tokens, temperature = query_llm(question, context)
        print(f"Antwort:\n{answer}")
        print("=" * 80)
        
        # Store result
        results.append({
            "frage": question,
            "antwort": answer
        })
    
    # Save all documentation
    save_documentation(current_dir, questions, context, prompt_template, results, max_tokens, temperature)

if __name__ == "__main__":
    main()
