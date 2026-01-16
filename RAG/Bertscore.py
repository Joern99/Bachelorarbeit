import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv
from openai import OpenAI

# .env laden
load_dotenv()

# Konfiguration aus .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","sk-GsX9Ax0dzOKS8gAqJaPKkQ")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL","https://adesso-ai-hub.3asabc.de/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-oss-120b-sovereign")

# OpenAI-Client im gewünschten Stil
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=60)

def load_full_context_from_json(path: str = "Kontext.json") -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    parts = []
    for i, item in enumerate(data.get("context", []), 1):
        src = item.get("source", "")
        parts.append(f"=== Abschnitt {i} ===\n{src}")
    ctx = "\n\n".join(parts)
    # Vorsichtiges Limit, damit Frage und Abschluss sicher im Fenster bleiben
    MAX_CHARS = 200000
    return ctx[:MAX_CHARS]

def load_fragen(path: str = "Fragen.json") -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # Markdown-/Tabellen-Symbole grob entfernen und Whitespace normalisieren
    s = re.sub(r"[*_`#>~-]+", " ", s)
    s = re.sub(r"\|", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def build_prompt(context: str, question: str) -> str:
    return (
        "Beantworte die Frage ausschließlich anhand des folgenden Kontexts.\n"
        "Antworte als Fließtext, ohne Listen, Tabellen oder Markdown-Formatierung.\n"
        "Wenn die Antwort im Kontext nicht eindeutig steht, sage das klar.\n\n"
        f"FRAGE:\n{question}\n\n"
        f"KONTEXT BEGINN\n{context}\nKONTEXT ENDE\n"
    )

def ask_llm_full_context(context: str, question: str, max_tokens: int = 900, temperature: float = 0.0) -> Tuple[str, Dict[str, Any]]:
    """
    Synchroner Call ohne Streaming. Wartet, bis das LLM antwortet.
    Gibt den Antworttext und Metadaten zurück.
    """
    prompt = build_prompt(context, question)
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    for attempt in range(1, 3 + 1):
        try:
            response = client.chat.completions.create(**payload)
            text = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            prompt_toks = usage.prompt_tokens if usage else None
            comp_toks = usage.completion_tokens if usage else None
            meta = {
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "model_used": OPENAI_MODEL,
                "error": None
            }
            return text, meta
        except Exception as e:
            err_msg = str(e)
            print(f"[WARN] LLM-Call fehlgeschlagen (Versuch {attempt}/3): {err_msg}")
            if attempt == 3:
                return "", {"prompt_tokens": None, "completion_tokens": None, "model_used": OPENAI_MODEL, "error": err_msg}
            time.sleep(1.5 * attempt)

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY fehlt in .env")
    if not OPENAI_BASE_URL:
        raise RuntimeError("OPENAI_BASE_URL fehlt in .env")

    context = load_full_context_from_json("Kontext.json")
    items = load_fragen("Fragen.json")

    results_for_json = []
    questions, references, candidates = [], [], []

    for idx, item in enumerate(items, 1):
        q = normalize_text(item.get("Frage"))
        ref = normalize_text(item.get("Musterantwort"))

        if not q or not ref:
            results_for_json.append({"Nr": idx, "Frage": q or "", "Musterantwort": ref or "", "Antwort": "", "meta": {"error": "missing q/ref"}})
            continue

        ans_raw, meta = ask_llm_full_context(context, q, max_tokens=900, temperature=0.0)
        ans = normalize_text(ans_raw)

        results_for_json.append({
            "Nr": idx,
            "Frage": q,
            "Musterantwort": ref,
            "Antwort": ans,
            "meta": meta
        })

        # Für BERTScore nur nicht-leere Antworten verwenden
        if ans:
            questions.append(q)
            references.append(ref)
            candidates.append(ans)

    # JSON speichern und ausgeben
    json_str = json.dumps(results_for_json, ensure_ascii=False, indent=2)
    Path("LLM_Antworten.json").write_text(json_str, encoding="utf-8")
    print("\nLLM-Antworten als JSON gespeichert in LLM_Antworten.json")

    print(f"\nAnzahl ausgewerteter Paare (nach Filter, nicht-leer): {len(candidates)}")
    if not candidates:
        print("[INFO] Keine nicht-leeren Kandidaten vorhanden. Kontextlimit, Modellname und max_tokens prüfen.")
        return

    # BERTScore mit multilingualem Modell
    from bert_score import score
    P, R, F1 = score(
        candidates,
        references,
        model_type="xlm-roberta-large",
        rescale_with_baseline=True,
        lang="de"  # Sprache hinzufügen (de für Deutsch)
    )

    for i, (q, cand, ref, p, r, f1) in enumerate(zip(questions, candidates, references, P, R, F1), 1):
        print("\n==============================")
        print(f"[{i}] Frage: {q}")
        print("Candidate:", cand)
        print("Reference:", ref)
        print("BERTScore -> P:{:.4f} R:{:.4f} F1:{:.4f}".format(p.item(), r.item(), f1.item()))

    import numpy as np
    print("\nDurchschnitt:")
    print("P_mean:", float(np.mean(P.numpy())))
    print("R_mean:", float(np.mean(R.numpy())))
    print("F1_mean:", float(np.mean(F1.numpy())))

if __name__ == "__main__":
    main()
