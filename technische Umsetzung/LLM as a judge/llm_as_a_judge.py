
import json
import pandas as pd
import os
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel

# ============================================================================
# KONFIGURATION
# ============================================================================

# Pfad relativ zum Skript-Verzeichnis
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "Dokumentation_Komplett_20251202_174107.json")
API_KEY = "sk-G1lcRvIZNmgTXZTlQ9AtHQ"
BASE_URL = "https://adesso-ai-hub.3asabc.de/v1"
MODEL_NAME = "gpt-4.1"

# Anzahl der Durchläufe
NUM_EPISODES = 100

# Temperatur für den Judge
TEMPERATURE = 0.0

# Output-Dateiname mit Temperatur
OUTPUT_CSV = os.path.join(SCRIPT_DIR, f"llm_evaluation_results_100_episodes_temp{TEMPERATURE}.csv")

# Set environment variables for DeepEval
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = BASE_URL

# Initialize custom GPTModel with adesso API and specified temperature
custom_model = GPTModel(model=MODEL_NAME, temperature=TEMPERATURE)

# ============================================================================
# 1. METRIKEN DEFINIEREN (nach deiner Umfrage-Struktur)
# ============================================================================

# INPUT + OUTPUT + CONTEXT - braucht alle drei für Kontext-Frage-Antwort-Beziehung
kontextverstaendnis = GEval(
    name="Kontextverständnis",
    model=custom_model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT
    ],
    evaluation_steps=[
        "Die Antwort erfasst die relevanten Informationen aus dem Kontext",
        "Die Antwort zeigt ein korrektes Verständnis des Kontexts",
        "Die Antwort nutzt relevante Informationen aus dem gegebenen Kontext zur Beantwortung der Frage (statt allgemeinem Wissen)"
    ]
)

# NUR OUTPUT - interne Logik der Antwort
kohaerenz = GEval(
    name="Kohärenz",
    model=custom_model,
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Die Aussagen in der Antwort widersprechen sich nicht",
        "Die Schlussfolgerungen in der Antwort ergeben sich logisch aus den Gründen/Prämissen",
        "Die Gedankenführung der Antwort ist klar und gut zu folgen"
    ]
)

# INPUT + OUTPUT - Verhältnis Frage zu Antwort
angemessenheit = GEval(
    name="Angemessenheit",
    model=custom_model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Die Detailtiefe der Antwort ist passend zur Fragestellung – nicht zu oberflächlich, nicht zu ausschweifend",
        "Die Antwort adressiert alle wesentlichen Aspekte der Frage – es fehlen keine wichtigen Informationen"
    ]
)

# INPUT + OUTPUT + CONTEXT - Gesamteindruck
plausibilitaet = GEval(
    name="Gesamtplausibilität",
    model=custom_model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT
    ],
    evaluation_steps=[
        "Wie plausibel ist die Antwort, basierend auf dem gegebenen Kontext?",
        "Nutze folgende Skala für deine Bewertung:",
        "1 = Unmöglich – Die Antwort widerspricht der Logik oder dem Kontext vollständig.",
        "2 = Technisch möglich – Theoretisch denkbar, aber extrem unwahrscheinlich.",
        "3 = Plausibel – Könnte zutreffen, ist aber nicht der Normalfall.",
        "4 = Wahrscheinlich – Klingt vernünftig und entspricht der Erwartung.",
        "5 = Sehr wahrscheinlich – Sehr überzeugend; so würde ein Experte antworten."
    ]
)

# ============================================================================
# 2. HILFSFUNKTIONEN
# ============================================================================

def convert_score_to_likert(deepeval_score):
    """
    Konvertiert DeepEval Score (0-1) zu Likert-Skala (1-5)
    
    0.0 → 1.0 (Unmöglich)
    0.25 → 2.0 (Technisch möglich)
    0.5 → 3.0 (Plausibel)
    0.75 → 4.0 (Wahrscheinlich)
    1.0 → 5.0 (Sehr wahrscheinlich)
    """
    return round(1 + (deepeval_score * 4), 2)


def load_data(filepath):
    """Lädt die JSON-Datei mit Fragen und Antworten"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_test_cases(data):
    """Erstellt DeepEval Test-Cases aus den JSON-Daten"""
    kontext = data["kontext"]
    test_cases = []
    
    for item in data["antworten"]:
        test_case = LLMTestCase(
            input=item["frage"],
            actual_output=item["antwort"],
            context=[kontext]
        )
        test_cases.append(test_case)
    
    return test_cases


# ============================================================================
# 3. HAUPTFUNKTION
# ============================================================================

def run_single_evaluation(data, test_cases, metrics, episode_nr):
    """Führt eine einzelne Evaluation durch und gibt die Ergebnisse zurück"""
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        row = {
            "Episode": episode_nr,
            "Frage_Nr": i,
            "Frage": test_case.input[:50] + "...",
        }
        for metric in metrics:
            # Reason-Generierung deaktivieren, falls möglich
            if hasattr(metric, 'generate_reason'):
                metric.generate_reason = False
            metric.measure(test_case)
            score_01 = metric.score if metric.score is not None else 0.0
            score_15 = convert_score_to_likert(score_01)
            row[f"{metric.name}_Score"] = score_15
        results.append(row)
    return results


def run_evaluation():
    """Führt die G-Eval Evaluation für alle Episoden durch und speichert Ergebnisse"""
    
    data = load_data(DATA_FILE)
    num_questions = len(data['antworten'])
    print(f"Gefunden: {num_questions} Fragen/Antworten")
    print(f"Starte {NUM_EPISODES} Episoden...\n")
    
    # Test-Cases erstellen
    test_cases = create_test_cases(data)
    
    # Metriken definieren
    metrics = [kontextverstaendnis, kohaerenz, angemessenheit, plausibilitaet]
    
    # Prüfe, wie viele Episoden schon in der CSV stehen
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        if 'Episode' in df_existing.columns:
            done_episodes = df_existing['Episode'].max()
        else:
            done_episodes = 0
        all_results = df_existing.to_dict(orient='records')
    else:
        done_episodes = 0
        all_results = []

    # Episoden durchlaufen, nur fehlende berechnen
    for episode in range(done_episodes + 1, NUM_EPISODES + 1):
        print(f"{'='*60}")
        print(f"EPISODE {episode}/{NUM_EPISODES}")
        print(f"{'='*60}")

        episode_results = run_single_evaluation(data, test_cases, metrics, episode)
        all_results.extend(episode_results)

        # Fortschritt für diese Episode anzeigen
        for result in episode_results:
            print(f"  Frage {result['Frage_Nr']}: "
                  f"Kontext={result['Kontextverständnis_Score']:.1f}, "
                  f"Kohärenz={result['Kohärenz_Score']:.1f}, "
                  f"Angem.={result['Angemessenheit_Score']:.1f}, "
                  f"Plaus.={result['Gesamtplausibilität_Score']:.1f}")

        # Zwischenspeichern nach jeder Episode (Sicherheit bei Abbruch)
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"  -> Zwischengespeichert ({len(all_results)} Einträge)\n")
    
    # Finale Zusammenfassung
    print("\n" + "=" * 60)
    print("FINALE ZUSAMMENFASSUNG")
    print("=" * 60)
    
    df = pd.DataFrame(all_results)
    score_columns = [col for col in df.columns if col.endswith("_Score")]
    
    print(f"\nGesamtanzahl Evaluierungen: {len(df)}")
    print(f"({NUM_EPISODES} Episoden × {num_questions} Fragen)\n")
    
    print("Durchschnittliche Scores über alle Episoden:")
    for col in score_columns:
        mean_score = df[col].mean()
        std_score = df[col].std()
        print(f"  {col}: Ø {mean_score:.2f} (±{std_score:.2f})")
    
    print(f"\n✓ Evaluation abgeschlossen!")
    print(f"✓ Ergebnisse gespeichert in: {OUTPUT_CSV}")
    
    return df


# ============================================================================
# 4. AUSFÜHRUNG
# ============================================================================

if __name__ == "__main__":
    results_df = run_evaluation()
    print(results_df)