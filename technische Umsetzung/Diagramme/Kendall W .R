library(irr)
library(ggplot2)
library(reshape2)

# Daten einlesen (Pfad angepasst für Start aus Bachelorarbeit)
df <- read.csv("technische Umsetzung/LLM as a judge/llm_evaluation_results_100_episodes.csv")

# Kendalls W für jede Dimension (Metrik) berechnen
metriken <- c("Kontextverständnis_Score","Kohärenz_Score","Angemessenheit_Score","Gesamtplausibilität_Score")
kendall_w_dim <- sapply(metriken, function(metric) {
  # Matrix: Zeilen = Fragen (Items), Spalten = Episoden (Rater)
  mat <- dcast(df, Frage_Nr ~ Episode, value.var = metric)
  mat <- mat[,-1]  # Frage_Nr-Spalte entfernen
  kendall(mat)$value
})


# Ausgabe der Werte
print(kendall_w_dim)

