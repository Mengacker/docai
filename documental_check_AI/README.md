## 1. Informazioni generali


**Obiettivo del prodotto**
Automatizzare il confronto mensile dei PDF delle campagne marketing (condizioni economiche, CGC, schede sintetiche e schede di ripielogo) per evidenziare le differenze tra i documenti con la campagna precedente.

**Destinatari del prodotto**
Marketing

**Destinatari del documento**
Data scientist, data engineer, business analyst, management, marketing stakeholder

**Data inizio progetto** 10 giu 2025
**Go‑live MVP** inizio luglio 2025

---

## 2. Introduzione

Soluzione AI‐assisted che estrae, confronta e riassume automaticamente le differenze tra le parti contrattuali dei PDF delle campagne. L’output viene condiviso come file Excel affinché il team marketing verifichi rapidamente eventuali errori prima della pubblicazione ufficiale.

### 2.1 Contesto e obiettivo del progetto

Ogni campagna commodity (luce, gas, dual) genera 6‑10 PDF per segmento (residenziale, micro‑business, adotta‑un‑pannello). Il team marketing confronta manualmente ogni mese i nuovi PDF con quelli del mese precedente, operazione ripetitiva e time‑critical.
Obiettivo: ridurre del tempo manuale e diminuire il rischio di errori di contenuto.

### 2.2 Valore e metriche

- **Tempo risparmiato:** da ~2 giorni persona/campagna a circa 3 ore
- **Accuratezza confronto:** >95 % di rilevazione differenze testuali
- **Copertura documenti:** 100 % dei PDF di campagna
- **Adoption:** ≥2 utenti marketing attivi al mese (Alessandra, Marta)
  (Valori da validare in fase UAT.)

### 2.3 Vincoli e limitazioni

- Deadline MVP: 11 lug 2025 (prima campagna con nuovo layout)
- Cost constraints ⇒ evitato Azure Document Intelligence a pagamento
- Accesso limitato Databricks per utenti business (upload manuale su volume)

---

## 3. Descrizione del prodotto

### 3.1 Dati

| Fonte               | Tipo                            | Formato                                        | Volumetria    | Note                         |
| ------------------- | ------------------------------- | ---------------------------------------------- | ------------- | ---------------------------- |
| SharePoint ➜ Volume | Documenti PDF (CE, SR, ST, CGC) | PDF, ITA                                       | ~60 file/mese | Upload manuale da Marketing  |
| Storage DBFS        | Output CSV                       | testo estratto, flag identico/diverso, summary | ~4 file/mese   | Un file per tipo documento|

### 3.2 Architettura

1. **Ingest**: cartella campagna caricata su volume Databricks (o mount SharePoint futuro)
2. **Workflow Databricks** (CI/CD Azure DevOps) lancia notebook triggered da caricamneto file campagna su volume
3. **Parser PDF**: `pdfplumber` + logica custom per filtrare tabelle
4. **LLM Modules** (Extraction → Comparison → Summary) con GPT‑4o‑turbo; Signature prompt per controllo schema input/output
5. **Prompt optimization**: libreria DSPy in test phase
6. **Persistenza** risultati su Delta Lake

> **Architettura di sistema:**

```
SharePoint/Volume → PDF Ingestion → PDF Parser → LLM Pipeline → Excel Output
                                         ↓              ↓
                                    Delta Lake    MLflow Tracking
```

**Componenti tecnici:**

| Componente | Tecnologia | Versione | Scopo |
|------------|------------|----------|-------|
| PDF Parser | pdfplumber | v0.11 | Estrazione testo + filtraggio tabelle (regex/coordinate) |
| LLM Backbone | GPT-4o-turbo (OpenAI) | API latest | Context window 128k token, schema-prompt |
| Orchestrazione | Databricks Workflows | - | Scheduling e trigger automatici |
| Storage | Delta Lake / DBFS | - | Persistenza risultati e storico |
| Optimization | DSPy | latest | Prompt optimization su training-set |
| Tracing | MLflow | planned | Versioning prompt, Modello, metriche qualità e monitoring |
| CI/CD | Azure DevOps | - | Deploy automatico notebook e workflow |

**Moduli LLM (pipeline a 3 stadi):**

1. **Extraction Module:**
   - Estrae parti contrattuali strutturate da PDF
   - Output: JSON schema con sezioni rilevanti
   - Prompt: signature-based con esempi few-shot

2. **Comparison Module:** (non LLM )
   - Confronto binario documento corrente vs precedente
   - Output: match/no-match per sezione + dettaglio differenze
   - Logic: semantic similarity + exact matching

3. **Summary Module:**
   - Sintesi differenze in linguaggio naturale italiano
   - Output: testo human-readable per report Excel
   - Style: conciso

**Metriche e monitoring:**

| Metrica | Target | Misurazione |
|---------|--------|-------------|
| % paragrafi estratti correttamente | >95% | Sample validation manuale |
| Match rate confronti | >95% accuracy | Confronto con ground truth UAT |
| Tempo elaborazione per campagna | < 3 ore | End-to-end pipeline duration |

**Alerting:**

- Email automatica su job failure
- Notifica su anomalie qualità output (es. troppi errori parsing)
- Dashboard monitoraggio in Databricks

**Manutenzione:**

- Retraining prompt: mensile con feedback utenti
- Aggiornamento parser: on-demand quando cambiano template PDF
- Review performance LLM: trimestrale
- Backup storico risultati: automatico


### 3.3 Algoritmi e modellistica

- **PDF Parsing**: `pdfplumber` v0.11; eliminazione tabelle via regex/coordinate.
- **LLM Backbone**: GPT‑4o (OpenAI, 128k context) con schema‑prompt; planned upgrade a GPT‑4o Vision per PDF immagine.
- **Moduli**: Extraction, Comparison (binary match), Summary (Italian natural language).
- **Optimization**: DSPy optimizers su training‑set contrattuale.
- **Tracing**: integrazione MLflow (planned) per versione prompt/metriche.

---

## 4. Descrizione output prodotto

### 4.1 Servito

- **File Excel** con:
  - Tabella documenti vs mese precedente (flag "Uguale / Diverso")


### 4.2 Monitoraggio e manutenzione

- **Job Databricks**: log successo/fallimento, alert email su errori
- **Metriche qualità**: % paragrafi estratti, match rate, token count
- **Retraining prompt**: mensile con feedback utenti
- **Manutenzione**: aggiornamento layout parser quando cambiano i template PDF.
