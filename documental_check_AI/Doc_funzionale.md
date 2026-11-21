**VERSIONE DOCUMENTO 1 DEL 27/01/2026**

**LEGENDA**

- 🔴 indica un paragrafo obbligatorio
- 🏁 indica un paragrafo da redigere in fase iniziale (origination)
- 🚧 indica un paragrafo da redigere in fase di esercizio (execution)
- 🆙 indica un paragrafo da redigere alla consegna ed aggiornare successivamente (delivery & follow-up)

[[_TOC_]]

<div style="page-break-after: always;"></div>

# 0. Scheda use case 🏁🚧🆙

**Team di sviluppo:** Red Hot Data Potters (RHDP)

**Obiettivo del prodotto:**
Automatizzare il confronto mensile dei PDF delle campagne marketing (condizioni economiche, CGC, schede sintetiche, schede di riepilogo e proposta di contratto) per evidenziare le differenze tra i documenti della campagna corrente rispetto alla campagna precedente.

**Destinatari del prodotto:**
Marketing Document Check Team (Alessandra Magri, Marta Paolini + risorse esterne)

**Destinatari del documento:**
Data scientist, data engineer, business analyst, management, marketing stakeholder

**Data inizio progetto:** 10 giugno 2025
**Go-live MVP:** luglio 2025

<div style="page-break-after: always;"></div>

# 1 Introduzione <a name="introduzione"></a> 🔴🏁🚧

## 1.1 Descrizione del prodotto 🔴🏁🚧

Soluzione AI-assisted che estrae, confronta e riassume automaticamente le differenze tra le parti contrattuali dei PDF delle campagne marketing. L'output viene condiviso come file Excel affinché il team marketing verifichi rapidamente eventuali errori prima della pubblicazione ufficiale.

## 1.2 Contesto e obiettivo del progetto 🔴🏁

Ogni campagna commodity (luce, gas, dual) genera 6-10 PDF per segmento (residenziale, micro-business, adotta-un-pannello). Il team marketing confronta manualmente ogni mese i nuovi PDF con quelli del mese precedente, operazione ripetitiva e time-critical.

**Obiettivo:** ridurre il tempo manuale e diminuire il rischio di errori di contenuto.

## 1.3 Vincoli e limitazioni 🔴🏁

- Deadline MVP: luglio 2025 (prima campagna con nuovo layout)
- Cost constraints: evitato Azure Document Intelligence a pagamento
- Accesso limitato Databricks per utenti business (upload manuale su volume)

<div style="page-break-after: always;"></div>

# 2 Descrizione del Servizio <a name="descrizioneservizio"></a> 🔴🏁🚧

## 2.1 Tecnologie e architettura 🔴🏁🚧

**Componenti tecnici:**

| Componente | Tecnologia | Versione | Scopo |
|------------|------------|----------|-------|
| PDF Parser | pdfplumber | v0.11 | Estrazione testo + filtraggio tabelle (regex/coordinate) |
| LLM Backbone | GPT-4o-turbo (OpenAI) | API latest | Context window 128k token, schema-prompt |
| Orchestrazione | Databricks Workflows | - | Scheduling e trigger automatici |
| Storage | Delta Lake / Unity Catalog | - | Persistenza risultati e storico |
| Optimization | DSPy | latest | Prompt optimization su training-set |
| Tracing | MLflow | - | Versioning prompt, modello, metriche qualità e monitoring |
| CI/CD | Azure DevOps | - | Deploy automatico notebook e workflow |

**Moduli LLM (pipeline a 3 stadi):**

1. **Extraction Module:** Estrae parti contrattuali strutturate da PDF con output JSON schema
2. **Comparison Module:** Confronto binario documento corrente vs precedente (match/no-match per sezione)
3. **Summary Module:** Sintesi differenze in linguaggio naturale italiano per report Excel

## 2.2 Ambiente di sviluppo 🔴🏁🚧

- **Ambiente:** Databricks P1 Sandbox MARK
- **Catalogo Unity Catalog:** `p_sbx_mark.check_documentale`
- **Repository:** Azure DevOps - 1850-ProjectsDataScienceDP/Check_Documentale

<div style="page-break-after: always;"></div>

# 3 Value Model <a name="valuemodel"></a> 🔴🏁🚧

## 3.1 Business Model iniziale 🔴🏁🚧

I benefici attesi consistono nell'incremento dell'efficacia e dell'efficienza delle attività di controllo documentale relative alle campagne marketing. Il prodotto genera valore mediante l'identificazione sistematica delle discrepanze tra documenti, le quali vengono segnalate ai fini della successiva rettifica:

- File CSV contenente l'elenco delle differenze testuali contrattuali tra ciascun documento della campagna corrente e il corrispondente documento della campagna precedente
- Sintesi che specifica la localizzazione e la natura esatta di ciascuna differenza rilevata.

In assenza di una metodologia che consenta di quantificare con precisione il numero di errori effettivi presenti tra le differenze individuate, la determinazione del model value si fonda sui seguenti elementi:

Il valore di business (BV) stimato con il business case nel 2026 è di 265.000€ l'anno.
Questo valore è definito come segue:
```
BV/anno = (risparmio mensile attività * 12 mesi) + mancato ingaggio fornitore = (18.750€ * 12) + 40.000€ = 225.000€ + 40.000€ = 265.000€
```

Dove:
```
Costo 2 FTE al mese: circa 70.000€/mese
```
```
Impegno nei 8 giorni solo per questa specifica attività: circa 18.750€/mese (= 70.000€ × 8/30 giorni)
```
```
Mancato ingaggio fornitore esterno: circa 40.000€/anno
```

<div style="page-break-after: always;"></div>

# 4. User Experience <a name="userexperience"></a> 🔴🏁🚧

Di seguito vengono dettagliati gli output del servizio e la loro interpretazione ed utilizzo.
Viene inoltre descritta l'interazione prevista fra i clienti ed il prodotto.

## 4.1 Descrizione output servito <a name="output"></a> 🔴🏁🚧

Il prodotto elabora i documenti PDF delle campagne marketing e produce un output strutturato per ogni tipologia documentale (Condizione Econimica, Condizione Generale di Contratto, Scheda di Riepilogo, Scheda Sintetica, Proposta di Contratto).

**Input del sistema:**
- Cartella della campagna corrente contenente i PDF organizzati per macrosegmento (RESIDENZIALI, MICROBUSINESS, AUP)
- Cartella della campagna precedente come riferimento per il confronto
- I documenti sono caricati su un volume Databricks nella sandbox MARK

**Output del sistema:**
Il sistema produce i seguenti output:
1. **Tabelle Delta Lake** persistite nel catalogo Unity Catalog contenenti i risultati strutturati del confronto.
2. **File CSV/Excel** per ciascuna tipologia documentale, riportante esclusivamente le differenze rilevate:
   - Riferimento al macrosegmento, alla commodity e al nome del file in cui è stata individuata la differenza
   - Sintesi delle differenze rilevate in linguaggio naturale italiano
   - Descrizione dettagliata delle differenze specifiche tra il documento corrente e il documento precedente
   - Testo della campagna precedente e relativo testo corrispondente della campagna corrente
   - Titolo del chunk e posizione del paragrafo per l'identificazione puntuale della differenza all'interno del file

## 4.2 Interazione AI-utente <a name="InterazioneAI-utente"></a> 🔴🏁🚧

Il processo operativo è articolato nelle seguenti fasi:

**Flusso operativo:**
1. **Upload documenti**: l'utente carica manualmente le cartelle delle campagne sul volume Databricks
2. **Elaborazione automatica**: il workflow Databricks viene attivato ed esegue l'elaborazione dei documenti
3. **Consultazione output**: l'utente scarica i file Excel/CSV generati dal volume per verificare le differenze rilevate
4. **Validazione**: l'utente valuta i risultati e procede con le eventuali correzioni prima della pubblicazione, ovvero segnala gli errori riscontrati.

**Alerting e notifiche:**
- Invio automatico di email in caso di fallimento del job di elaborazione
- Notifica in presenza di anomalie nella qualità dell'output

<div style="page-break-after: always;"></div>

# 5 Dati <a name="dati"></a> 🔴🏁🚧

## 5.1 Data Model 🔴🏁🚧

Il _data model_ è strutturato con dataset master distinti in funzione della tipologia documentale e della fase di elaborazione.

Per ogni tipologia di documento (CE, CGC, SR, SS, PC), esistono le seguenti tabelle:
- **master_[tipo]_train**: tabella di training contenente i chunk di testo dopo estrazione e pre-processing, arricchita con le colonne di Ground Truth;
- **master_[tipo]_infer**: tabella di inferenza contenente i chunk di testo estratti dai PDF;
- **master_[tipo]_predict**: tabella finale con i risultati del confronto e i riassunti delle differenze.

### 5.1.1 Training Master

La tabella **master_ingestion** contiene i metadati delle corrispondenze tra i file PDF presenti nella campagna corrente e quelli della campagna precedente. Questa tabella viene popolata automaticamente a ciscun workflow.

La tabella di ingestion contiene le seguenti colonne principali:

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| DATA_SNAPSHOT | date | Data di elaborazione |
| macrosegmento | string | Segmento di mercato (RESIDENZIALI, MICROBUSINESS, AUP) |
| commodity | string | Tipo commodity (Luce, Gas, Dual) |
| document_type | string | Tipologia documento (CE, CGC, SR, SS, PC) |
| current_file_path | string | Percorso del file PDF della campagna corrente |
| past_file_path | string | Percorso del file PDF della campagna precedente |
| current_campaign | string | Identificativo campagna corrente |
| past_campaign | string | Identificativo campagna precedente |

A partire dalla tabella **master_ingestion** vengono costruite le tabelle di training, contenenti i chunk di testo estratti dai PDF, arricchite con le colonne di Ground Truth.

Le tabelle di training contengono le seguenti colonne principali:

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| DATA_SNAPSHOT | date | Data di elaborazione |
| macrosegmento | string | Segmento di mercato (RESIDENZIALI, MICROBUSINESS, AUP) |
| commodity | string | Tipo commodity (Luce, Gas, Dual) |
| document_type | string | Tipologia documento (CE, CGC, SR, SS, PC) |
| current_campaign | string | Identificativo campagna corrente |
| past_campaign | string | Identificativo campagna precedente |
| chunk | string | Indice del chunk estratto |
| chunk_title | string | Titolo/marker del chunk |
| extracted_chunk_clean | string | Testo estratto e pulito |
| token_count | integer | Numero di token nel chunk |
| chunk_check | boolean | Flag che indica se il chunk esiste in entrambi i documenti |
| comparison | integer | Numero totale di differenze rilevate (Ground Truth) |
| comparison_lists | map | Lista di flag per ogni paragrafo, 0=uguale, 1=diverso (Ground Truth) |
| summary | map | Riassunto delle differenze in italiano (Ground Truth) |
| list_of_differences | map | Elenco dettagliato delle differenze (Ground Truth) |
| corresponding_contractual_paragraphs | map | Paragrafi contrattuali corrispondenti tra documento corrente e precedente (Ground Truth) |

### 5.1.2 Inference Master

Le tabelle di inferenza (`master_[tipo]_infer`) contengono i chunk di testo estratti dai nuovi documenti PDF da elaborare. La struttura è analoga alle tabelle di training, ma senza le colonne di Ground Truth.

Le tabelle di predict (`master_[tipo]_predict`) contengono i risultati finali del confronto generati dal modello, con le stesse colonne di output delle tabelle di training, ma popolate con l'output del modello.

<div style="page-break-after: always;"></div>

## 5.2 Volumi e flussi 🔴🏁🚧

### 5.2.1 Volumi dei dati

I volumi dei dati gestiti sono:
- ~60 file PDF/mese caricati dalle campagne marketing
- 5 file CSV di output per mese (uno per tipologia documentale)
- 3 tabelle UC per tipologia documentale (15 tabelle totali)

I documenti PDF sono organizzati in cartelle per campagna, con sottocartelle per macrosegmento (RESIDENZIALI, MICROBUSINESS, AUP) e tipologia documentale.

### 5.2.2 Flussi di elaborazione

I flussi di elaborazione possono essere distinti in due gruppi: quelli di preparazione dei dataset master e quelli successivi di servizio.

**Flussi di preparazione dei dataset master**

I primi materializzano (in ordine di elenco) la tabella di mapping tra documenti correnti e precedenti (`master_ingestion`) e i dataset master per ogni tipologia documentale, e vengono eseguiti con frequenza mensile in corrispondenza di ogni nuova campagna marketing.

- Ingestion -> Task che scansione cartelle delle campagne e creazione tabella di mapping |
- InferenceMasterCreation_[tipo]-> Task per estrare e pulizia testo dai PDF per ogni tipologia documentale (CE, CGC, SS, SR, PC) |

**Flussi di modellazione e servizio**

I secondi eseguono il training dei modelli LLM con i dati di Ground Truth e l'inferenza per la generazione dei confronti documentali. Il training viene eseguito on-demand, mentre l'inferenza viene attivata con frequenza mensile in corrispondenza di ogni nuova campagna.

- BatchTraining_[tipo] -> Task di addestramento modelli LLM con Ground Truth e ottimizzazione prompt DSPy
- BatchInference_[tipo] -> Task di applicazione modelli per generare confronti e riassunti delle differenze

## 5.3 Sicurezza e protezione 🔴🏁🚧

I dati sono protetti tramite i meccanismi di sicurezza nativi di Databricks Unity Catalog. L'accesso al catalogo `p_sbx_mark` è limitato ai membri del team autorizzati tramite RBAC (Role-Based Access Control).

## 5.4 Retention e cancellazione 🔴🏁🚧

I risultati delle elaborazioni vengono conservati nel catalogo Unity Catalog. Le cartelle delle campagne precedenti vengono mantenute come riferimento per i confronti futuri.

## 5.5 Qualità e validazione 🔴🚧

La qualità dei dati estratti viene verificata attraverso:
- Controllo del flag `chunk_check` che indica se un chunk è stato estratto correttamente da entrambi i documenti.
- Feedback degli utenti marketing sui risultati generati.

## 5.6 Pre-processing e feature engineering 🔴🚧

Il pre-processing dei documenti PDF include:
- **Estrazione testo**: utilizzo di `pdfplumber` per estrarre il testo dai PDF
- **Filtraggio tabelle**: esclusione delle tabelle tramite regex e coordinate per isolare il testo contrattuale
- **Chunking**: suddivisione del testo in chunk basati su marker predefiniti (titoli di sezione)
- **Normalizzazione**: pulizia del testo estratto (normalizzazione apostrofi, rimozione spazi multipli)
- **Stop words**: definizione di parole/frasi da ignorare nel confronto

## 5.7 Bias e fairness 🔴🚧

Il sistema non tratta dati personali e opera su documenti contrattuali standardizzati. Non sono stati identificati rischi di bias o fairness.

<div style="page-break-after: always;"></div>

---

# 11. Appendici <a name="appendice"></a>

## 11.1 Documentazione tecnica <a name="doctecnica"></a> 🏁🚧🆙
Si riporta il link alla documentazione tecnica contenuta nel file [README.md](https://dev.azure.com/DevOps-Applications-EGL/1850-ProjectsDataScienceDP/_git/Check_Documentale?path=/README.md).

## 11.2 Repository codice 🏁
Il codice del prodotto, sviluppato in ambiente di sviluppo DEV, cioè ambiente P1 in sandbox MARK, è archiviato in DevOps sotto l'organizzazione: devosp-applications-Egl -> 1850-ProjectsDataScienceDP (area progetti di data science) nella cartella [Check_Documentale](https://dev.azure.com/DevOps-Applications-EGL/1850-ProjectsDataScienceDP/_git/Check_Documentale).

## 11.3 Riferimenti normativi <a name="norme"></a> 🏁
- Regulation (EU) 2024/1689 - EU AI Act
- Regolamento (UE) 2016/679 - GDPR (General Data Protection Regulation)
- D.lgs. n. 101/2018 che recepisce l'informativa sul trattamento dei dati personali GDPR.

## 11.4 Allegati <a name="allegati"></a>

### 11.4.1 Canvas Business Understanding e Roadmap
Il Canvas Business Understanding e la Roadmap sono presenti nel [Mural](https://app.mural.co/t/eglstream59143/m/eglstream59143/1747834091574/1722977f36169d6797e6228f4a22144e1f78e50c) di progetto.
