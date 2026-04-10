# TraceData ML Pipeline: A Guide for Fullstack Developers

> **Run the repo first:** **[GETTING_STARTED.md](GETTING_STARTED.md)** (copy-paste steps).  
> This guide is a **long analogy-based explainer**—optional after you have a working train + predict loop.

> **Who this is for:** You know what a REST API, a database, and a Docker container are. You've never trained an ML model.
> **How to read this:** Every ML concept will be mapped to something you already understand.

---

## The One-Sentence Summary

This system takes raw sensor readings from a car, runs them through a math formula stored in a file, and produces a score — the same way a REST API takes a JSON body, processes it in a service class, and returns a response. The "ML" part is just an unusually smart service class.

---

## 🗺️ The Whole System at a Glance

```mermaid
flowchart LR
    A["📡 Car Sensor Data"] --> B["📐 Feature Extractor\n(like a DTO transformer)"]
    B --> C["🤖 Scoring Model\n(like a service class,\nbut trained not coded)"]
    C --> D["🔍 Why-Explainer\n(like a stack trace\nfor the model's decision)"]
    C --> E["⚖️ Fairness Checker\n(cohort comparison)"]
    D --> F["🧠 Behavior Agent\n(JSON → English sentence)"]
    E --> F
    F --> G["📊 Driver Dashboard"]
```

---

## Stage 1: Getting Data In

### Telemetry = A Stream of Events

Think of telemetry like **server access logs** — except instead of HTTP requests, you are logging what a car is doing every 30 seconds.

```mermaid
timeline
    title One 5-Minute Trip (One Log Entry Every 30 Seconds)
    T+00s : speed=45 kmh, acceleration=0.2
    T+30s : speed=72 kmh, acceleration=1.8
    T+60s : speed=68 kmh, acceleration=-0.4
    T+90s : speed=90 kmh, acceleration=2.2
    T+120s : speed=85 kmh, acceleration=-1.6
    T+150s : speed=60 kmh, acceleration=-2.1
```

### The Database Schema

This is just a **normalized relational database**. Nothing unfamiliar here.

```mermaid
erDiagram
    DRIVERS {
        int driver_id PK
        string name
        float smoothness_avg "Average score across all trips"
        float safety_avg
        float overall_avg
        int trip_count
        json explanation_json "ML explanation, stored as JSON"
        json fairness_metadata_json "Cohort comparison result"
        text coaching_narrative "Final English sentence from Agent"
    }
    TRIPS {
        int trip_id PK
        int driver_id FK
        datetime start_time
        datetime end_time
        float smoothness_score "Output of the ML model"
        float safety_score "Output of the rules engine"
        json explanation_json "Why this trip scored this way"
        json fairness_metadata_json "How this trip compares to peers"
    }
    TELEMETRY_POINTS {
        int point_id PK
        int trip_id FK
        datetime timestamp
        float speed_kmh
        float acceleration_ms2 "The key raw value"
        float lat
        float lon
    }

    DRIVERS ||--o{ TRIPS : "has many"
    TRIPS ||--o{ TELEMETRY_POINTS : "captured as"
```

---

## Stage 2: Feature Engineering — The "DTO Transformation" of ML

### The Problem

A model cannot understand "this driver brakes harshly." It only understands **numbers in columns**, same as how your backend cannot process a blob of text — it needs a structured DTO.

**Feature Engineering is converting raw sensor arrays into a structured row of meaningful numbers.** You are the translator between physics and math.

```mermaid
flowchart TD
    subgraph "RAW INPUT (like a raw HTTP body)"
        R["acceleration_readings:\n[0.2, 1.8, -0.4, 2.2, -1.6, -2.1]"]
    end

    subgraph "features.py (like a DTO transformer / mapper)"
        F1["Compute Jerk = change in acceleration\n[1.6, -2.2, 2.6, -3.8, -0.5]"]
        F2["accel_fluidity = std_dev(Jerk) = 2.1\n🡒 High means jerky"]
        F3["driving_consistency = std_dev(accel) = 1.7\n🡒 High means erratic"]
        F4["comfort_zone_percent = 33%\n🡒 % of time in gentle range"]
    end

    subgraph "STRUCTURED OUTPUT (like a DTO)"
        M["{\n  accel_fluidity: 2.1,\n  driving_consistency: 1.7,\n  comfort_zone_percent: 33.0\n}"]
    end

    R --> F1 --> F2
    R --> F3
    R --> F4
    F2 --> M
    F3 --> M
    F4 --> M
```

| Feature | Your Domain Analogy | Meaning |
| :--- | :--- | :--- |
| `accel_fluidity` | Response time variance across API calls | High = unstable / jerky |
| `driving_consistency` | Throughput variance across requests | High = unpredictable load behavior |
| `comfort_zone_percent` | % of requests under SLA threshold | High = mostly gentle driving |

---

## Stage 3: ML Training — "Compiling" Business Logic

### The Core Insight

In traditional development, you **write** business logic: `if speed > 120 => deduct points`.

In ML, you **show** the system thousands of examples and let it figure out the logic itself. Training is the equivalent of **compilation** — you do it once offline, and the output is a binary artifact you ship.

```mermaid
flowchart LR
    subgraph "OFFLINE: Training (trainer.py)"
        TD["Training Dataset\n(thousands of trips\nwith known scores)"] --> XG["XGBoost\n.fit()"]
        XG --> ART["models/smoothness_model.joblib\n📦 The ML Binary Artifact"]
    end

    subgraph "ONLINE: Runtime (scoring.py)"
        NR["New Trip\n(never seen before)"] --> P["model.predict()"]
        ART2["models/smoothness_model.joblib"] --> P
        P --> SC["Score: 73 / 100"]
    end

    ART -.-> |"Loaded into memory\nat container startup"| ART2
```

> **Analogy:** The `.joblib` file is equivalent to a compiled `.jar` or `.dll`. You do not recompile on every request. You load it once and call it.

### The Training Loop (Step-by-Step)

```mermaid
sequenceDiagram
    participant T as trainer.py
    participant DB as SQLite DB
    participant XG as XGBoost
    participant FS as File System

    T->>DB: Get all trips with [fluidity, consistency, comfort] + known scores
    DB-->>T: Rows of (features, score) pairs
    T->>T: Build input matrix X and label array y
    T->>XG: model.fit(X_train, y_train)
    Note over XG: Learns: high fluidity + low comfort\n→ low smoothness score
    XG-->>T: Trained model object in memory
    T->>FS: joblib.dump(model, "models/smoothness_model.joblib")
    Note over FS: "Learned knowledge" frozen to disk\nlike a compiled binary
    T->>XG: model.score(X_test, y_test)
    XG-->>T: R² = 0.97 (97% accuracy on test data)
```

---

## Stage 4: Inference — Calling the Trained Service

"Inference" is the ML word for **running the model at runtime to get a prediction**. Think of it like calling a service method. The model is just a really complex `calculateScore(features)` function.

```mermaid
flowchart TD
    subgraph "What scoring.py does (the orchestrator)"
        T["New telemetry arrives"] --> FE["features.py\nextract_features()"]
        FE --> ML["model.predict(features)\n🤖 ML Smoothness Score"]
        T --> RULES["Safety Rules Engine\nif speed > 120 → deduct 20pts"]
        ML --> CS["Composite Score\n= 0.6 × smooth + 0.4 × safety"]
        RULES --> CS
        CS --> DB["Persist to trips table"]
    end
```

> **Why use ML for smoothness but rules for safety?**
>
> Safety is **objective**: you either violated a speed limit or you did not. A rule is better.
> Smoothness is **subjective**: what feels smooth varies. An ML model trained on patterns is better.

---

## Stage 5: Explainability (XAI) — The Stack Trace for ML Decisions

### The Problem
The model gives you a score of 62. You cannot ship that to a driver without explaining *why*. In software, when something fails, you read the stack trace. In ML, you use **SHAP** to produce the equivalent.

```mermaid
flowchart LR
    subgraph "WITHOUT SHAP (opaque)"
        IN["[fluidity=2.1,\n consistency=3.8,\n comfort=25.8]"] --> MODEL["⬛ Black Box"]
        MODEL --> OUT["Score: 62\n(No explanation)"]
    end

    subgraph "WITH SHAP (explain.py)"
        BV["Starting Point (Base Value): 67.1"]
        C1["accel_fluidity effect: -5.0\n'Your jerkiness cost you 5 points'"]
        C2["driving_consistency effect: +3.8\n'Your consistency earned you 3.8 points'"]
        C3["comfort_zone effect: -3.9\n'Low comfort cost you 3.9 points'"]
        RESULT["Final Score: 67.1 - 5.0 + 3.8 - 3.9 = 62 ✅"]
    end

    OUT -.->|"SHAP reverse-engineers this"| BV
```

### Two Levels of Explanation

```mermaid
graph TD
    subgraph "Trip-Level (Local): What happened on Trip 47?"
        T1["Trip #47 Score: 62"]
        T1 --> T2["SHAP → accel_fluidity caused -5pts\n(jerky start at T+30s)"]
    end

    subgraph "Driver-Level (Global): What is Ahmad's driving signature?"
        D1["Ahmad's Average Score: 74"]
        D1 --> D2["Aggregate SHAP → consistency is\nhis strongest trait across 50 trips"]
    end
```

---

## Stage 6: Fairness — Is the Score Context-Aware?

### The Question
If a 68-year-old driver scores 70 — is that good or bad? Without context, we don't know. Fairness analysis adds cohort benchmarking.

```mermaid
flowchart TD
    subgraph "fairness.py"
        DS["Driver's Score: 98"]
        CG["Driver's Age Cohort: 61-70 years"]
        DB2["All drivers age 61-70 → Average: 68.15"]
        DS --> DIFF["diff = 98 - 68.15 = +29.85"]
        DB2 --> DIFF
        DIFF --> STATUS["This driver is outperforming their cohort ✅"]
        STATUS --> JSON["fairness_metadata_json:\n{ age_cohort_avg: 68.15, diff: +29.85 }"]
    end
```

### The Design Decision: Show Context, Never Change the Score

```mermaid
stateDiagram-v2
    state "Bias signal detected" as BIAS
    state "Silently adjust score downward" as BAD
    state "Show cohort context alongside score" as GOOD

    BIAS --> BAD : ❌ Bad approach\n(doctor adjusting test results)
    BIAS --> GOOD : ✅ TraceData approach\n(doctor showing population statistics)

    note right of BAD : Driver never knows.\nTrust breaks.
    note right of GOOD : "You scored 29pts above\nyour age group average."
```

---

## Stage 7: The Behavior Agent — Converting JSON to English

### Why This Exists

All the stages above produce **structured JSON**. Your drivers are not data scientists. They need a sentence, not a dictionary. The Behavior Agent does this translation.

```mermaid
flowchart TD
    subgraph "src/agents/behavior/agent.py"
        IN1["XAI JSON:\n{accel_fluidity: 1.28,\n driving_consistency: 3.80,\n base_value: 67.1}"]
        IN2["Fairness JSON:\n{age_cohort_avg: 68.15, diff: +29.86}"]

        S1["1. Filter out 'base_value'\n(it's a model intercept, not a behavior)"]
        S2["2. Find top feature\n(driving_consistency = 3.80 is highest)"]
        S3["3. Map fairness diff to status\n(+29.86 → 'outperforming')"]
        S4["4. Fill sentence template\nor later: call Gemini LLM"]

        OUT["'Based on your recent trips, you are outperforming\nyour age cohort by 29.86 points. Your success is\ndriven by your excellent driving consistency.'"]
    end

    IN1 --> S1 --> S2
    IN2 --> S3
    S2 --> S4
    S3 --> S4
    S4 --> OUT
```

---

## Stage 8: Deployment — MLOps Is Just DevOps for Models

### The Full Picture

MLOps = DevOps + the additional concern of managing ML artifacts (model files) and ensuring that what you trained offline is exactly what runs at production.

```mermaid
flowchart TB
    subgraph "📡 Data Sources"
        IOT["IoT Device\n(Vehicle Sensor)"]
        APP["Driver App"]
    end

    subgraph "docker-compose.yml"
        REDIS["🔴 Redis\n(Message Broker)"]

        subgraph "core-api container"
            API["FastAPI\n• POST /feedback\n• GET /driver/{id}"]
        end

        subgraph "agent-worker container"
            WORKER["Background worker\n• Feature Engineering\n• XGBoost.predict()\n• SHAP\n• Fairness\n• Behavior Agent"]
        end

        DB[("SQLite / PostgreSQL")]
    end

    IOT -->|"Direct push\n(high-volume telemetry)"| REDIS
    APP -->|"POST /feedback"| API
    API -->|"Enqueue task"| REDIS
    REDIS -->|"Dequeue & process"| WORKER
    WORKER -->|"Write scores + narrative"| DB
    APP -->|"GET /driver/1"| API
    API -->|"Read pre-computed result"| DB
    DB -->|"Dashboard data"| API
```

### Why Two Containers?

```mermaid
quadrantChart
    title What Each Container Optimises For
    x-axis "Response Speed: Slow ←——→ Fast"
    y-axis "Compute: Light ←——→ Heavy"
    quadrant-1 "Heavy + Fast\n(ideal but expensive)"
    quadrant-2 "Heavy + Slow\n(background workers)"
    quadrant-3 "Light + Slow\n(batch jobs)"
    quadrant-4 "Light + Fast\n(real-time APIs)"
    "core-api": [0.85, 0.2]
    "agent-worker": [0.2, 0.85]
```

**`core-api`** must be fast because a user is waiting. It only reads from the database.

**`agent-worker`** can be slow because it runs in the background. It does all the heavy computation (ML inference, SHAP, LLM).

---

## End-to-End: The Complete Request Journey

```mermaid
sequenceDiagram
    actor Driver as 🧑 Driver
    participant App as Driver App
    participant API as core-api (FastAPI)
    participant Redis as 🔴 Redis
    participant Worker as agent-worker (async jobs)
    participant DB as 🗄️ Database

    Note over Driver,Redis: A new trip ends
    App->>Redis: Telemetry payload pushed directly
    Redis->>Worker: Consume task
    Worker->>Worker: features.py → 3 numbers
    Worker->>Worker: model.predict() → smoothness: 73
    Worker->>Worker: safety_rules() → safety: 100
    Worker->>Worker: shap.explain() → explanation_json
    Worker->>Worker: fairness.audit() → fairness_json
    Worker->>Worker: agent.generate() → narrative text
    Worker->>DB: UPDATE drivers SET coaching_narrative, score...

    Note over Driver,DB: Later — driver opens the app
    Driver->>App: Opens profile
    App->>API: GET /driver/1
    API->>DB: SELECT * FROM drivers WHERE id=1
    DB-->>API: All pre-computed data
    API-->>App: { score, xai, fairness, coaching_narrative }
    App-->>Driver: "You are outperforming your\ncohort by 29.86 points..."
```

---

## 📂 Where to Find Everything

```mermaid
graph TD
    subgraph "src/core/ — The Brains"
        FT["features.py\nRaw data → structured numbers"]
        SC["scoring.py\nOrchestrates ML + Rules"]
        EX["explain.py\nSHAP stack trace"]
        FA["fairness.py\nCohort comparison"]
    end

    subgraph "src/inference/ — Serving"
        INF["smoothness_inference.py\nPing-window scoring"]
        DEV["device_trip_scorer.py\nDevice aggregates → trip score"]
    end

    subgraph "src/utils/ — The Toolbox"
        SIM["simulator.py\nGenerates fake training data"]
        TR["trainer.py\nTrains + saves the model"]
        PR["processor.py\nBatch-extracts features"]
    end

    subgraph "Root"
        MAIN["main.py\nFastAPI app"]
        DC["docker-compose.yml\nAll 3 containers"]
        MOD["models/\n.joblib artifact files"]
    end
```

---

## 📖 Glossary: ML → Fullstack Translations

| ML/AI Term | What You Already Know | TraceData File |
|:---|:---|:---|
| **Feature Engineering** | DTO transformation / data mapping | `src/core/features.py` |
| **Training** | Compiling code from source | `src/utils/trainer.py` |
| **Model Artifact** | A compiled `.jar` or `.dll` binary | `models/smoothness_model.joblib` |
| **Inference** | Calling a service method at runtime | `model.predict()` in `scoring.py` |
| **SHAP (XAI)** | A stack trace for the model's decision | `src/core/explain.py` |
| **Fairness Auditing** | A/B comparison with population stats | `src/core/fairness.py` |
| **Synthetic Data** | Mocked unit test data | `src/utils/simulator.py` |
| **Model Drift** | A memory leak that accumulates over time | *(Future monitoring phase)* |
| **MLOps** | DevOps — but also managing the model artifact lifecycle | `tracedata-mlops` CLI + MLflow |
