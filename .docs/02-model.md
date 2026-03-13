# 02 — train.py: Серце експерименту

> [[01-guide|Назад до теорії]] | [[03-practice|Далі: практика]]

---

## 6. train.py — серце експерименту

Файл ~1366 рядків. Це **єдиний файл** який змінюється під час експериментів.

### 6.1. Загальна архітектура файлу

```mermaid
flowchart TB
    subgraph RC ["Runtime Configuration (рядки 38–310)"]
        R1["GpuProfile — профіль GPU"]
        R2["Autotune — підбір batch size"]
        R3["detect_runtime() — визначення GPU"]
    end

    subgraph MODEL ["GPT Model (рядки 321–640)"]
        M1["GPTConfig — конфігурація"]
        M2["CausalSelfAttention — увага з RoPE"]
        M3["MLP — feedforward (ReluSquared)"]
        M4["Block — Attention + MLP"]
        M5["GPT — повна модель"]
    end

    subgraph OPT ["Optimizer (рядки 640–790)"]
        O1["Muon — для матриць"]
        O2["AdamW — для скалярів/embeddings"]
        O3["MuonAdamW — гібрид"]
    end

    subgraph HYPER ["Hyperparameters (рядки 794–820)"]
        H1["DEPTH, ASPECT_RATIO, HEAD_DIM"]
        H2["LR, BATCH, WARMUP, WARMDOWN"]
    end

    subgraph TRAIN ["Training Loop (рядки 820–1366)"]
        T1["build_model_config()"]
        T2["_autotune_train_candidate()"]
        T3["_run_training_once()"]
        T4["main()"]
    end

    RC --> MODEL --> OPT --> TRAIN
    HYPER --> TRAIN

    style MODEL fill:#2ed573,color:#000
    style OPT fill:#ff9f43,color:#fff
    style HYPER fill:#4a9eff,color:#fff
```

### 6.2. Runtime Configuration

Автоматично визначає GPU і налаштовує параметри:

```mermaid
flowchart LR
    GPU["Твоя GPU"] --> DETECT["detect_runtime()"]
    DETECT --> PROFILE["GpuProfile"]
    PROFILE --> BATCH["Batch candidates"]
    PROFILE --> CKPT["Checkpointing"]
    DETECT --> AMP["AMP dtype<br/>(bf16/fp16)"]
    DETECT --> TF32["TF32 enable"]

    style GPU fill:#4a9eff,color:#fff
    style PROFILE fill:#ff9f43,color:#fff
```

**GPU Profiles** — передналаштовані конфігурації:

| Профіль | GPU | VRAM | Batch candidates |
|---------|-----|------|------------------|
| `turing-8-11gb` | RTX 2060/2070 | 8–11 GB | `[8, 4, 2, 1]` |
| `turing-12-15gb` | RTX 2080 Ti | 11–15 GB | `[16, 8, 4]` |
| `ampere-10-15gb` | RTX 3060/3080 | 10–15 GB | `[16, 8, 4]` |
| `ada-10-15gb` | RTX 4070/4070TiS | 10–15 GB | `[16, 8, 4]` |
| `ada-16gb` | RTX 4080 | 16 GB | `[32, 16, 8, 4]` |
| `ada-24gb-plus` | RTX 4090 | 24+ GB | `[64, 32, 16, 8, 4]` |

**Autotune**: при першому запуску пробує різні batch size і обирає найшвидший. Результат кешується в `~/.cache/autoresearch/autotune_cache.json`.

### 6.3. Модель — GPT

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048   # Довжина контексту
    vocab_size: int = 8192     # Розмір словника (з prepare.py)
    n_layer: int = 12          # Кількість transformer шарів
    n_head: int = 6            # Кількість "голів" уваги
    n_kv_head: int = 6         # KV-голови (GQA)
    n_embd: int = 768          # Розмір embedding вектора
    window_pattern: str = "SSSL"
```

> Ці дефолти перевизначаються через `build_model_config()` на основі `DEPTH` та `ASPECT_RATIO`.

### 6.4. Компоненти моделі

```mermaid
flowchart TB
    INPUT["Вхідні токени [B, T]"] --> WTE

    subgraph WTE ["Embedding (wte)"]
        W1["token → вектор dim"]
    end

    WTE --> VE["Value Embeddings<br/>(кожен 2-й шар)"]

    WTE --> BLOCKS

    subgraph BLOCKS ["Transformer Blocks x N"]
        direction TB
        subgraph BLOCK ["Один Block"]
            NORM1["RMSNorm"] --> ATT
            subgraph ATT ["CausalSelfAttention"]
                QKV["Q, K, V projections"]
                ROPE["RoPE (позиційне кодування)"]
                SDPA["Scaled Dot-Product Attention"]
                VEG["Value Embedding gate"]
                QKV --> ROPE --> SDPA
                VEG --> SDPA
            end
            ATT --> SKIP1["+ residual (resid_lambda)"]
            SKIP1 --> NORM2["RMSNorm"]
            NORM2 --> MLP_B
            subgraph MLP_B ["MLP"]
                FC["Linear(dim, 3*dim)"]
                ACT["ReluSquared: relu(x)^2"]
                PROJ["Linear(3*dim, dim)"]
                FC --> ACT --> PROJ
            end
            MLP_B --> SKIP2["+ residual"]
        end
    end

    VE -.->|"gate"| ATT
    BLOCKS --> X0["+ x0_lambda * embedding"]
    X0 --> LMH["LM Head<br/>Linear(dim, vocab_size)"]
    LMH --> OUTPUT["Ймовірності слів"]

    style INPUT fill:#4a9eff,color:#fff
    style OUTPUT fill:#2ed573,color:#000
```

Ключові архітектурні деталі:

- **RMSNorm** — нормалізація замість LayerNorm (простіша, без bias)
- **RoPE** — Rotary Positional Embeddings (кодує позицію токена в реченні)
- **GQA** — Grouped Query Attention: кількість KV-голів може бути меншою за Q-голови
- **Value Embeddings** — додаткові learnable вектори в кожному 2-му шарі, з gated додаванням
- **Residual lambdas** — learnable масштаб skip-connection для кожного шару
- **x0 connection** — додає масштабований embedding до виходу (shortcut через усі блоки)
- **ReluSquared** — `relu(x)^2` замість GELU (простіше, часто краще для малих моделей)

### 6.5. Sliding Window Attention

Патерн `SSSL` означає:

```mermaid
flowchart LR
    subgraph Pattern ["Патерн SSSL (повторюється по шарах)"]
        L1["Шар 0: S<br/>short window<br/>1024 токени"]
        L2["Шар 1: S<br/>short window<br/>1024 токени"]
        L3["Шар 2: S<br/>short window<br/>1024 токени"]
        L4["Шар 3: L<br/>long window<br/>2048 токенів"]
        L1 --> L2 --> L3 --> L4
    end

    style L4 fill:#2ed573,color:#000
```

Short-шари швидші (менше обчислень), long-шар бачить повний контекст.

### 6.6. Оптимізатор — MuonAdamW

Гібридний оптимізатор з окремим підходом для різних типів параметрів:

```mermaid
flowchart TB
    PARAMS["Всі параметри моделі"] --> SPLIT{"Тип параметра?"}

    SPLIT -- "Матриці (weights)" --> MUON["Muon optimizer<br/>Ортогональні оновлення<br/>QR-декомпозиція"]
    SPLIT -- "Embedding шар" --> ADAM_E["AdamW<br/>LR = EMBEDDING_LR"]
    SPLIT -- "LM Head (unembedding)" --> ADAM_U["AdamW<br/>LR = UNEMBEDDING_LR"]
    SPLIT -- "Скаляри, lambdas, VE gates" --> ADAM_S["AdamW<br/>LR = SCALAR_LR"]

    MUON --> STEP["optimizer.step()"]
    ADAM_E --> STEP
    ADAM_U --> STEP
    ADAM_S --> STEP

    style MUON fill:#ff9f43,color:#fff
    style ADAM_E fill:#4a9eff,color:#fff
    style ADAM_U fill:#4a9eff,color:#fff
    style ADAM_S fill:#4a9eff,color:#fff
```

| Тип параметра | Оптимізатор | LR | Чому |
|---------------|-------------|-----|------|
| Матриці (weights) | **Muon** | `MATRIX_LR` (0.12) | Ортогональні оновлення — зберігає геометрію |
| Embedding | **AdamW** | `EMBEDDING_LR` (1.2) | Великий LR для швидкої адаптації embedding |
| Unembedding (LM Head) | **AdamW** | `UNEMBEDDING_LR` (0.008) | Маленький LR для стабільності виходу |
| Скаляри, lambdas | **AdamW** | `SCALAR_LR` (0.5) | Стандартний |

### 6.7. Training Loop

```mermaid
flowchart TB
    START["main()"] --> CONFIG["build_model_config()"]
    CONFIG --> AUTOTUNE["autotune batch size"]
    AUTOTUNE --> INIT["Створити модель + оптимізатор"]
    INIT --> LOOP

    subgraph LOOP ["Training Loop (300 секунд)"]
        BATCH["Взяти батч [B, 2048]"] --> FWD["Forward pass → loss"]
        FWD --> BWD["Backward pass → градієнти"]
        BWD --> ACC{"Accumulation\nготовий?"}
        ACC -- "Ні" --> BATCH
        ACC -- "Так" --> STEP["optimizer.step()"]
        STEP --> LOG_S["Логувати step"]
        LOG_S --> TIME{"Час вичерпано?"}
        TIME -- "Ні" --> BATCH
        TIME -- "Так" --> EVAL
    end

    EVAL["evaluate_bpb()"] --> REPORT["Вивести результат"]

    style START fill:#4a9eff,color:#fff
    style EVAL fill:#2ed573,color:#000
    style REPORT fill:#2ed573,color:#000
```

**Gradient Accumulation**: `TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * MAX_SEQ_LEN)` міні-батчів збираються перед одним кроком оптимізатора. Це дозволяє симулювати великий batch size на GPU з малим VRAM.

### 6.8. Формула розміру моделі

```python
base_dim = DEPTH * ASPECT_RATIO
model_dim = ceil(base_dim / HEAD_DIM) * HEAD_DIM  # вирівнювання по HEAD_DIM
num_heads = model_dim // HEAD_DIM
```

Приклади:

| DEPTH | AR | base_dim | model_dim | heads | ~params |
|-------|-----|----------|-----------|-------|---------|
| 2 | 256 | 512 | 512 | 4 | 18.9M |
| 3 | 171 | 513 | 640 | 5 | 33.3M |
| 4 | 128 | 512 | 512 | 4 | 29.4M |
| 6 | 128 | 768 | 768 | 6 | 66.8M |
| 8 | 64 | 512 | 512 | 4 | 50.3M |

---

> **Далі**: [[03-practice|Практика: гіперпараметри, запуск, логи, автономний режим]]
