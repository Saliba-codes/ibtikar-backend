# **Ibtikar — Backend System**

**Created by:** Saliba Rishmawi  
**Initial Release:** 15th of November 2025

### _Full Documentation — Development • Deployment • API Reference_

This repository contains the complete backend for **Ibtikar**, a system designed to:

1. Fetch posts (tweets) from connected X/Twitter accounts
2. Run toxicity analysis using a custom Arabic BERT model
3. Store predictions in a SQLite database
4. Provide analytics endpoints for authors, labels, languages, timestamps, etc.
5. Serve clean API endpoints to a dedicated frontend

---

# **Table of Contents**

1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Folder Structure](#folder-structure)
4. [Environment Setup](#environment-setup)
5. [How to Run the Backend](#how-to-run-the-backend)
6. [OAuth Guide (X/Twitter)](#oauth-guide-xtwitter)
7. [API Endpoints Summary](#api-endpoints-summary)
8. [Full API Reference](#full-api-reference)
9. [Database Schema](#database-schema)
10. [Prediction Logic & Duplicate Prevention](#prediction-logic--duplicate-prevention)
11. [Deployment Guide](#deployment-guide)
12. [Maintainer Notes](#maintainer-notes)
13. [License](#license)

---

# **Project Overview**

The backend consists of **two independent APIs**:

---

## **1. Toxicity Model API** (port **9000**)

- Hosts the custom Arabic BERT classifier
- Endpoint: `/predict`
- Input: raw text
- Output:

  ```json
  {
    "label": "safe" | "harmful" | "unknown",
    "score": float
  }
  ```

---

## **2. Main Backend API** (port **8000**)

- Handles OAuth authentication with X/Twitter
- Fetches the user’s following feed
- Sends each post to the model API for classification
- Saves results in `ngodb.sqlite3`
- Provides analytics endpoints for frontend dashboards

---

# **Technology Stack**

| Layer        | Technology                     |
| ------------ | ------------------------------ |
| Backend API  | FastAPI, Uvicorn               |
| Model API    | FastAPI, Transformers, PyTorch |
| Database     | SQLite + SQLAlchemy            |
| Auth         | OAuth 2.0 PKCE                 |
| External API | X/Twitter API v2               |
| Deployment   | Uvicorn                        |

---

# **Folder Structure**

```
NGO/
├─ backend/                     # Main backend API
│ ├─ api/                       # FastAPI routes
│ │ ├─ init.py
│ │ └─ main.py
│ │
│ ├─ clients/                   # X API client + Model API client
│ │ ├─ init.py
│ │ ├─ ibtikar_client.py
│ │ ├─ x_api.py
│ │ └─ x_client.py
│ │
│ ├─ core/                     # Settings & configuration
│ │ ├─ init.py
│ │ ├─ config.py
│ │ ├─ crypto.py
│ │ ├─ memory.py
│ │ ├─ normalize.py
│ │ └─ schemas.py
│ │
│ ├─ db/                       # SQLAlchemy engine, sessions, models
│ │ ├─ init.py
│ │ ├─ init_db.py
│ │ ├─ models.py
│ │ └─ session.py
│
│
├─ IbtikarAI/                   # Toxicity model microservice
│ ├─ arabert_toxic_classifier/
│ │ ├─ config.json
│ │ ├─ model.safetensors         # Model weights (Git LFS)
│ │ ├─ special_tokens_map.json
│ │ ├─ tokenizer_config.json
│ │ ├─ tokenizer.json
│ │ └─ vocab.txt
│ │
│ ├─ .gitattributes
│ ├─ finetunning.py
│ ├─ flip_code.py
│ ├─ ibtikar_api.py               # Model inference API
│ ├─ test.py
│ └─ validation_test.py
│
├─ .env.example                   # Template for environment variables
├─ .gitignore                     # Git ignore rules
├─ LICENSE                        # MIT License
├─ README.md
└─ requirements.txt
```

---

# **Environment Setup**

### **1. Create a virtual environment**

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Create your `.env` file**

```bash
cp .env.example .env
```

### **4. Fill in the required values**

```env
# --- X OAuth ---
X_CLIENT_ID=YOUR_X_CLIENT_ID
X_REDIRECT_URI=http://localhost:8000/v1/oauth/x/callback
X_SCOPES=tweet.read users.read follows.read offline.access

# --- Token Encryption ---
FERNET_KEY=GENERATE_A_32_BYTE_KEY_BASE64

# --- Database ---
DATABASE_URL=sqlite:///./ngodb.sqlite3

# --- Toxicity Model API URL ---
IBTIKAR_URL=http://127.0.0.1:9000
```

---

# **How to Run the Backend**

You must run **two APIs** simultaneously.

---

## **Terminal 1 — Main Backend (port 8000)**

Run from the project root:

```bash
uvicorn backend.api.main:app --reload --port 8000
```

Docs:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## **Terminal 2 — Toxicity Model API (port 9000)**

Inside `IbtikarAI/`:

```bash
uvicorn ibtikar_api:app --reload --port 9000
```

Docs:
[http://127.0.0.1:9000/docs](http://127.0.0.1:9000/docs)

---

# **OAuth Guide (X/Twitter)**

### **1. Start OAuth**

Open in browser:

```
http://127.0.0.1:8000/v1/oauth/x/start?user_id=USER_ID
```

### **2. User logs in → approves scopes**

Required scopes include tweet read access & following feed access.

### **3. Callback received**

X returns:

```
/v1/oauth/x/callback?state=xxxx&code=yyyy
```

The backend then:

- Exchanges `code` for access + refresh tokens
- Saves tokens in `x_tokens` table
- Links them to the internal `user_id`

---

# **API Endpoints Summary**

| Endpoint               | Method | Description                      |
| ---------------------- | ------ | -------------------------------- |
| `/v1/oauth/x/start`    | GET    | Start OAuth flow                 |
| `/v1/oauth/x/callback` | GET    | OAuth callback                   |
| `/v1/analysis/preview` | POST   | Fetch → analyze → save new posts |
| `/v1/analysis/posts`   | GET    | Retrieve analyzed posts          |
| `/v1/analysis/authors` | GET    | Per-author statistics            |
| `/predict`             | POST   | Toxicity model endpoint          |

---

# **Full API Reference**

## **POST /v1/analysis/preview**

Runs the full pipeline:

1. Fetch following feed
2. Deduplicate by `(user_id, source, post_id)`
3. Classify via model API
4. Insert new records
5. Return summary response

---

## **GET /v1/analysis/posts**

Supports filtering by:

- `label` — safe | harmful | unknown
- `author_id`
- `lang` — ar | en
- `from_created_at`
- `to_created_at`
- `limit`
- `offset`

---

## **GET /v1/analysis/authors**

Returns aggregated statistics for each author:

- Total posts analyzed
- Harmful/safe counts
- Language distribution
- Date ranges

---

# **Database Schema**

### **Table: predictions**

| Column          | Type     | Description              |
| --------------- | -------- | ------------------------ |
| id              | int (PK) | Internal record ID       |
| user_id         | int      | Internal system user     |
| source          | str      | e.g., "x"                |
| post_id         | str      | Tweet ID                 |
| author_id       | str      | Tweet author             |
| lang            | str      | ar/en/etc                |
| text            | str      | Tweet text               |
| label           | str      | safe / harmful / unknown |
| score           | float    | Confidence               |
| post_created_at | datetime | Timestamp from X         |
| created_at      | datetime | Inserted timestamp       |

**Unique Constraint:**

```
UNIQUE(user_id, source, post_id)
```

---

# **Prediction Logic & Duplicate Prevention**

For each fetched post:

1. Check DB for `(user_id, source, post_id)`
2. If it exists → **skip**
3. If not → **run prediction → insert into DB**

This ensures:

- No duplicates
- Faster future runs
- Accurate analytics

---

# **Deployment Guide**

### **Local**

- Use Uvicorn directly
- Make sure both APIs use dedicated ports
- Ensure `.env` is configured

### **Production Notes**

- Run behind Nginx or Caddy
- Use systemd or Docker to keep services alive
- Switch from SQLite to PostgreSQL for multi-user load
- Add HTTPS before connecting from production frontend

---

# **Maintainer Notes**

- Never commit `.env`
- Regenerate `SECRET_KEY` before production
- SQLite is fine for development, but not for multi-user scaling
- Token refresh logic should run periodically (cron or scheduled task)

---

# **License**

This project is licensed under the MIT License.
