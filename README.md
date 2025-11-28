# 🤖 Automated Quiz Solver Agent

A robust, intelligent agent designed to autonomously solve dynamic data analysis quizzes. This application leverages a powerful combination of **FastAPI**, **Playwright**, and **Multimodal LLMs** (Gemini 2.0 Flash, GPT-5-nano, Claude 4 Opus) to navigate web pages, scrape data, transcribe audio, analyze datasets, and submit answers within strict time limits.

## ✨ Key Features

*   **Autonomous Navigation:** Uses Playwright to interact with dynamic web pages, handle forms, and manage pagination.
*   **Multimodal Intelligence:**
    *   **Audio/Image Processing:** Google Gemini 2.0 Flash for transcribing audio and analyzing visual content.
    *   **Logic & Code Generation:** GPT-5-nano (via OpenRouter) for generating complex Pandas analysis code, with automatic fallback to Claude 4 Opus.
*   **Data Analysis Pipeline:** Automatically cleans, processes, and analyzes diverse data formats (CSV, JSON, APIs) using generated Python code.
*   **Resilient Architecture:** Implements smart retries, error handling for API failures (404s), and fallback mechanisms for LLM providers.
*   **Secure:** All sensitive credentials are managed via environment variables.

## 📂 Project Structure

```
.
├── app/
│   ├── core/
│   │   └── config.py          # Configuration and environment variable management
│   ├── services/
│   │   ├── browser.py         # Playwright browser automation service
│   │   ├── gemini_helper.py   # Interface for Google Gemini API (Multimodal)
│   │   ├── llm_helper.py      # Unified LLM interface with fallback logic (OpenAI/Claude)
│   │   ├── quiz_solver.py     # Main orchestration logic for solving quizzes
│   │   └── question_handlers.py # Specialized handlers for specific question types
│   └── main.py                # FastAPI application entry point
├── .env                       # Environment variables (Git-ignored)
├── Dockerfile                 # Containerization setup
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.9+
*   Google Gemini API Key
*   OpenAI/OpenRouter API Key (for GPT-5-nano)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Playwright browsers:**
    ```bash
    playwright install chromium
    ```

4.  **Configure Environment:**
    Create a `.env` file in the root directory with the following variables:

    ```ini
    # Application Secrets
    APP_SECRET=your_app_secret
    QUIZ_EMAIL=your_email@example.com
    QUIZ_SECRET=your_quiz_secret

    # LLM Provider Configuration
    # Primary Multimodal Model (Audio/Images)
    GEMINI_API_KEY=your_gemini_api_key
    GEMINI_MODEL=gemini-2.0-flash

    # Primary Logic/Code Model
    LLM_PROVIDER=openai
    OPENAI_API_KEY=your_openrouter_key
    OPENAI_BASE_URL=https://aipipe.org/openrouter/v1
    OPENAI_MODEL=gpt-5-nano
    OPENAI_FALLBACK_MODEL=claude 4 opus

    # System Settings
    DEBUG=True
    TIMEOUT_SECONDS=180
    ```

### Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## 🔌 API Usage

### Solve Quiz Endpoint

**URL:** `/quiz`
**Method:** `POST`

**Request Payload:**

```json
{
  "email": "student@example.com",
  "secret": "your_quiz_secret",
  "url": "https://example.com/quiz-start-url"
}
```

**Response:**

*   `200 OK`: Quiz solving process initiated and completed.
*   `403 Forbidden`: Invalid secret provided.
*   `500 Internal Server Error`: Application error during processing.

## 🛠️ Design Choices

*   **Dual-LLM Strategy:** We utilize Gemini 2.0 Flash for its superior speed and native multimodal capabilities (audio/vision), while offloading complex logical reasoning and code generation to GPT-5-nano (with Claude 4 Opus as a safety net). This ensures the best tool is used for each specific sub-task.
*   **Dynamic Code Execution:** Instead of hardcoding logic for every possible data question, the agent generates and executes Pandas code on the fly, allowing it to handle unforeseen data analysis queries flexibly.
*   **Robustness:** The system is designed to handle network flakes and API errors gracefully, returning default values or retrying where appropriate to maximize the chance of a successful submission within the time limit.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
