# BabyBIONN – Layer 0 Contextual Intelligence

**BabyBIONN is not another LLM.** It is the fundamental reasoning layer that gives LLMs context, memory, understanding, and continuity. Think of it as the **"operating system for intelligence"** – the Layer 0 that makes AI systems feel alive, coherent, and trustworthy.

Each BabyBIONN instance is a single **Virtual Brain Cell (VBC)**. When connected to an LLM (like DeepSeek), it acts as the **brain** while the LLM serves as the **mouth**. Our ultimate vision is to connect millions of VBCs hosted on devices worldwide into a gigantic, decentralized **Virtual Brain** – a global network of contextual reasoners with memory, secured by blockchain‑inspired consensus protocols. This opens doors for applications far beyond chatbots: self‑driving cars, robotics, agentic systems, and more.

> **Current focus**: A single‑node neural mesh that dramatically reduces hallucinations and provides true contextual reasoning.

---

## ✨ Features of Each VBC
- **Hybrid reasoning pipeline** – VNIs perform deep reasoning; an LLM (DeepSeek/OpenAI) articulates the final response.
- **Multi‑domain VNIs** – Specialized modules for medical, legal, technical, and general queries.
- **Hebbian learning** – Connections between VNIs strengthen or weaken based on co‑activation and outcome quality.
- **Conflict detection & consensus** – The aggregator identifies disagreements and computes consensus levels.
- **Greeting handler** – Simple greetings are caught early for a snappy response.
- **Dockerized** – Easy setup with Docker Compose.
- **Mock mode** – Develop and test without an LLM or actual VNI reasoning by using a mock response provider.

---

## 🧠 Architecture Overview
User Query → Neural Mesh (activates VNIs) → Aggregator → (optional) LLM → Final Response

text

- **VNIs** (Virtual Networked Intelligences) – Domain‑expert modules that return an opinion (text) and a confidence score.
- **Neural Mesh** – Routes the query to the most relevant VNIs based on keyword matching and learned patterns.
- **Aggregator** – Collects VNI outputs, detects conflicts, calculates consensus, and optionally calls an LLM.
- **LLM Gateway** – If enabled and an API key is provided, the aggregator sends a prompt built from the VNIs' reasoning to an LLM (DeepSeek/OpenAI) and returns the generated text.
- **Memory** – Stores past interactions and learned patterns (supports FAISS for fast similarity search).

---

## 📦 Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- (Optional) An API key for [DeepSeek](https://platform.deepseek.com/) or [OpenAI](https://platform.openai.com/)
- Git (to clone the repository)

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/deyvitt/bionn-demo.git
cd bionn-demo
2. Configure environment variables
Copy the example environment file and edit it:

bash
cp .env.example .env
Edit .env with your preferred settings. The most important variables are:

Variable	Description	Default
MOCK_MODE	Use mock responses (bypass real VNIs/LLM)	false
LLM_PROVIDER	LLM to use (deepseek or openai)	deepseek
DEEPSEEK_API_KEY	Your DeepSeek API key	–
OPENAI_API_KEY	Your OpenAI API key	–
MOCK_RESPONSE_PROVIDER	Domain for mock responses (general, medical, etc.)	general
MOCK_CONFIDENCE_START	Starting confidence for mock responses	0.7
MOCK_CONFIDENCE_INCREMENT	Confidence increase per interaction (simulated learning)	0.01
3. Build and start the containers
bash
docker-compose -f docker-compose.dev.yml up --build
The application will be available at http://localhost:8002.
The main chat interface is at http://localhost:8002/chat (or the root /).

🎮 Usage
Chat Interface
Open your browser and go to http://localhost:8002. Type a message and press Enter.

If MOCK_MODE=true, you will receive a canned response.

If MOCK_MODE=false and an LLM is configured, the system will run the VNIs and then call the LLM to generate the answer. If the LLM call fails, it falls back to a template‑based response.

API Endpoints
POST /api/chat – Send a JSON {"message": "your query", "session_id": "optional"}

GET /api/health – Health check

GET /api/config/llm-provider – Get current LLM provider

POST /api/config/llm-provider – Update LLM provider (JSON {"provider": "deepseek"})

🐳 Running with Docker (Official Image)
1. Pull the image
bash
docker pull deyvitt69/babybionn:latest
2. Choose your mode
🎭 Test mode (no API key needed)
bash
docker run -d -p 8002:8002 -e MOCK_MODE=true --name babybionn deyvitt69/babybionn:latest
🧠 Full reasoning mode (with DeepSeek API key)
bash
docker run -d -p 8002:8002 \
  -e MOCK_MODE=false \
  -e DEEPSEEK_API_KEY=your-actual-key-here \
  --name babybionn \
  deyvitt69/babybionn:latest
3. Access the chat interface
Open http://localhost:8002 in your browser.

4. Managing the container
Stop: docker stop babybionn

Remove: docker rm babybionn

Switch modes: stop, remove, then run with new environment variables.

5. (Optional) Persist learned data
bash
docker run -d -p 8002:8002 \
  -e MOCK_MODE=false \
  -e DEEPSEEK_API_KEY=your-key \
  -v ./babybionn_data:/app/vni_data \
  --name babybionn \
  deyvitt69/babybionn:latest
6. View logs
bash
docker logs babybionn
📁 Project Structure (High‑Level)
text
babybionn-demo/
├── docker-compose.dev.yml          # Docker compose for development
├── Dockerfile                       # Main Dockerfile
├── main.py                          # FastAPI application entry point
├── config.py                        # Configuration loader
├── enhanced_vni_classes/            # All domain VNIs
│   ├── domains/
│   │   ├── medical.py
│   │   ├── legal.py
│   │   ├── technical.py
│   │   ├── general.py
│   │   └── dynamic_vni.py
│   └── modules/                     # Shared modules (knowledge_base, etc.)
├── neuron/                           # Core neural mesh components
│   ├── aggregator.py                 # Main aggregator with Hebbian learning
│   ├── vni_memory.py                 # Memory system
│   ├── vni_storage.py                # Storage manager (thumbdrive persistence)
│   ├── demoHybridAttention.py         # Attention mechanism
│   ├── smart_activation_router.py     # Routing logic
│   └── reinforcement_learning/        # Training data and pretraining processor
├── new/                               # New API and compatibility layer
├── knowledge_base/                    # Source knowledge files (JSON)
├── llm_Gateway.py                     # LLM client wrapper
├── template_engine.py                  # Template fallback when LLM fails
├── Babybionn_integration.py            # Integration with the main system
└── .env.example                        # Example environment variables
🧪 Development
Mock mode – Set MOCK_MODE=true to test the UI and API without invoking VNIs or an LLM.

Real mode – Set MOCK_MODE=false and optionally provide an LLM API key.

Adding a new VNI – Create a new file in enhanced_vni_classes/domains/ following the pattern of existing VNIs. Ensure its process method returns a dictionary with vni_id, domain, confidence, and opinion_text.

Training data – Place pretrain/finetune JSON files in neuron/reinforcement_learning/training/. Use pretraining_processor.py to convert them into knowledge base files if needed.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

#____________________________________________________________________________
# IMPORTANT NOTE!
When you're ready to run with DeepSeek (after adding credits), use this command:

bash
docker run -d -p 8002:8002 \
  -e MOCK_MODE=false \
  -e DEEPSEEK_API_KEY=sk-c3d5b5e316f947dcb66907295aa6681b \
  --name babybionn \
  deyvitt69/babybionn:latest
If you already have a container running with a different name:
bash
# Stop and remove the old container
docker stop babybionn
docker rm babybionn

# Then run the new one with DeepSeek
docker run -d -p 8002:8002 -e MOCK_MODE=false -e DEEPSEEK_API_KEY=your-key --name babybionn deyvitt69/babybionn:latest
Using an environment file (optional)
You can also store your key in a file (e.g., deepseek.env) to avoid typing it each time:

bash
# deepseek.env
MOCK_MODE=false
DEEPSEEK_API_KEY=sk-[your-deepseek-api key] or if you use OPEN AI put in their api key
Then run with:

bash
docker run -d -p 8002:8002 --env-file deepseek.env --name babybionn deyvitt69/babybionn:latest
The .env file from your local development is not used by Docker Hub images – you must pass variables explicitly.
#_________________________________________________________________________________

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

🙏 Acknowledgments
Inspired by biological neural networks and Hebbian learning principles.
Built with FastAPI, Docker, PyTorch, and the amazing open‑source community.
