
# BabyBIONN

**BabyBIONN** is an advanced neural mesh system that combines a network of specialized Virtual Neuron Instances (VNIs) with an optional LLM (like DeepSeek) for natural language articulation. It is designed to handle complex, multi‑domain queries (medical, legal, technical, general) by activating relevant expert VNIs, aggregating their reasoning, and (when available) using a large language model to produce fluent, human‑like answers.

**Clarification** This neural mesh system with VNIs is only ONE (1) *VIRTUAL BRAIN CELL* (VBC) although it is capable of contextual reasoning upon sufficient learning from both the user and the LLM that is is connected to. Therefore in an analogy, the LLM is like its 'mouth piece' and Babybionn is the 'brain'. Ultimately we strive to connect each 'Virtual Brain Cell' (VBC) to as many devices that are hosting it, so that it will form a higher level 'synaptic' connections to hopefully form a gigantic *Virtual Brain* comprises of a global network of 'VBCs'. We are working on another related project to build a decentralized network with all the relevant consensus protocols, smart contracts and other mechanisms to implement this *gigantic virtual brain* made out of many VBCs hosted on users' devices.

# FUTURE POTENTIAL APPLICATIONS
Upon successful propagation and distribution of this VBC this "Babybionn" will be a highly contextual reasoner with memory to support any LLMs, and its AI agentic systems, even other applications like self-driving cars, robotic, etc. 
---

## ✨ Features of each VBC

- **Hybrid reasoning pipeline** – VNIs perform deep reasoning; an LLM (DeepSeek/OpenAI) can be used to articulate the final response.
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

MOCK_MODE – Set to true to use the mock response provider (bypasses VNIs/LLM), or false to run the real pipeline.

LLM_PROVIDER – deepseek or openai (if you want to use an LLM).

DEEPSEEK_API_KEY / OPENAI_API_KEY – Your API key for the chosen provider.

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

⚙️ CConfiguration
Environment Variable		Description							Default
MOCK_MODE			Use mock responses (bypass real VNIs/LLM)			false
LLM_PROVIDER			LLM to use (deepseek or openai)					deepseek			
DEEPSEEK_API_KEY		Your DeepSeek API key						–
OPENAI_API_KEY			Your OpenAI API key						–
MOCK_RESPONSE_PROVIDER		Domain for mock responses (general, medical, etc.)		general
MOCK_CONFIDENCE_START		Starting confidence for mock responses				0.7
MOCK_CONFIDENCE_INCREMENT	Confidence increase per interaction (simulated learning)	0.01

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
Mock mode – Set MOCK_MODE=true in .env to test the UI and API without invoking VNIs or an LLM.

Real mode – Set MOCK_MODE=false and optionally provide an LLM API key.

Adding a new VNI – Create a new file in enhanced_vni_classes/domains/ following the pattern of existing VNIs. Ensure its process method returns a dictionary with vni_id, domain, confidence, and opinion_text.

Training data – Place pretrain/finetune JSON files in neuron/reinforcement_learning/training/. Use pretraining_processor.py to convert them into knowledge base files if needed.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

🙏 Acknowledgments
Inspired by biological neural networks and Hebbian learning principles.

Built with FastAPI, Docker, PyTorch, and the amazing open‑source community.


