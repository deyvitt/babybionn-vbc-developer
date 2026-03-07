# BabyBIONN – Layer 0 Contextual Intelligence

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
Copyright (c) 2026, BabyBIONN Contributors

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

- **VNIs** (Virtual Neuron Instances) – Domain‑expert modules that return an opinion (text) and a confidence score.
- **Neural Mesh** – Routes the query to the most relevant VNIs based on keyword matching and learned patterns.
- **Aggregator** – Collects VNI outputs, detects conflicts, calculates consensus, and optionally calls an LLM.
- **LLM Gateway** – If enabled and an API key is provided, the aggregator sends a prompt built from the VNIs' reasoning to an LLM (DeepSeek/OpenAI) and returns the generated text.
- **Memory** – Stores past interactions and learned patterns (supports FAISS for fast similarity search).

---

## 🔐 Proprietary Aggregator Binary

The **aggregator** (which contains the Hebbian learning engine, consensus algorithms, conflict detection, and response synthesis) is **proprietary** and distributed as a compiled binary. It is the only component required to participate in the global BabyBIONN network and is responsible for:

- Generating cryptographic identity (key pairs).
- Signing all network messages (responses, votes).
- Participating in consensus protocols.

The rest of the codebase (VNIs, managers, P2P stubs, utilities) is open source under the **MPL 2.0** license.

To obtain the aggregator binary:

1. Clone the open‑source repository from GitHub.
2. Download the signed binary from [https://downloads.babybionn.net](https://downloads.babybionn.net) (free for network participants; registration may be required).
3. Place the binary in the `neuron/` directory of your cloned repository (e.g., `neuron/aggregator_core.so` or `.pyd`).
4. Run your VBC as usual – the open‑source code will automatically detect and use the binary.

> **Note**: Without the aggregator binary, your VBC will run in a limited offline mode and cannot join the global network.

---

## 📦 Prerequisites
- [Docker Desktop](https://docs.docker.com/get-docker/) (for Windows, Mac, or Linux)
- At least **4 GB RAM** (8 GB recommended for full LLM integration)
- 2 CPU cores (more for heavy usage)
- ~5 GB free disk space (more if you plan to store training data)
- (Optional) An API key for [DeepSeek](https://platform.deepseek.com/) or [OpenAI](https://platform.openai.com/)
- Git (if you want to clone the repository for development)

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
MOCK_RESPONSE_PROVIDER	Domain for mock responses (general, medical)	general
MOCK_CONFIDENCE_START	Starting confidence for mock responses	0.7
MOCK_CONFIDENCE_INCREMENT	Confidence increase per interaction	0.01
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
Method	Endpoint	Description
POST	/api/chat	Send a message {"message": "your query", "session_id": "optional"}
GET	/api/health	Health check
GET	/api/config/llm-provider	Get current LLM provider
POST	/api/config/llm-provider	Update LLM provider {"provider": "deepseek"}
🚀 Getting Started – Run BabyBIONN with Docker Desktop
1. Install Docker Desktop
If you don’t have Docker Desktop yet, download and install it from docker.com. After installation, make sure Docker is running (you should see the Docker icon in your system tray).

2. Pull the BabyBIONN image
Open a terminal (Command Prompt, PowerShell, or Terminal) and run:

bash
docker pull deyvitt69/babybionn:latest
Alternatively, you can search for "babybionn" in Docker Desktop’s Images view and pull it from there.

3. Run the container
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
🔐 Set a custom admin password (recommended)
The admin panel is protected by a password. By default, the password is babybionn_admin_2024 – change it in production by passing the ADMIN_PASSWORD environment variable:

bash
docker run -d -p 8002:8002 \
  -e ADMIN_PASSWORD="your_secure_password" \
  -e MOCK_MODE=false \
  -e DEEPSEEK_API_KEY=your-key \
  --name babybionn \
  deyvitt69/babybionn:latest
4. Access the chat interface
Open your browser and go to http://localhost:8002. You'll see the BabyBIONN chat interface.

Click the Admin tab to log in with the password you set (or the default) and upload training data.

Start chatting! If you didn't provide an API key, the system will run in mock mode and use canned responses.

5. Managing the container
Stop: docker stop babybionn

Remove: docker rm babybionn

View logs: docker logs babybionn

Persist learned data (mount a volume):

bash
docker run -d -p 8002:8002 \
  -e ADMIN_PASSWORD="your_password" \
  -v ./babybionn_data:/app/vni_data \
  --name babybionn \
  deyvitt69/babybionn:latest
🖥️ Minimum Hardware Specifications
Component	Minimum	Recommended
RAM	4 GB	8 GB
CPU	2 cores	4 cores
Disk	5 GB	10 GB
Network	Broadband internet	Stable connection for LLM APIs
These specs assume you run BabyBIONN in mock mode or with a small local knowledge base. If you plan to use large language models (via API), the system itself is lightweight; the API calls depend on your internet connection.

# To start back babybionn in the docker:
First access and login to your docker desktop
Then ensure you already run the docker : 
docker run -d -p 8002:8002 --name babybionn deyvitt69/babybionn:latest

wait for response, if you get 
"docker: Error response from daemon: Conflict. The container name "/babybionn" is already in use by container "d154bad7b5d868353b4fbd12a4e25d31eb04311d753ebcdbe6e04e03b40dc08e". You have to remove (or rename) that container to be able to reuse that name.

Run 'docker run --help' for more information

access your ubuntu CLI key in: 
docker start babybionn, then /> enter

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
│   ├── aggregator.py                 # Stub that loads the proprietary binary
│   ├── vni_memory.py                 # Memory system
│   ├── vni_storage.py                # Storage manager
│   ├── demoHybridAttention.py         # Attention mechanism (open source)
│   ├── smart_activation_router.py     # Routing logic (open source)
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

🌐 The Vision: A Decentralized Virtual Brain
Imagine millions of VBCs running on devices worldwide – personal computers, servers, edge devices – all interconnected in a peer‑to‑peer (P2P) network. Each VBC maintains its own local knowledge, learned patterns, and Hebbian connections, but can also collaborate with others to solve complex problems, share insights, and reach consensus. This creates a resilient, scalable, and truly decentralized intelligence layer.

🧠 How a Single VBC Works Today
Currently, each BabyBIONN instance is a standalone reasoning engine. It has:

Its own neural mesh (VNIs).

Local memory and learning (Hebbian plasticity).

An optional LLM gateway for articulation.

When you run the Docker container, you get one isolated VBC.

🔗 Connecting VBCs – A Proposed Architecture
To build the decentralized network, we need additional layers:

Peer‑to‑Peer Communication Layer
Each VBC must be able to discover and communicate with other VBCs. Technologies like libp2p (used by IPFS) provide a solid foundation for P2P networking, including peer discovery, NAT traversal, and secure channels.

Consensus & Coordination Protocol
When multiple VBCs are asked the same question, they need to agree on a final answer. A consensus mechanism – inspired by blockchain – could be used:

Voting‑based consensus: VBCs submit their opinions with confidence scores; a supermajority threshold determines the final answer.

Proof of Work/Stake: To prevent spam and reward useful contributions, nodes might need to stake reputation or compute power.

Byzantine Fault Tolerance (BFT): To handle malicious or faulty nodes.

Query Routing & Load Balancing
A query could be broadcast to a subset of VBCs based on their expertise (domain), reputation, or geographic proximity. This requires a distributed hash table (DHT) or similar discovery mechanism.

Distributed Knowledge Sharing
VBCs could exchange learned patterns (Hebbian weights, memory embeddings) in a privacy‑preserving way, allowing the network to learn collectively without centralizing data. Techniques like federated learning or secure aggregation could be applied.

Incentives & Tokenomics (Optional)
To encourage participation, you could introduce a cryptocurrency token that rewards nodes for contributing high‑quality reasoning, maintaining uptime, or participating in consensus.

🛠️ Next Steps for Implementation
This is a long‑term roadmap – the current BabyBIONN focuses on a single VBC. But if you’d like to start moving toward this vision, consider:

Add a P2P module – Experiment with libp2p or a simpler WebSocket‑based mesh.

Design a simple consensus protocol – Start with a basic voting mechanism among a small set of trusted nodes.

Extend the aggregator – Modify the UnifiedAggregator to handle responses from remote VBCs in addition to local ones.

Create a node registry – Use a DHT (like Kademlia) to let nodes find each other by domain or capability.

Think about incentives – Even without tokens, you can build reputation scores based on past performance.

💡 Contributions Welcome
This is a community‑driven project. If you're excited about building the decentralized network, you can:

Open an issue or discussion on the GitHub repository to share ideas.

Start prototyping a P2P layer and share your progress.

Collaborate with others who are interested in distributed AI.

🤝 Contributing
Contributions are welcome! By submitting a pull request, you agree that your contributions will be licensed under the MPL 2.0. Please ensure that your changes do not introduce dependencies that are incompatible with this license.

📄 License
The BabyBIONN project is dual‑licensed:

All code except the aggregator core is open source under the Mozilla Public License 2.0 (MPL 2.0). This includes VNIs, managers, utilities, and the P2P networking layer. You are free to use, modify, and distribute these parts under the terms of the MPL 2.0.

The aggregator core (containing the Hebbian learning engine, consensus algorithms, conflict detection, and response synthesis) is proprietary and distributed as a compiled binary. It is not open source and requires a separate download (free for network participants).

For more details, see the LICENSE file.
