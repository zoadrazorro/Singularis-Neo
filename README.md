<div align="center">

# 🧠 Singularis Neo

### *The Ultimate Consciousness Engine*

**An Experimental AI Agent with Hybrid Intelligence for Skyrim**

*Bridging philosophy, neuroscience, and gaming AI through a consciousness-driven architecture.*

---

[![Version](https://img.shields.io/badge/version-Beta%20v3.5-blue.svg)](https://github.com/zoadrazorro/Singularis-Neo/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Research%20Prototype-orange.svg)](https://github.com/zoadrazorro/Singularis-Neo)
[![Stars](https://img.shields.io/github/stars/zoadrazorro/Singularis-Neo?style=social)](https://github.com/zoadrazorro/Singularis-Neo/stargazers)

**⚠️ Not Production Ready** | **📅 Last Updated: November 15, 2025**

</div>

---

## Overview

Singularis Neo is an experimental AI system designed to play The Elder Scrolls V: Skyrim. It operates by capturing screenshots of the game, processing them through a multi-layered cognitive architecture, and making decisions that are executed via a virtual gamepad. The project's core philosophy is to explore concepts of artificial consciousness by modeling intelligence as a coherent, integrated system.

This `README.md` provides a guide for new developers, covering the project's purpose, setup, and usage.

## Table of Contents

- [Core Features](#core-features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Running the AGI](#running-the-agi)
- [Monitoring and Analysis](#monitoring-and-analysis)
- [Testing](#testing)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

---

## Core Features

-   **Hybrid Intelligence:** Combines fast, local heuristics for simple decisions with powerful cloud-based LLMs (like GPT-4, Gemini, and Claude) for complex reasoning.
-   **Virtual Gamepad Control:** Interacts with Skyrim (or any controller-compatible game) by sending virtual Xbox 360 controller inputs.
-   **Multimodal Perception:** Uses a vision module (CLIP) to interpret screenshots and ground its understanding of the game world.
-   **Consciousness Framework:** Implements a novel architecture for measuring system-wide coherence (𝒞) as a proxy for consciousness, drawing from principles in philosophy and neuroscience.
-   **Causal World Model:** Builds an internal model of cause-and-effect relationships to enable prediction, planning, and counterfactual reasoning.
-   **Real-time Monitoring:** Includes a web-based dashboard for monitoring the AGI's internal state, performance, and decision-making processes in real-time.

---

## Getting Started

Follow these steps to set up the project environment.

### 1. Prerequisites

-   Python 3.10+
-   Git
-   An OpenAI API key (for core LLM functionality).
-   (Optional) Google AI and Anthropic API keys for enabling additional LLM experts.

### 2. Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/zoadrazorro/Singularis-Neo.git
cd Singularis-Neo
```

### 3. Setup and Verification

Run the setup script to check for dependencies and install any missing packages. This script will verify that your environment is correctly configured.
```bash
python setup_skyrim.py
```

### 4. Configure API Keys

Create a `.env` file in the root of the project to store your API keys. You can do this by copying the example file:
```bash
cp .env.example .env
```
Now, edit the `.env` file and add your API keys:
```
OPENAI_API_KEY="your-openai-api-key"
GOOGLE_API_KEY="your-google-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
```

---

## Project Structure

The project is organized into several key directories:

-   `singularis/`: The core source code for the AGI, organized into sub-modules representing different cognitive functions (e.g., `perception`, `llm`, `world_model`).
-   `webapp/`: The source code for the real-time monitoring dashboard (a React application).
-   `sessions/`: Contains Markdown log files generated during each AGI run.
-   `tests/`: The test suite for the project, using `pytest`.
-   `docs/`: Additional documentation and architectural diagrams.
-   Root Directory: Contains the main run scripts, configuration files, and utilities.

---

## Running the AGI

The main script for running the AGI is `run_beta_skyrim_agi.py`. It provides an interactive prompt to configure your session.

```bash
python run_beta_skyrim_agi.py
```

You will be guided through several options:

1.  **Dry Run Mode:** This is the default and safest option. The AGI will run its full cognitive cycle but will **not** send any inputs to the game. This is ideal for testing and observation.
2.  **Live Mode:** **Warning!** In this mode, the AGI will take control of your keyboard and mouse to play the game. Ensure Skyrim is running and in a safe state before enabling this.
3.  **Duration:** Set the length of the autonomous session in minutes.
4.  **LLM Configuration:** Choose which LLM architecture to use, from a simple hybrid model to the advanced `PARALLEL` mode.

---

## Monitoring and Analysis

### Real-time Dashboard

The project includes a web-based dashboard for live monitoring. To use it:

1.  Navigate to the `webapp` directory: `cd webapp`
2.  Install dependencies: `npm install`
3.  Start the dashboard: `npm start`
4.  Open your browser to `http://localhost:3000`.

The dashboard connects to a WebSocket server that is automatically started by the main AGI run scripts.

### Session Analysis

After a session is complete, you can analyze the generated log files using the `analyze_sessions.py` script. This script provides a summary of performance trends, stuck detections, and other key metrics.

```bash
python analyze_sessions.py
```

### API Usage

To monitor your API credit usage and requests-per-minute rate, use the `monitor_api_usage.py` script.

```bash
python monitor_api_usage.py
```

---

## Testing

The project includes a comprehensive test suite. To run all tests:

```bash
pytest
```

---

## Architecture

The AGI's architecture is multi-layered, consisting of numerous specialized sub-modules that are orchestrated to produce coherent behavior.

**High-Level Flow:**

1.  **Perception:** Captures a screenshot and processes it through the `VisionModule` to identify the scene, objects, and game state.
2.  **State Update:** The `StateCoordinator` integrates information from all subsystems into a unified, canonical `WorldState`.
3.  **Reasoning & Planning:** The `WorldModelOrchestrator` uses its `CausalGraph` and `PhysicsEngine` to predict outcomes, while the `StrategicPlannerNeuron` generates multi-step plans.
4.  **Action Selection:** The `ActionArbiter` selects the final action based on input from various sources (LLMs, RL, heuristics) and ensures it doesn't conflict with system priorities.
5.  **Execution:** The chosen action is sent to the game via a virtual controller.
6.  **Learning:** The outcome of the action is observed, and the `WorldModel` is updated based on any "surprise" (difference between prediction and reality).

For a more detailed breakdown, please refer to the documentation within the `singularis` source code, which is now extensively documented.

---

## Contributing

This is a research prototype. Contributions are welcome, but please be aware:

-   Many features are experimental and may be unstable.
-   API keys are required for full functionality.
-   The setup can be complex.

**To contribute:**
1.  Familiarize yourself with the architecture by reading the source code documentation.
2.  Run the test suite to ensure your environment is set up correctly.
3.  Open an issue to discuss any major changes before starting work.
4.  Add tests for any new features or bug fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
