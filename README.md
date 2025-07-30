# Multi-Agent Mobile QA System

An LLM-powered multi-agent system that functions as a full-stack mobile QA team, built on top of Agent-S architecture and AndroidWorld. This system simulates a complete QA workflow for Android applications using collaborative AI agents.

## 🚀 Overview

This project implements a modular multi-agent architecture where specialized agents work together to perform comprehensive QA testing on Android applications. The system leverages Large Language Models (LLMs) and learned policies to intelligently test mobile UI applications.

### Key Features

- **Modular Agent Architecture**: Four specialized agents working collaboratively  
- **Real Android Environment Integration**: Tests on simulated Android environments
- **LLM-Powered Intelligence**: Uses GPT-4 for intelligent test planning and verification
- **Message Bus Communication**: Event-driven architecture for agent coordination
- **Comprehensive Reporting**: Detailed test results with bug detection metrics
- **UI Inspection**: Advanced UI hierarchy analysis for robust testing

## 🏗️ Architecture

The system consists of four main agents:

1. **Planner Agent** (`agents/planner_agent.py`) - Decomposes high-level QA goals into actionable subgoals
2. **Executor Agent** (`agents/executor_agent.py`) - Executes actions in the Android environment with UI inspection
3. **Verifier Agent** (`agents/verifier_agent.py`) - Validates expected outcomes and detects bugs
4. **Supervisor Agent** (`agents/supervisor_agent.py`) - Reviews test episodes and provides improvement recommendations

### Communication Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Planner   │────▶│  Executor   │────▶│  Verifier   │────▶│ Supervisor  │
│    Agent    │     │    Agent    │     │    Agent    │     │    Agent    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       ▲                    │                    │                    │
       └────────────────────┴────────────────────┴────────────────────┘
                          Message Bus (Event-Driven)
```

## 📋 Requirements

- Python 3.8+
- OpenAI API key (for GPT-4 integration)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/shrsai123/MultiAgentQASystem.git
cd MultiAgentQASystem
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from android_multiagent import MultiAgentQASystem

async def main():
    # Initialize the system
    qa_system = MultiAgentQASystem(
        task_name="settings_wifi",  # Choose from available tasks
        config={
            "max_retries": 3,
            "action_delay": 0.5
        }
    )
    
    # Initialize system components
    if not await qa_system.initialize():
        print("Initialization failed")
        return
    
    try:
        # Run a test
        test_goal = "Test turning Wi-Fi on and off"
        test_result = await qa_system.run_test(test_goal)
        
        # Display results
        print(f"Test Status: {'PASSED' if test_result['passed'] else 'FAILED'}")
        print(f"Bug Detected: {'YES' if test_result['bug_detected'] else 'NO'}")
        
        # Generate evaluation report
        report_path = await qa_system.supervisor.generate_report([test_result])
        print(f"Report saved to: {report_path}")
        
    finally:
        await qa_system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Running the System

```bash
# Run the main multi-agent system
python android_multiagent.py
```

### Available Test Tasks

The system supports the following Android tasks:
- `settings_wifi` - Test Wi-Fi settings functionality
- `clock_alarm` - Test alarm functionality in Clock app
- `email_search` - Test email search functionality

## 📁 Project Structure

```
MultiAgentQASystem/
├── agents/
│   ├── __pycache__/
│   ├── __init__.py            # Agents module initialization
│   ├── executor_agent.py      # UI interaction execution
│   ├── planner_agent.py       # Task planning and decomposition
│   ├── supervisor_agent.py    # Test review and feedback
│   └── verifier_agent.py      # Result verification
├── android_integration/       # Android environment integration
├── android_world/            # AndroidWorld environment files
├── core/
│   ├── __pycache__/
│   ├── __init__.py           # Core module initialization
│   ├── agent_base.py         # Base agent class
│   └── message_bus.py        # Event-driven message bus
├── outputs/                  # Output directory for logs and reports
├── android_multiagent.py     # Main multi-agent system orchestrator
└── README.md                 # This file
```

### Key Files

- **`android_multiagent.py`** - Main entry point containing the `MultiAgentQASystem` class
- **`core/agent_base.py`** - Base class for all agents with message bus integration
- **`core/message_bus.py`** - Event-driven communication system between agents

## 🔧 Configuration

The system can be configured through the initialization parameters:

```python
config = {
    "max_retries": 3,           # Maximum retry attempts for actions
    "action_delay": 0.5,        # Delay between actions in seconds
    "log_level": "INFO",        # Logging level
    "screenshot_enabled": True,  # Enable screenshot capture
    "ui_inspection": True       # Enable UI hierarchy inspection
}
```

## 📊 Output Formats

### Episode Log Format

Test episodes are saved as JSON files in `outputs/logs/` with the naming pattern: `episode_{task_name}_{timestamp}.json`

Example structure:
```json
{
  "task_name": "settings_wifi",
  "goal": "Test turning Wi-Fi on and off",
  "start_time": 1753882781.7695515,
  "steps": [
    {
      "subgoal": "Navigate to home screen",
      "execution_result": {
        "ui_hierarchy": {
          "nodes": [/* UI elements */]
        },
        "screenshot_path": "outputs/screenshots/settings_wifi_step1_1753882807721.png",
        "current_step": 1,
        "task_name": "settings_wifi",
        "timestamp": 1753882807.7338269,
        "episode_time": 25.967280387878418
      },
      "timestamp": 1753882808.2933388,
      "execution_success": true,
      "verification_result": {
        "passed": true,
        "bug_detected": false,
        "error_message": null
      }
    }
  ],
  "logs": [],
  "passed": true,
  "bug_detected": false,
  "config": {
    "task_name": "settings_wifi",
    "max_retries": 3,
    "action_delay": 0.5
  },
  "initial_plan": ["Navigate to home screen", "Wait for system to stabilize"],
  "end_time": 1753882818.4730384,
  "duration": 36.703486919403076,
  "message_history": [/* Message bus communication logs */],
  "supervisor_feedback": {
    "prompt_improvements": [],
    "plan_issues": [],
    "coverage_suggestions": [],
    "execution_issues": [],
    "verification_issues": [],
    "overall_rating": "good",
    "summary": "Detailed assessment of the test episode"
  }
}
```

### Evaluation Report Format

Evaluation reports are saved as `evaluation_report.json` in `outputs/logs/`

Example structure:
```json
{
  "summary": "Overall test suite assessment",
  "bug_detection_accuracy": {
    "bugs_found": 1,
    "detection_rate": "100.0%",
    "steps_analyzed": 2,
    "steps_with_bugs": 2,
    "assessment": "Detailed analysis of bug detection effectiveness"
  },
  "agent_performance": {
    "planner": "Planning phase assessment",
    "executor": "Execution phase assessment",
    "verifier": "Verification phase assessment"
  },
  "key_findings": [
    "Major discoveries from testing"
  ],
  "recommendations": [
    "Actionable improvement suggestions"
  ],
  "next_steps": [
    "Future testing directions"
  ]
}
```

### Key Metrics Explained

- **Detection Rate**: Percentage of test steps where bugs were identified
- **Steps Analyzed**: Total number of UI interaction steps executed
- **Episode Time**: Time taken from start to completion of a test
- **Message History**: Complete log of inter-agent communication via message bus

## 📈 Features

### Message Bus Architecture
- Asynchronous communication between agents
- Event-driven coordination
- Message history and statistics tracking

### UI Inspection Capabilities
- Real-time UI hierarchy analysis
- Element detection and matching
- Adaptive action execution based on UI state

### Error Recovery
- Automatic retry with exponential backoff
- Dynamic plan adaptation on failures
- Comprehensive error logging and analysis

### Reporting System
- Detailed test execution logs
- Bug detection metrics
- Performance analytics
- Actionable recommendations


### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📚 Documentation

### Agent Documentation

Each agent has specific responsibilities:

- **Planner Agent**: Uses GPT-4 to break down high-level goals into executable steps
- **Executor Agent**: Interacts with Android UI, performs actions, handles UI inspection
- **Verifier Agent**: Validates outcomes using LLM reasoning and heuristics
- **Supervisor Agent**: Reviews complete test sessions and provides feedback

### Message Types

The system uses various message types for coordination:
- `PLAN_REQUEST` / `PLAN_RESPONSE`
- `EXECUTION_REQUEST` / `EXECUTION_RESPONSE`
- `VERIFICATION_REQUEST` / `VERIFICATION_RESPONSE`
- `SUPERVISION_REQUEST` / `SUPERVISION_RESPONSE`
- `SYSTEM_ERROR` / `SYSTEM_STATUS`

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Error**
   - Ensure your API key is correctly set in `.env`
   - Check API rate limits and quotas

2. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Android Environment Issues**
   - Ensure task name is valid (see available tasks)
   - Check logs in `outputs/` directory for details

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [Agent-S](https://github.com/simular-ai/Agent-S) architecture
- Uses [AndroidWorld](https://github.com/google-research/android_world) for environment simulation
- Powered by OpenAI's GPT-4 for intelligent reasoning

---

**Note**: This is a research project demonstrating multi-agent systems for mobile QA for the QualGent Coding Challenge.