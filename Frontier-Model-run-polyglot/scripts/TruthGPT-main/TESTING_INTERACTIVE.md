# 🧪 Testing TruthGPT Interactive Command Center

This guide explains how to verify the new interactive menu and the System 5.9 terminal experience.

## 🏁 Prerequisites
Ensure you have the dependencies installed:
```bash
pip install rich typer httpx torch
```

## 🛠️ Step 1: Launch the Command Center
Run the following from the root directory:
```bash
run.bat
```
Alternatively, navigate to `optimization_core` and run:
```bash
python main.py
```

## 📋 Step 2: Verification Checklist

### 1. The Visual Experience
- [ ] Verify the **ASCII Banner** appears correctly.
- [ ] Verify the **Categorical Menu** (Core, Agents, Research, System, Health) is displayed in a panel.

### 2. Core Functionality (Test these options)
- [ ] **Option 14 (Version)**: Should show a blue panel with version info.
- [ ] **Option 6 (List Papers)**: Should display a table of research papers.
- [ ] **Option 5 (List Agents)**: Should show active swarm agents.
- [ ] **Option 10 (Health)**: Should perform a status check on the API.

### 3. CLI Mode Pass-through
Verify that CLI arguments still work directly:
```bash
python main.py version
```
*Expected: Shows the version panel and exits immediately (no menu).*

### 4. Interactive Flow
1. Select **Option 1** (Inference).
2. Enter a test prompt.
3. Observe the loading spinner while it processes.
4. Return to the menu and select **0** to Exit.

## 🦞 Integration with OpenClaw
The system is now fully integrated with the OpenClaw ecosystem. You can use the `openclaw` command directly if configured in your environment:
```bash
openclaw --help
```

---
*Built with ❤️ for Industrial-Grade AI Optimization.*
