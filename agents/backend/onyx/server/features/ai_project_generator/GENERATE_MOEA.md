# MOEA Project Generation Guide

This guide explains how to generate the MOEA (Multi-Objective Evolutionary Algorithm) optimization system project.

## Project Description

The MOEA project is a comprehensive system for solving optimization problems with multiple conflicting objectives. It includes:

- Support for various MOEA algorithms (NSGA-II, NSGA-III, MOEA/D, SPEA2)
- Visualization of Pareto fronts
- Performance metrics calculation (hypervolume, IGD, GD)
- Comparison tools
- Interactive parameter tuning
- Real-time optimization
- Batch processing
- Export results in various formats

## Method 1: Using the API Server (Recommended)

### Step 1: Start the Server

```bash
cd C:\blatam-academy\agents\backend\onyx\server\features\ai_project_generator
python main.py
```

The server will start on `http://localhost:8020`

### Step 2: Generate the Project

The project has already been added to the queue file (`project_queue.json`). When the server starts, it will automatically process it.

Alternatively, you can make a direct API request:

```bash
curl -X POST "http://localhost:8020/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A Multi-Objective Evolutionary Algorithm (MOEA) system for solving optimization problems with multiple conflicting objectives. The system should support various MOEA algorithms like NSGA-II, NSGA-III, MOEA/D, and SPEA2. It should include visualization of Pareto fronts, performance metrics calculation (hypervolume, IGD, GD), comparison tools, and interactive parameter tuning.",
    "project_name": "moea_optimization_system",
    "author": "Blatam Academy",
    "priority": 5
  }'
```

### Step 3: Check Status

```bash
curl "http://localhost:8020/api/v1/queue"
curl "http://localhost:8020/api/v1/status"
```

## Method 2: Direct Python Script

Run the direct generation script:

```bash
cd C:\blatam-academy\agents\backend\onyx\server\features\ai_project_generator
python generate_moea_direct.py
```

## Method 3: Using Python Interactively

```python
import asyncio
from ai_project_generator.core.project_generator import ProjectGenerator

async def generate():
    generator = ProjectGenerator(base_output_dir="generated_projects")
    result = await generator.generate_project(
        description="A Multi-Objective Evolutionary Algorithm (MOEA) system...",
        project_name="moea_optimization_system",
        author="Blatam Academy"
    )
    print(result)

asyncio.run(generate())
```

## Project Output

Once generated, the project will be located at:

```
generated_projects/moea_optimization_system/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── services/
│   │   └── utils/
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── utils/
│   ├── package.json
│   └── README.md
├── docker-compose.yml
└── README.md
```

## Features Included

Based on the description, the generator will automatically include:

- **Backend (FastAPI)**:
  - REST API endpoints for MOEA operations
  - Algorithm implementations (NSGA-II, NSGA-III, MOEA/D, SPEA2)
  - Performance metrics calculation
  - Batch processing support
  - Real-time optimization endpoints
  - Export functionality

- **Frontend (React + TypeScript)**:
  - Interactive parameter tuning interface
  - Pareto front visualization
  - Performance metrics dashboard
  - Algorithm comparison tools
  - Real-time optimization monitoring
  - Results export interface

- **Additional Features**:
  - Docker configuration
  - Tests (if enabled)
  - Documentation
  - CI/CD configuration (if enabled)

## Next Steps

1. Navigate to the generated project directory
2. Install dependencies:
   ```bash
   # Backend
   cd generated_projects/moea_optimization_system/backend
   pip install -r requirements.txt
   
   # Frontend
   cd ../frontend
   npm install
   ```

3. Start the development servers:
   ```bash
   # Backend
   cd backend
   uvicorn app.main:app --reload
   
   # Frontend (in another terminal)
   cd frontend
   npm run dev
   ```

## Troubleshooting

If you encounter issues:

1. **Python not found**: Make sure Python 3.8+ is installed and in your PATH
2. **Module import errors**: Install dependencies: `pip install -r requirements.txt`
3. **Server not starting**: Check if port 8020 is available
4. **Generation fails**: Check the logs in the console output

## Queue Status

The MOEA project has been pre-added to the queue file. The continuous generator will process it automatically when the server starts.

