"""
Dashboard Template - Template HTML para el dashboard del agente
==============================================================

Template HTML completo para la interfaz web del agente.
"""


def get_dashboard_html() -> str:
    """
    Obtener HTML del dashboard del agente.
    
    Returns:
        String con el HTML completo del dashboard.
    """
    return """<!DOCTYPE html>
<html>
<head>
    <title>Cursor Agent 24/7</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .status {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .status-label {
            color: #666;
        }
        .status-value {
            font-weight: 600;
            color: #333;
        }
        .status-value.running { color: #10b981; }
        .status-value.stopped { color: #ef4444; }
        .status-value.paused { color: #f59e0b; }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        button {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-start {
            background: #10b981;
            color: white;
        }
        .btn-start:hover { background: #059669; }
        .btn-stop {
            background: #ef4444;
            color: white;
        }
        .btn-stop:hover { background: #dc2626; }
        .btn-pause {
            background: #f59e0b;
            color: white;
        }
        .btn-pause:hover { background: #d97706; }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .command-input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .command-input:focus {
            outline: none;
            border-color: #667eea;
        }
        .tasks {
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        .task-item {
            background: #f9fafb;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 12px;
        }
        .task-command {
            color: #333;
            margin-bottom: 4px;
        }
        .task-status {
            color: #666;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Cursor Agent 24/7</h1>
        <p class="subtitle">Agente persistente que ejecuta comandos sin parar</p>
        
        <div class="status" id="status">
            <div class="status-item">
                <span class="status-label">Estado:</span>
                <span class="status-value" id="status-value">Cargando...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Tareas:</span>
                <span class="status-value" id="tasks-value">-</span>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn-start" id="btn-start" onclick="startAgent()">▶️ Iniciar</button>
            <button class="btn-pause" id="btn-pause" onclick="pauseAgent()" disabled>⏸️ Pausar</button>
            <button class="btn-stop" id="btn-stop" onclick="stopAgent()" disabled>⏹️ Detener</button>
        </div>
        
        <input type="text" class="command-input" id="command-input" 
               placeholder="Escribe un comando y presiona Enter..." 
               onkeypress="if(event.key==='Enter') sendCommand()">
        
        <div class="tasks" id="tasks"></div>
    </div>
    
    <script>
        let statusInterval;
        
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('status-value').textContent = data.status.toUpperCase();
                document.getElementById('status-value').className = 'status-value ' + data.status;
                document.getElementById('tasks-value').textContent = 
                    `${data.tasks_completed}/${data.tasks_total} completadas`;
                
                const isRunning = data.status === 'running';
                const isPaused = data.status === 'paused';
                
                document.getElementById('btn-start').disabled = isRunning || isPaused;
                document.getElementById('btn-pause').disabled = !isRunning;
                document.getElementById('btn-stop').disabled = !isRunning && !isPaused;
                
                loadTasks();
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }
        
        async function loadTasks() {
            try {
                const response = await fetch('/api/tasks?limit=10');
                const data = await response.json();
                
                const tasksDiv = document.getElementById('tasks');
                tasksDiv.innerHTML = data.tasks.map(task => `
                    <div class="task-item">
                        <div class="task-command">${task.command.substring(0, 50)}${task.command.length > 50 ? '...' : ''}</div>
                        <div class="task-status">${task.status} - ${new Date(task.timestamp).toLocaleString()}</div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading tasks:', error);
            }
        }
        
        async function startAgent() {
            try {
                await fetch('/api/start', { method: 'POST' });
                updateStatus();
            } catch (error) {
                alert('Error al iniciar: ' + error.message);
            }
        }
        
        async function stopAgent() {
            if (!confirm('¿Detener el agente?')) return;
            try {
                await fetch('/api/stop', { method: 'POST' });
                updateStatus();
            } catch (error) {
                alert('Error al detener: ' + error.message);
            }
        }
        
        async function pauseAgent() {
            try {
                await fetch('/api/pause', { method: 'POST' });
                updateStatus();
            } catch (error) {
                alert('Error al pausar: ' + error.message);
            }
        }
        
        async function sendCommand() {
            const input = document.getElementById('command-input');
            const command = input.value.trim();
            if (!command) return;
            
            try {
                await fetch('/api/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command })
                });
                input.value = '';
                updateStatus();
            } catch (error) {
                alert('Error al enviar comando: ' + error.message);
            }
        }
        
        updateStatus();
        statusInterval = setInterval(updateStatus, 2000);
    </script>
</body>
</html>"""

