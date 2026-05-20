import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export const maxDuration = 60;
export const dynamic = 'force-dynamic';

const TASKS_DIR = path.join(process.cwd(), 'data', 'tasks');
const TASKS_FILE = path.join(TASKS_DIR, 'tasks.json');
const PROCESSING_FILE = path.join(TASKS_DIR, 'processing.json');

// Asegurar que el directorio existe
async function ensureTasksDir() {
  try {
    await fs.mkdir(TASKS_DIR, { recursive: true });
  } catch (error) {
    console.error('Error creating tasks directory:', error);
  }
}

// Cargar tareas
async function loadTasks(): Promise<any[]> {
  try {
    await ensureTasksDir();
    const data = await fs.readFile(TASKS_FILE, 'utf-8');
    return JSON.parse(data);
  } catch (error: any) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }
}

// Guardar tareas
async function saveTasks(tasks: any[]): Promise<void> {
  await ensureTasksDir();
  await fs.writeFile(TASKS_FILE, JSON.stringify(tasks, null, 2), 'utf-8');
}

// Cargar tareas en procesamiento
async function loadProcessing(): Promise<Set<string>> {
  try {
    await ensureTasksDir();
    const data = await fs.readFile(PROCESSING_FILE, 'utf-8');
    return new Set(JSON.parse(data));
  } catch {
    return new Set();
  }
}

// Guardar tareas en procesamiento
async function saveProcessing(processing: Set<string>): Promise<void> {
  await ensureTasksDir();
  await fs.writeFile(PROCESSING_FILE, JSON.stringify(Array.from(processing), null, 2), 'utf-8');
}

// POST: Detener una tarea
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { taskId } = body;

    if (!taskId) {
      return NextResponse.json(
        { error: 'Se requiere taskId' },
        { status: 400 }
      );
    }

    // Intentar detener via backend unificado
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8030';
    try {
      const backendResponse = await fetch(`${backendUrl}/api/v1/agent/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: taskId }),
        signal: AbortSignal.timeout(5000),
      });
      if (backendResponse.ok) {
        console.log(`✅ [API] Tarea ${taskId} detenida en backend unificado`);
      }
    } catch (backendError: any) {
      console.warn(`⚠️ [API] Backend unificado no disponible para stop:`, backendError.message);
    }

    const tasks = await loadTasks();
    const taskIndex = tasks.findIndex((t: any) => t.id === taskId);

    if (taskIndex === -1) {
      return NextResponse.json(
        { error: 'Tarea no encontrada' },
        { status: 404 }
      );
    }

    // Remover de procesamiento
    const processing = await loadProcessing();
    processing.delete(taskId);
    await saveProcessing(processing);

    // Actualizar estado de la tarea (preservar plan de cambios si existe)
    tasks[taskIndex] = {
      ...tasks[taskIndex],
      status: 'stopped',
      error: 'Procesamiento detenido por el usuario',
      // NO eliminar pendingApproval ni streamingContent para que se pueda ver el plan
    };
    await saveTasks(tasks);

    return NextResponse.json({
      success: true,
      message: 'Tarea detenida exitosamente',
      task: tasks[taskIndex]
    });
  } catch (error: any) {
    console.error('Error stopping task:', error);
    return NextResponse.json(
      { error: 'Error al detener tarea', details: error.message },
      { status: 500 }
    );
  }
}

