import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

// Configurar timeout máximo
export const maxDuration = 600;
export const dynamic = 'force-dynamic';

const TASKS_DIR = path.join(process.cwd(), 'data', 'tasks');
const TASKS_FILE = path.join(TASKS_DIR, 'tasks.json');

// Asegurar que el directorio existe
async function ensureTasksDir() {
  try {
    await fs.mkdir(TASKS_DIR, { recursive: true });
  } catch (error) {
    console.error('Error creating tasks directory:', error);
  }
}

// Cargar tareas desde el archivo
async function loadTasks(): Promise<any[]> {
  try {
    await ensureTasksDir();
    const data = await fs.readFile(TASKS_FILE, 'utf-8');
    
    // Check if file is empty or only whitespace
    if (!data || !data.trim()) {
      return [];
    }
    
    try {
      return JSON.parse(data);
    } catch (parseError: any) {
      // Handle JSON parsing errors specifically
      console.error('Error parsing tasks JSON:', parseError.message);
      // If JSON is invalid, return empty array and optionally backup the corrupted file
      try {
        const backupPath = `${TASKS_FILE}.corrupted.${Date.now()}`;
        await fs.writeFile(backupPath, data, 'utf-8');
        console.warn(`⚠️ [API] Corrupted tasks file backed up to: ${backupPath}`);
      } catch (backupError) {
        // Ignore backup errors
      }
      return [];
    }
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      return [];
    }
    console.error('Error loading tasks:', error);
    return [];
  }
}

// Guardar tareas en el archivo
async function saveTasks(tasks: any[]): Promise<void> {
  try {
    await ensureTasksDir();
    await fs.writeFile(TASKS_FILE, JSON.stringify(tasks, null, 2), 'utf-8');
  } catch (error) {
    console.error('Error saving tasks:', error);
    throw error;
  }
}

// GET: Obtener todas las tareas
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const taskId = searchParams.get('taskId');

    console.log('📥 [API] GET /api/tasks - Cargando tareas...', {
      taskId: taskId || 'all',
    });

    const tasks = await loadTasks();

    if (taskId) {
      const task = tasks.find((t: any) => t.id === taskId);
      if (!task) {
        console.warn(`⚠️ [API] Tarea ${taskId} no encontrada`);
        return NextResponse.json(
          { error: 'Tarea no encontrada' },
          { status: 404 }
        );
      }

      console.log(`✅ [API] GET /api/tasks?taskId=${taskId} - Tarea encontrada`);
      return NextResponse.json({ task });
    }

    console.log(`✅ [API] GET /api/tasks - ${tasks.length} tareas encontradas`);
    return NextResponse.json({ tasks });
  } catch (error: any) {
    console.error('❌ [API] Error getting tasks:', error);
    // Retornar array vacío en lugar de error para que el frontend pueda continuar
    return NextResponse.json({ tasks: [] }, { status: 200 });
  }
}

// POST: Crear una nueva tarea
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { task } = body;

    if (!task) {
      return NextResponse.json(
        { error: 'Se requiere un objeto task' },
        { status: 400 }
      );
    }

    const tasks = await loadTasks();
    
    // Generar ID si no existe
    if (!task.id) {
      task.id = `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }
    
    // Asegurar createdAt
    if (!task.createdAt) {
      task.createdAt = new Date().toISOString();
    }

    tasks.push(task);
    await saveTasks(tasks);

    return NextResponse.json({ task, tasks });
  } catch (error: any) {
    console.error('Error creating task:', error);
    return NextResponse.json(
      { error: 'Error al crear tarea', details: error.message },
      { status: 500 }
    );
  }
}

// PUT: Actualizar una tarea
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const { taskId, updates } = body;

    if (!taskId) {
      return NextResponse.json(
        { error: 'Se requiere taskId' },
        { status: 400 }
      );
    }

    const tasks = await loadTasks();
    const taskIndex = tasks.findIndex((t: any) => t.id === taskId);

    if (taskIndex === -1) {
      return NextResponse.json(
        { error: 'Tarea no encontrada' },
        { status: 404 }
      );
    }

    tasks[taskIndex] = { ...tasks[taskIndex], ...updates };
    await saveTasks(tasks);

    return NextResponse.json({ task: tasks[taskIndex] });
  } catch (error: any) {
    console.error('Error updating task:', error);
    return NextResponse.json(
      { error: 'Error al actualizar tarea', details: error.message },
      { status: 500 }
    );
  }
}

// DELETE: Eliminar una tarea
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const taskId = searchParams.get('taskId');

    if (!taskId) {
      return NextResponse.json(
        { error: 'Se requiere taskId' },
        { status: 400 }
      );
    }

    const tasks = await loadTasks();
    const filteredTasks = tasks.filter((t: any) => t.id !== taskId);
    await saveTasks(filteredTasks);

    return NextResponse.json({ success: true, tasks: filteredTasks });
  } catch (error: any) {
    console.error('Error deleting task:', error);
    return NextResponse.json(
      { error: 'Error al eliminar tarea', details: error.message },
      { status: 500 }
    );
  }
}

