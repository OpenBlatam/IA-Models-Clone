/**
 * Utilidades para el almacenamiento de tareas
 */
import { promises as fs } from 'fs';
import path from 'path';

const TASKS_DIR = path.join(process.cwd(), 'data', 'tasks');
const TASKS_FILE = path.join(TASKS_DIR, 'tasks.json');
const PROCESSING_FILE = path.join(TASKS_DIR, 'processing.json');

/**
 * Asegurar que el directorio de tareas existe
 */
export async function ensureTasksDir(): Promise<void> {
  try {
    await fs.mkdir(TASKS_DIR, { recursive: true });
  } catch (error: any) {
    console.error('❌ [STORAGE] Error creating tasks directory:', error);
    console.error('❌ [STORAGE] Error details:', {
      code: error.code,
      message: error.message,
      path: TASKS_DIR,
      cwd: process.cwd(),
    });
    // No lanzar error, intentar continuar
    // throw error;
  }
}

/**
 * Cargar todas las tareas
 */
export async function loadTasks(): Promise<any[]> {
  try {
    await ensureTasksDir();
    const data = await fs.readFile(TASKS_FILE, 'utf-8');
    if (!data || data.trim() === '') {
      console.warn('⚠️ [STORAGE] Archivo de tareas vacío, retornando array vacío');
      return [];
    }
    return JSON.parse(data);
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      console.log('ℹ️ [STORAGE] Archivo de tareas no existe, retornando array vacío');
      return [];
    }
    console.error('❌ [STORAGE] Error loading tasks:', error);
    console.error('❌ [STORAGE] Error details:', {
      code: error.code,
      message: error.message,
      path: TASKS_FILE,
    });
    // Retornar array vacío en lugar de lanzar error
    return [];
  }
}

/**
 * Guardar tareas
 */
export async function saveTasks(tasks: any[]): Promise<void> {
  try {
    await ensureTasksDir();
    await fs.writeFile(TASKS_FILE, JSON.stringify(tasks, null, 2), 'utf-8');
  } catch (error: any) {
    console.error('❌ [STORAGE] Error saving tasks:', error);
    console.error('❌ [STORAGE] Error details:', {
      code: error.code,
      message: error.message,
      path: TASKS_FILE,
    });
    throw error;
  }
}

/**
 * Cargar tareas en procesamiento
 */
export async function loadProcessing(): Promise<Set<string>> {
  try {
    await ensureTasksDir();
    const data = await fs.readFile(PROCESSING_FILE, 'utf-8');
    return new Set(JSON.parse(data));
  } catch {
    return new Set();
  }
}

/**
 * Guardar tareas en procesamiento
 */
export async function saveProcessing(processing: Set<string>): Promise<void> {
  await ensureTasksDir();
  await fs.writeFile(
    PROCESSING_FILE,
    JSON.stringify(Array.from(processing), null, 2),
    'utf-8'
  );
}

/**
 * Buscar una tarea por ID
 */
export async function findTask(taskId: string): Promise<any | null> {
  try {
    const tasks = await loadTasks();
    const task = tasks.find((t: any) => t.id === taskId) || null;
    if (!task) {
      console.warn(`⚠️ [STORAGE] Tarea ${taskId} no encontrada en ${tasks.length} tareas`);
    }
    return task;
  } catch (error: any) {
    console.error(`❌ [STORAGE] Error finding task ${taskId}:`, error);
    return null;
  }
}

/**
 * Actualizar una tarea
 */
export async function updateTask(taskId: string, updates: Partial<any>): Promise<void> {
  try {
    const tasks = await loadTasks();
    const taskIndex = tasks.findIndex((t: any) => t.id === taskId);
    
    if (taskIndex === -1) {
      console.error(`❌ [STORAGE] Tarea ${taskId} no encontrada para actualizar`);
      console.error(`❌ [STORAGE] Tareas disponibles:`, tasks.map(t => t.id));
      throw new Error(`Tarea ${taskId} no encontrada`);
    }
    
    tasks[taskIndex] = {
      ...tasks[taskIndex],
      ...updates,
    };
    
    await saveTasks(tasks);
    console.log(`✅ [STORAGE] Tarea ${taskId} actualizada exitosamente`);
  } catch (error: any) {
    console.error(`❌ [STORAGE] Error updating task ${taskId}:`, error);
    throw error;
  }
}

