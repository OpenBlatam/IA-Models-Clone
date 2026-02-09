/**
 * Unit tests para task-storage
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as taskStorage from '../../app/api/tasks/utils/task-storage';
import { promises as fs } from 'fs';
import path from 'path';

// Usar el mismo directorio que la implementación real
const TASKS_DIR = path.join(process.cwd(), 'data', 'tasks');
const TASKS_FILE = path.join(TASKS_DIR, 'tasks.json');

describe('TaskStorage', () => {
  beforeEach(async () => {
    // Limpiar antes de cada test
    try {
      await fs.rm(TASKS_DIR, { recursive: true, force: true });
    } catch (e) {
      // Ignorar si no existe
    }
  });

  afterEach(async () => {
    // Limpiar después de cada test
    try {
      await fs.rm(TASKS_DIR, { recursive: true, force: true });
    } catch (e) {
      // Ignorar
    }
  });

  it('debería crear el directorio si no existe', async () => {
    await taskStorage.ensureTasksDir();
    const exists = await fs.access(TASKS_DIR).then(() => true).catch(() => false);
    expect(exists).toBe(true);
  });

  it('debería cargar tareas vacías si el archivo no existe', async () => {
    const tasks = await taskStorage.loadTasks();
    expect(Array.isArray(tasks)).toBe(true);
  });

  it('debería guardar y cargar tareas correctamente', async () => {
    const testTasks = [
      { id: 'test-1', instruction: 'Test 1', status: 'pending' },
      { id: 'test-2', instruction: 'Test 2', status: 'processing' },
    ];

    await taskStorage.saveTasks(testTasks);
    const loaded = await taskStorage.loadTasks();
    
    expect(loaded.length).toBe(2);
    expect(loaded[0].id).toBe('test-1');
  });

  it('debería encontrar una tarea por ID', async () => {
    const testTasks = [
      { id: 'test-1', instruction: 'Test 1', status: 'pending' },
      { id: 'test-2', instruction: 'Test 2', status: 'processing' },
    ];

    await taskStorage.saveTasks(testTasks);
    const found = await taskStorage.findTask('test-2');
    
    expect(found).toBeTruthy();
    expect(found?.id).toBe('test-2');
  });

  it('debería actualizar una tarea existente', async () => {
    const testTasks = [
      { id: 'test-1', instruction: 'Test 1', status: 'pending' },
    ];

    await taskStorage.saveTasks(testTasks);
    await taskStorage.updateTask('test-1', { status: 'completed' });
    
    const updated = await taskStorage.findTask('test-1');
    expect(updated?.status).toBe('completed');
  });
});

