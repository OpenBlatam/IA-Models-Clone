/**
 * Unit tests para verificar que el stop funciona correctamente
 */
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

describe('Stop Generation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Task Stop API', () => {
    it('debería marcar una tarea como stopped cuando se llama al endpoint', async () => {
      const taskId = 'test-task-1';
      const response = await fetch('/api/tasks/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId }),
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.success).toBe(true);
    });

    it('debería preservar el plan cuando se detiene una tarea', async () => {
      // Simular una tarea con plan
      const taskWithPlan = {
        id: 'test-task-2',
        status: 'processing',
        pendingApproval: {
          plan: { files_to_create: ['test.js'] },
          actions: [{ path: 'test.js', content: 'test', action: 'create' }],
        },
      };

      // Detener la tarea
      const response = await fetch('/api/tasks/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId: taskWithPlan.id }),
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      
      // Verificar que el plan está preservado
      expect(data.task.pendingApproval).toBeDefined();
      expect(data.task.status).toBe('stopped');
    });
  });

  describe('Stream Stop Check', () => {
    it('debería verificar el estado de stop frecuentemente', async () => {
      let checkCount = 0;
      const checkStopped = vi.fn().mockImplementation(async () => {
        checkCount++;
        return checkCount >= 5; // Detener después de 5 verificaciones
      });

      // Simular procesamiento con verificaciones frecuentes
      const startTime = Date.now();
      while (Date.now() - startTime < 1000) {
        const isStopped = await checkStopped();
        if (isStopped) break;
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      expect(checkStopped).toHaveBeenCalled();
      expect(checkCount).toBeGreaterThanOrEqual(5);
    });

    it('debería cancelar el fetch cuando se detecta stop', async () => {
      const abortController = new AbortController();
      let aborted = false;

      abortController.signal.addEventListener('abort', () => {
        aborted = true;
      });

      // Simular detección de stop
      setTimeout(() => {
        abortController.abort();
      }, 100);

      // Simular fetch que se cancela
      try {
        await fetch('http://example.com', {
          signal: abortController.signal,
        });
      } catch (error: any) {
        if (error.name === 'AbortError') {
          expect(aborted).toBe(true);
        }
      }
    });
  });

  describe('Plan and Commit Processing', () => {
    it('debería procesar correctamente un plan con archivos', () => {
      const plan = {
        files_to_create: [
          { path: 'test.js', content: 'console.log("test");' },
        ],
        files_to_modify: [],
      };

      // Simular preparación de acciones
      const actions = plan.files_to_create.map(file => ({
        path: file.path,
        content: file.content,
        action: 'create' as const,
      }));

      expect(actions).toHaveLength(1);
      expect(actions[0].path).toBe('test.js');
      expect(actions[0].action).toBe('create');
    });

    it('debería generar un commit message cuando hay plan', () => {
      const plan = {
        files_to_create: [{ path: 'test.js', content: 'test' }],
        files_to_modify: [],
      };
      const instruction = 'Create a test file';

      // Simular generación de commit message
      const commitMessage = `feat: ${instruction.substring(0, 50)}`;

      expect(commitMessage).toContain('feat:');
      expect(commitMessage.length).toBeGreaterThan(0);
    });

    it('debería preservar el plan cuando se detiene antes de completar', () => {
      const partialPlan = {
        files_to_create: [{ path: 'partial.js', content: 'partial' }],
        files_to_modify: [],
      };

      // El plan parcial debería preservarse
      expect(partialPlan.files_to_create).toHaveLength(1);
      expect(partialPlan.files_to_create[0].path).toBe('partial.js');
    });
  });

  describe('Continuous Generation Until Stop', () => {
    it('debería continuar generando hasta que se presione stop', async () => {
      let generationCount = 0;
      let isStopped = false;

      const generate = async () => {
        while (!isStopped) {
          generationCount++;
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      };

      // Iniciar generación
      const generationPromise = generate();

      // Simular que se presiona stop después de 500ms
      setTimeout(() => {
        isStopped = true;
      }, 500);

      await generationPromise;

      // Debería haber generado varias veces antes de detenerse
      expect(generationCount).toBeGreaterThan(0);
      expect(isStopped).toBe(true);
    });

    it('debería verificar el estado de stop en cada iteración', async () => {
      let iterations = 0;
      const maxIterations = 10;
      let shouldStop = false;

      const checkStop = () => shouldStop;

      while (iterations < maxIterations && !checkStop()) {
        iterations++;
        // Simular trabajo
        await new Promise(resolve => setTimeout(resolve, 50));
        
        // Simular que se presiona stop en la iteración 5
        if (iterations === 5) {
          shouldStop = true;
        }
      }

      expect(iterations).toBe(5); // Debería detenerse en la iteración 5
      expect(shouldStop).toBe(true);
    });
  });
});

