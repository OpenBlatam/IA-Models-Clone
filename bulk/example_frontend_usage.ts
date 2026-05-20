/**
 * Ejemplo de uso del cliente BUL API desde TypeScript
 * 
 * Este archivo muestra diferentes formas de usar el cliente API
 */

import { createBULApiClient } from './bul-api-client';
import type { DocumentRequest, TaskStatus } from './frontend_types';

// Configurar cliente
const apiClient = createBULApiClient({
  baseUrl: 'http://localhost:8000',
  timeout: 30000
});

// ============================================================================
// Ejemplo 1: Generación básica de documento
// ============================================================================

async function ejemplo1_Basico() {
  console.log('=== Ejemplo 1: Generación Básica ===');

  try {
    // Generar documento
    const response = await apiClient.generateDocument({
      query: 'Crear un plan de negocios para una startup tecnológica',
      priority: 1
    });

    console.log('Task ID:', response.task_id);
    console.log('Estado:', response.status);

    // Verificar estado hasta completar
    let status = await apiClient.getTaskStatus(response.task_id);
    while (status.status === 'queued' || status.status === 'processing') {
      console.log(`Progreso: ${status.progress}%`);
      await new Promise(resolve => setTimeout(resolve, 2000));
      status = await apiClient.getTaskStatus(response.task_id);
    }

    if (status.status === 'completed') {
      const document = await apiClient.getTaskDocument(response.task_id);
      console.log('Documento generado:', document.document.content.substring(0, 200));
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

// ============================================================================
// Ejemplo 2: Generación con polling automático
// ============================================================================

async function ejemplo2_PollingAutomatico() {
  console.log('=== Ejemplo 2: Polling Automático ===');

  try {
    const document = await apiClient.generateDocumentAndWait(
      {
        query: 'Estrategia de marketing digital para e-commerce',
        business_area: 'marketing',
        document_type: 'strategy',
        priority: 2
      },
      {
        pollingInterval: 2000,
        onProgress: (status: TaskStatus) => {
          console.log(`Progreso: ${status.progress}% - Estado: ${status.status}`);
        }
      }
    );

    console.log('Documento completo:');
    console.log(document.document.content);
  } catch (error) {
    console.error('Error:', error);
  }
}

// ============================================================================
// Ejemplo 3: Múltiples documentos
// ============================================================================

async function ejemplo3_MultiplesDocumentos() {
  console.log('=== Ejemplo 3: Múltiples Documentos ===');

  const queries: DocumentRequest[] = [
    {
      query: 'Plan de recursos humanos para empresa mediana',
      business_area: 'hr',
      document_type: 'plan',
      priority: 1
    },
    {
      query: 'Estrategia de ventas B2B',
      business_area: 'sales',
      document_type: 'strategy',
      priority: 2
    },
    {
      query: 'Manual de operaciones',
      business_area: 'operations',
      document_type: 'manual',
      priority: 1
    }
  ];

  try {
    // Generar todos los documentos
    const responses = await Promise.all(
      queries.map(query => apiClient.generateDocument(query))
    );

    console.log(`Generados ${responses.length} documentos`);

    // Esperar a que todos se completen
    const documents = await Promise.all(
      responses.map(response =>
        apiClient.waitForTaskCompletion(response.task_id)
      )
    );

    documents.forEach((status, index) => {
      console.log(`Documento ${index + 1}: ${status.status}`);
    });
  } catch (error) {
    console.error('Error:', error);
  }
}

// ============================================================================
// Ejemplo 4: Listar tareas y documentos
// ============================================================================

async function ejemplo4_Listar() {
  console.log('=== Ejemplo 4: Listar Tareas y Documentos ===');

  try {
    // Listar tareas completadas
    const tasks = await apiClient.listTasks({
      status: 'completed',
      limit: 10,
      offset: 0
    });

    console.log(`Total tareas completadas: ${tasks.total}`);
    tasks.tasks.forEach(task => {
      console.log(`- ${task.task_id}: ${task.query_preview.substring(0, 50)}...`);
    });

    // Listar documentos
    const documents = await apiClient.listDocuments(10, 0);
    console.log(`Total documentos: ${documents.total}`);
    documents.documents.forEach(doc => {
      console.log(`- ${doc.task_id}: ${doc.query_preview.substring(0, 50)}...`);
    });
  } catch (error) {
    console.error('Error:', error);
  }
}

// ============================================================================
// Ejemplo 5: Health check y estadísticas
// ============================================================================

async function ejemplo5_Sistema() {
  console.log('=== Ejemplo 5: Sistema ===');

  try {
    // Health check
    const health = await apiClient.getHealth();
    console.log('Estado del sistema:', health.status);
    console.log('Uptime:', health.uptime);
    console.log('Tareas activas:', health.active_tasks);

    // Estadísticas
    const stats = await apiClient.getStats();
    console.log('Total solicitudes:', stats.total_requests);
    console.log('Tasa de éxito:', (stats.success_rate * 100).toFixed(2) + '%');
    console.log('Tiempo promedio:', stats.average_processing_time.toFixed(2) + 's');
  } catch (error) {
    console.error('Error:', error);
  }
}

// ============================================================================
// Ejemplo 6: Cancelar tarea
// ============================================================================

async function ejemplo6_Cancelar() {
  console.log('=== Ejemplo 6: Cancelar Tarea ===');

  try {
    // Generar documento
    const response = await apiClient.generateDocument({
      query: 'Documento que será cancelado',
      priority: 1
    });

    console.log('Task ID:', response.task_id);

    // Esperar un poco
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Cancelar
    await apiClient.cancelTask(response.task_id);
    console.log('Tarea cancelada');

    // Verificar estado
    const status = await apiClient.getTaskStatus(response.task_id);
    console.log('Estado final:', status.status);
  } catch (error) {
    console.error('Error:', error);
  }
}

// ============================================================================
// Ejemplo 7: React Hook (simulado)
// ============================================================================

interface UseDocumentGenerationResult {
  generate: (query: string) => Promise<void>;
  status: TaskStatus | null;
  loading: boolean;
  error: string | null;
  document: string | null;
}

function useDocumentGeneration(): UseDocumentGenerationResult {
  // En un componente React real, usar useState
  let status: TaskStatus | null = null;
  let loading = false;
  let error: string | null = null;
  let document: string | null = null;

  const generate = async (query: string) => {
    loading = true;
    error = null;

    try {
      const doc = await apiClient.generateDocumentAndWait(
        { query },
        {
          onProgress: (s) => {
            status = s;
          }
        }
      );

      document = doc.document.content;
      status = doc as any;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Unknown error';
    } finally {
      loading = false;
    }
  };

  return { generate, status, loading, error, document };
}

// ============================================================================
// Ejecutar ejemplos
// ============================================================================

async function main() {
  console.log('Iniciando ejemplos de uso del cliente BUL API...\n');

  // Verificar que el servidor esté funcionando
  try {
    await apiClient.getHealth();
    console.log('✅ Servidor conectado\n');
  } catch (error) {
    console.error('❌ No se puede conectar al servidor. Asegúrate de que esté corriendo en http://localhost:8000');
    return;
  }

  // Ejecutar ejemplos (descomenta los que quieras probar)
  // await ejemplo1_Basico();
  // await ejemplo2_PollingAutomatico();
  // await ejemplo3_MultiplesDocumentos();
  // await ejemplo4_Listar();
  // await ejemplo5_Sistema();
  // await ejemplo6_Cancelar();

  console.log('\nEjemplos completados');
}

// Ejecutar si es el archivo principal
if (require.main === module) {
  main().catch(console.error);
}

export {
  ejemplo1_Basico,
  ejemplo2_PollingAutomatico,
  ejemplo3_MultiplesDocumentos,
  ejemplo4_Listar,
  ejemplo5_Sistema,
  ejemplo6_Cancelar,
  useDocumentGeneration
};
































