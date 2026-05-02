/**
 * Ejemplo de uso del SDK TypeScript
 * ==================================
 */

import { createBULApiClient } from '../bul-api-client';
import { DocumentRequest, TaskStatus } from '../frontend_types';

async function main(): Promise<void> {
    // Crear cliente
    const client = createBULApiClient({
        baseUrl: 'http://localhost:8000'
    });
    
    // Verificar salud
    try {
        const health = await client.getHealth();
        console.log('✅ API Health:', health.status);
    } catch (error) {
        console.error('❌ Error verificando salud:', error);
        return;
    }
    
    // Generar documento
    const request: DocumentRequest = {
        query: 'Crear un plan de marketing digital para una startup tecnológica',
        business_area: 'marketing',
        document_type: 'strategy',
        priority: 1
    };
    
    try {
        console.log('\n🚀 Generando documento...');
        
        const document = await client.generateDocumentAndWait(request, {
            useWebSocket: true,
            onProgress: (status: TaskStatus) => {
                console.log(`📊 Progreso: ${status.progress}% - Estado: ${status.status}`);
            }
        });
        
        console.log('\n✅ Documento generado:');
        console.log(`   Título: ${document.document.title}`);
        console.log(`   Longitud: ${document.document.content.length} caracteres`);
        console.log(`   Tipo: ${document.document.document_type}`);
        
    } catch (error) {
        console.error('❌ Error:', error);
    }
    
    // Listar documentos
    try {
        console.log('\n📋 Listando documentos recientes...');
        const documents = await client.listDocuments(5);
        console.log(`   Total documentos: ${documents.total || 0}`);
    } catch (error) {
        console.error('❌ Error listando documentos:', error);
    }
    
    // Estadísticas
    try {
        console.log('\n📊 Estadísticas de la API...');
        const stats = await client.getStats();
        console.log(`   Tareas completadas: ${stats.tasks_completed || 0}`);
        console.log(`   Tareas activas: ${stats.tasks_active || 0}`);
    } catch (error) {
        console.error('❌ Error obteniendo estadísticas:', error);
    }
}

// Ejecutar
main().catch(console.error);
































