/**
 * Script maestro para ejecutar todos los tests de debug
 * y encontrar el culpable del problema del stream
 */

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const tests = [
  { name: 'Test Stream Directo', command: 'npm run test:stream' },
  { name: 'Test DeepSeek Endpoint', command: 'npm run test:deepseek' },
  { name: 'Test Backend Worker', command: 'npm run test:backend-worker' },
  { name: 'Unit Tests', command: 'npm run test:unit' },
  { name: 'Integration Tests', command: 'npm run test:integration' },
];

async function runTest(name: string, command: string) {
  console.log(`\n${'='.repeat(80)}`);
  console.log(`🧪 Ejecutando: ${name}`);
  console.log(`📝 Comando: ${command}`);
  console.log('='.repeat(80));

  try {
    const { stdout, stderr } = await execAsync(command, {
      cwd: process.cwd(),
      maxBuffer: 10 * 1024 * 1024, // 10MB
    });

    console.log(stdout);
    if (stderr) {
      console.error(stderr);
    }

    console.log(`✅ ${name} completado exitosamente`);
    return { success: true, name };
  } catch (error: any) {
    console.error(`❌ ${name} falló:`);
    console.error(error.stdout || error.message);
    return { success: false, name, error: error.message };
  }
}

async function main() {
  console.log('🚀 Iniciando suite completa de tests de debug...\n');
  console.log('Este script ejecutará todos los tests para encontrar el culpable del problema del stream.\n');

  const results: Array<{ success: boolean; name: string; error?: string }> = [];

  for (const test of tests) {
    const result = await runTest(test.name, test.command);
    results.push(result);
    
    // Pequeña pausa entre tests
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  // Resumen final
  console.log(`\n${'='.repeat(80)}`);
  console.log('📊 RESUMEN FINAL');
  console.log('='.repeat(80));

  const successful = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;

  results.forEach(result => {
    const icon = result.success ? '✅' : '❌';
    console.log(`${icon} ${result.name}`);
    if (!result.success && result.error) {
      console.log(`   Error: ${result.error.substring(0, 100)}...`);
    }
  });

  console.log(`\n📈 Estadísticas:`);
  console.log(`   - Exitosos: ${successful}/${results.length}`);
  console.log(`   - Fallidos: ${failed}/${results.length}`);

  if (failed > 0) {
    console.log(`\n🔍 Tests fallidos indican dónde está el problema:`);
    results
      .filter(r => !r.success)
      .forEach(r => {
        console.log(`   - ${r.name}: Revisa este componente`);
      });
  }

  // Análisis del problema
  console.log(`\n🔍 ANÁLISIS:`);
  
  const streamTest = results.find(r => r.name.includes('Stream Directo'));
  const deepseekTest = results.find(r => r.name.includes('DeepSeek Endpoint'));
  const workerTest = results.find(r => r.name.includes('Backend Worker'));

  if (streamTest?.success && deepseekTest?.success && !workerTest?.success) {
    console.log('❌ PROBLEMA IDENTIFICADO: El backend worker no está procesando el stream correctamente');
    console.log('   - El endpoint funciona directamente');
    console.log('   - El problema está en cómo el worker procesa el stream');
    console.log('   - Revisa: app/api/tasks/process/route.ts y app/api/tasks/utils/stream-processor.ts');
  } else if (!deepseekTest?.success) {
    console.log('❌ PROBLEMA IDENTIFICADO: El endpoint /api/deepseek/stream no funciona');
    console.log('   - Revisa: app/api/deepseek/stream/route.ts');
  } else if (streamTest?.success && deepseekTest?.success && workerTest?.success) {
    console.log('✅ Todos los tests pasaron - el problema podría ser intermitente o relacionado con el contexto de ejecución');
    console.log('   - Revisa los logs del servidor cuando se ejecuta una tarea real');
  }

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((error) => {
  console.error('❌ Error fatal:', error);
  process.exit(1);
});

