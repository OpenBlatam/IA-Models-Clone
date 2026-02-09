#!/usr/bin/env node

/**
 * Asistente interactivo para refactorización
 * Guía paso a paso a través de la refactorización
 * 
 * Uso: node refactoring-assistant.js
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const componentsDir = path.join(__dirname, '../ChatInterface');
const hooksDir = path.join(componentsDir, 'hooks');
const contextsDir = path.join(componentsDir, 'contexts');

console.log('🤖 Asistente de Refactorización - ChatInterface.tsx\n');
console.log('═'.repeat(60));
console.log('Bienvenido al asistente interactivo de refactorización');
console.log('Te guiaré paso a paso a través del proceso\n');

function question(prompt) {
  return new Promise(resolve => {
    rl.question(prompt, resolve);
  });
}

async function main() {
  // Paso 1: Análisis inicial
  console.log('📊 PASO 1: Análisis Inicial\n');
  
  const runAnalysis = await question('¿Deseas ejecutar análisis de estados? (s/n): ');
  if (runAnalysis.toLowerCase() === 's') {
    console.log('\n🔍 Ejecutando análisis...');
    try {
      const { execSync } = require('child_process');
      execSync('node scripts/analyze-unused-states.js', { 
        cwd: path.dirname(__dirname),
        stdio: 'inherit'
      });
    } catch (error) {
      console.log('⚠️  Error ejecutando análisis (continuando...)');
    }
  }
  
  // Paso 2: Estructura
  console.log('\n📁 PASO 2: Estructura de Directorios\n');
  
  if (!fs.existsSync(componentsDir)) {
    const createStructure = await question('¿Crear estructura de directorios? (s/n): ');
    if (createStructure.toLowerCase() === 's') {
      console.log('📁 Creando estructura...');
      fs.mkdirSync(hooksDir, { recursive: true });
      fs.mkdirSync(contextsDir, { recursive: true });
      fs.mkdirSync(path.join(componentsDir, 'components'), { recursive: true });
      fs.mkdirSync(path.join(componentsDir, 'types'), { recursive: true });
      fs.mkdirSync(path.join(componentsDir, 'utils'), { recursive: true });
      console.log('✅ Estructura creada');
    }
  } else {
    console.log('✅ Estructura ya existe');
  }
  
  // Paso 3: Qué extraer primero
  console.log('\n🎯 PASO 3: Decidir qué extraer primero\n');
  
  console.log('Opciones:');
  console.log('1. Hook (useState, useEffect, etc.)');
  console.log('2. Componente UI (JSX)');
  console.log('3. Context Provider');
  console.log('4. Ver recomendaciones');
  
  const choice = await question('\n¿Qué deseas extraer? (1-4): ');
  
  switch (choice) {
    case '1':
      await extractHook();
      break;
    case '2':
      await extractComponent();
      break;
    case '3':
      await extractContext();
      break;
    case '4':
      await showRecommendations();
      break;
    default:
      console.log('Opción no válida');
  }
  
  // Paso 4: Validación
  console.log('\n✅ PASO 4: Validación\n');
  
  const runValidation = await question('¿Deseas ejecutar validación? (s/n): ');
  if (runValidation.toLowerCase() === 's') {
    console.log('\n🔍 Ejecutando validación...');
    try {
      const { execSync } = require('child_process');
      execSync('node scripts/validate-refactoring.js', { 
        cwd: path.dirname(__dirname),
        stdio: 'inherit'
      });
    } catch (error) {
      console.log('⚠️  Error ejecutando validación (continuando...)');
    }
  }
  
  console.log('\n✅ Sesión completada');
  console.log('📚 Revisa la documentación para más guías:');
  console.log('   - ChatInterface_COMPLETE_GUIDE.md');
  console.log('   - ChatInterface_MIGRATION_STEPS.md');
  console.log('   - ChatInterface_REFACTORING_EXAMPLES.md');
  
  rl.close();
}

async function extractHook() {
  console.log('\n🪝 Extracción de Hook\n');
  
  const hookName = await question('Nombre del hook (ej: useSearch): ');
  if (!hookName) {
    console.log('❌ Nombre requerido');
    return;
  }
  
  const states = await question('Estados a incluir (separados por comas): ');
  const statesList = states.split(',').map(s => s.trim()).filter(s => s);
  
  console.log(`\n📝 Generando template para ${hookName}...`);
  
  try {
    const { execSync } = require('child_process');
    const command = `node scripts/extract-hook-template.js ${hookName} ${statesList.join(' ')}`;
    execSync(command, { 
      cwd: path.dirname(__dirname),
      stdio: 'inherit'
    });
    console.log(`\n✅ Template generado para ${hookName}`);
    console.log(`📁 Revisa: ChatInterface/hooks/${hookName}.ts`);
  } catch (error) {
    console.log('⚠️  Error generando template');
  }
}

async function extractComponent() {
  console.log('\n🧩 Extracción de Componente\n');
  
  const componentName = await question('Nombre del componente (ej: MessageList): ');
  if (!componentName) {
    console.log('❌ Nombre requerido');
    return;
  }
  
  const startLine = await question('Línea de inicio: ');
  const endLine = await question('Línea de fin: ');
  const props = await question('Props (separadas por comas): ');
  const propsList = props.split(',').map(p => p.trim()).filter(p => p);
  
  console.log(`\n📝 Extrayendo componente ${componentName}...`);
  
  try {
    const { execSync } = require('child_process');
    const command = `node scripts/extract-component.js ${componentName} ${startLine} ${endLine} ${propsList.join(' ')}`;
    execSync(command, { 
      cwd: path.dirname(__dirname),
      stdio: 'inherit'
    });
    console.log(`\n✅ Componente extraído: ${componentName}`);
    console.log(`📁 Revisa: ChatInterface/components/${componentName}/`);
  } catch (error) {
    console.log('⚠️  Error extrayendo componente');
  }
}

async function extractContext() {
  console.log('\n🌐 Extracción de Context\n');
  
  const contextName = await question('Nombre del context (ej: ChatContext): ');
  if (!contextName) {
    console.log('❌ Nombre requerido');
    return;
  }
  
  console.log(`\n📝 Generando template para ${contextName}...`);
  
  const template = `import { createContext, useContext, ReactNode } from 'react';

interface ${contextName}Type {
  // Agregar propiedades aquí
}

const ${contextName} = createContext<${contextName}Type | null>(null);

export function ${contextName}Provider({ children }: { children: ReactNode }) {
  // TODO: Implementar lógica del provider
  const value: ${contextName}Type = {
    // TODO: Agregar valores
  };
  
  return (
    <${contextName}.Provider value={value}>
      {children}
    </${contextName}.Provider>
  );
}

export function use${contextName.replace('Context', '')}() {
  const context = useContext(${contextName});
  if (!context) {
    throw new Error(\`use${contextName.replace('Context', '')} must be used within ${contextName}Provider\`);
  }
  return context;
}
`;
  
  const contextPath = path.join(contextsDir, `${contextName}.tsx`);
  fs.writeFileSync(contextPath, template);
  
  console.log(`\n✅ Template generado para ${contextName}`);
  console.log(`📁 Revisa: ${contextPath}`);
}

async function showRecommendations() {
  console.log('\n💡 Recomendaciones\n');
  
  console.log('Basado en el análisis, te recomiendo:');
  console.log('');
  console.log('1. 🎯 PRIORIDAD ALTA:');
  console.log('   - Extraer useChatState (estados core del chat)');
  console.log('   - Extraer useSearch (funcionalidad de búsqueda)');
  console.log('   - Crear MessageList component');
  console.log('');
  console.log('2. 🟡 PRIORIDAD MEDIA:');
  console.log('   - Extraer useMessageManagement');
  console.log('   - Crear ChatContext');
  console.log('   - Extraer InputArea component');
  console.log('');
  console.log('3. 🟢 PRIORIDAD BAJA:');
  console.log('   - Extraer hooks de features avanzadas');
  console.log('   - Crear contexts adicionales');
  console.log('   - Optimizaciones de performance');
  console.log('');
  console.log('📚 Para más detalles, revisa:');
  console.log('   - ChatInterface_QUICK_WINS.md');
  console.log('   - ChatInterface_IMPLEMENTATION_ROADMAP.md');
}

main().catch(console.error);




