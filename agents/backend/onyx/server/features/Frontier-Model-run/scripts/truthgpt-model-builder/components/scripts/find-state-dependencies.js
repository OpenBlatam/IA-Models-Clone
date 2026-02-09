#!/usr/bin/env node

/**
 * Script para encontrar dependencias entre estados
 * Ayuda a identificar qué estados están relacionados y deben agruparse
 * 
 * Uso: node find-state-dependencies.js [ruta-al-archivo]
 */

const fs = require('fs');
const path = require('path');

const filePath = process.argv[2] || path.join(__dirname, '../ChatInterface.tsx');

if (!fs.existsSync(filePath)) {
  console.error(`❌ Archivo no encontrado: ${filePath}`);
  process.exit(1);
}

console.log(`🔍 Analizando dependencias en: ${filePath}\n`);

const content = fs.readFileSync(filePath, 'utf-8');
const lines = content.split('\n');

// Encontrar todos los useState
const useStateRegex = /const\s+\[(\w+),\s*set(\w+)\]\s*=\s*useState/g;
const states = [];
let match;

while ((match = useStateRegex.exec(content)) !== null) {
  states.push({
    name: match[1],
    setter: `set${match[2]}`,
    line: content.substring(0, match.index).split('\n').length
  });
}

console.log(`📊 Total de estados: ${states.length}\n`);

// Analizar dependencias
const dependencies = new Map();

states.forEach(({ name, setter }) => {
  const deps = {
    usedIn: [],
    dependsOn: [],
    usedTogether: []
  };
  
  // Buscar dónde se usa este estado
  const nameRegex = new RegExp(`\\b${name}\\b`, 'g');
  let nameMatch;
  while ((nameMatch = nameRegex.exec(content)) !== null) {
    const lineNum = content.substring(0, nameMatch.index).split('\n').length;
    const line = lines[lineNum - 1];
    
    // Verificar si se usa junto con otros estados
    states.forEach(otherState => {
      if (otherState.name !== name && line.includes(otherState.name)) {
        if (!deps.usedTogether.includes(otherState.name)) {
          deps.usedTogether.push(otherState.name);
        }
      }
    });
    
    // Verificar dependencias (si se usa en useEffect, useMemo, etc.)
    if (line.includes('useEffect') || line.includes('useMemo') || line.includes('useCallback')) {
      // Buscar otros estados en las dependencias
      const depMatch = line.match(/\[([^\]]+)\]/);
      if (depMatch) {
        const depsList = depMatch[1].split(',').map(d => d.trim());
        depsList.forEach(dep => {
          const depState = states.find(s => s.name === dep);
          if (depState && depState.name !== name) {
            if (!deps.dependsOn.includes(depState.name)) {
              deps.dependsOn.push(depState.name);
            }
          }
        });
      }
    }
    
    deps.usedIn.push(lineNum);
  });
  
  dependencies.set(name, deps);
});

// Agrupar estados relacionados
const groups = [];
const processed = new Set();

states.forEach(state => {
  if (processed.has(state.name)) return;
  
  const group = [state.name];
  const deps = dependencies.get(state.name);
  
  // Agregar estados usados juntos
  deps.usedTogether.forEach(otherState => {
    if (!processed.has(otherState)) {
      group.push(otherState);
      processed.add(otherState);
    }
  });
  
  // Agregar estados de los que depende
  deps.dependsOn.forEach(depState => {
    if (!processed.has(depState)) {
      group.push(depState);
      processed.add(depState);
    }
  });
  
  if (group.length > 1) {
    groups.push(group);
    group.forEach(s => processed.add(s));
  } else {
    processed.add(state.name);
  }
});

// Mostrar resultados
console.log('📊 GRUPOS DE ESTADOS RELACIONADOS\n');
console.log('═'.repeat(60));

if (groups.length > 0) {
  groups.forEach((group, index) => {
    console.log(`\n🔗 Grupo ${index + 1} (${group.length} estados relacionados):`);
    group.forEach(stateName => {
      const deps = dependencies.get(stateName);
      console.log(`   - ${stateName}`);
      if (deps.usedTogether.length > 0) {
        console.log(`     Usado junto con: ${deps.usedTogether.join(', ')}`);
      }
      if (deps.dependsOn.length > 0) {
        console.log(`     Depende de: ${deps.dependsOn.join(', ')}`);
      }
    });
    console.log(`\n   💡 Sugerencia: Considerar usarReducer o objeto de estado para este grupo`);
  });
} else {
  console.log('\n⚠️  No se encontraron grupos obvios de estados relacionados');
  console.log('   Esto puede indicar que los estados están bien separados');
  console.log('   o que el análisis necesita más contexto');
}

// Estados sin dependencias
const isolatedStates = states.filter(s => {
  const deps = dependencies.get(s.name);
  return deps.usedTogether.length === 0 && deps.dependsOn.length === 0;
});

console.log('\n' + '═'.repeat(60));
console.log(`\n📦 Estados aislados (sin dependencias obvias): ${isolatedStates.length}`);
if (isolatedStates.length > 0 && isolatedStates.length <= 20) {
  isolatedStates.forEach(s => {
    console.log(`   - ${s.name}`);
  });
}

// Sugerencias de consolidación
console.log('\n' + '═'.repeat(60));
console.log('\n💡 SUGERENCIAS DE CONSOLIDACIÓN\n');

// Buscar patrones comunes
const searchStates = states.filter(s => s.name.toLowerCase().includes('search'));
const messageStates = states.filter(s => s.name.toLowerCase().includes('message'));
const voiceStates = states.filter(s => s.name.toLowerCase().includes('voice'));
const filterStates = states.filter(s => s.name.toLowerCase().includes('filter'));
const managerStates = states.filter(s => s.name.toLowerCase().includes('manager'));

if (searchStates.length > 3) {
  console.log(`🔍 Estados de búsqueda (${searchStates.length}):`);
  console.log(`   Considerar consolidar en: useSearch hook`);
  searchStates.forEach(s => console.log(`   - ${s.name}`));
  console.log('');
}

if (messageStates.length > 5) {
  console.log(`💬 Estados de mensajes (${messageStates.length}):`);
  console.log(`   Considerar consolidar en: useMessageManagement hook`);
  messageStates.slice(0, 10).forEach(s => console.log(`   - ${s.name}`));
  if (messageStates.length > 10) {
    console.log(`   ... y ${messageStates.length - 10} más`);
  }
  console.log('');
}

if (voiceStates.length > 2) {
  console.log(`🎤 Estados de voz (${voiceStates.length}):`);
  console.log(`   Considerar consolidar en: useVoiceFeatures hook`);
  voiceStates.forEach(s => console.log(`   - ${s.name}`));
  console.log('');
}

if (filterStates.length > 2) {
  console.log(`🔧 Estados de filtros (${filterStates.length}):`);
  console.log(`   Considerar consolidar en: useFilters hook`);
  filterStates.forEach(s => console.log(`   - ${s.name}`));
  console.log('');
}

if (managerStates.length > 10) {
  console.log(`⚠️  Estados de managers (${managerStates.length}):`);
  console.log(`   ⚠️  ADVERTENCIA: Muchos estados de "manager"`);
  console.log(`   → Verificar si realmente se usan todos`);
  console.log(`   → Considerar eliminar los no usados`);
  managerStates.slice(0, 10).forEach(s => console.log(`   - ${s.name}`));
  if (managerStates.length > 10) {
    console.log(`   ... y ${managerStates.length - 10} más`);
  }
  console.log('');
}

// Generar reporte
const reportPath = path.join(__dirname, '../ChatInterface_DEPENDENCIES_REPORT.txt');
const report = `
ANÁLISIS DE DEPENDENCIAS - ChatInterface.tsx
Generado: ${new Date().toISOString()}

TOTAL DE ESTADOS: ${states.length}

GRUPOS DE ESTADOS RELACIONADOS (${groups.length}):
${groups.map((group, i) => `
Grupo ${i + 1}:
${group.map(s => `  - ${s.name}`).join('\n')}
`).join('\n')}

ESTADOS AISLADOS (${isolatedStates.length}):
${isolatedStates.map(s => `  - ${s.name}`).join('\n')}

SUGERENCIAS:
${searchStates.length > 3 ? `- Consolidar ${searchStates.length} estados de búsqueda en useSearch hook\n` : ''}
${messageStates.length > 5 ? `- Consolidar ${messageStates.length} estados de mensajes en useMessageManagement hook\n` : ''}
${voiceStates.length > 2 ? `- Consolidar ${voiceStates.length} estados de voz en useVoiceFeatures hook\n` : ''}
${managerStates.length > 10 ? `- ⚠️  Revisar ${managerStates.length} estados de manager (probablemente no usados)\n` : ''}
`;

fs.writeFileSync(reportPath, report);
console.log(`\n📄 Reporte guardado en: ${reportPath}`);
console.log('\n✅ Análisis completado!\n');




