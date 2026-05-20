#!/usr/bin/env node

/**
 * Script para analizar estados no usados en ChatInterface.tsx
 * 
 * Uso: node analyze-unused-states.js [ruta-al-archivo]
 */

const fs = require('fs');
const path = require('path');

const filePath = process.argv[2] || path.join(__dirname, '../ChatInterface.tsx');

if (!fs.existsSync(filePath)) {
  console.error(`❌ Archivo no encontrado: ${filePath}`);
  process.exit(1);
}

console.log(`📊 Analizando: ${filePath}\n`);

const content = fs.readFileSync(filePath, 'utf-8');

// Encontrar todos los useState
const useStateRegex = /const\s+\[(\w+),\s*set(\w+)\]\s*=\s*useState/g;
const states = [];
const stateLines = new Map();

let match;
let lineNumber = 1;
const lines = content.split('\n');

lines.forEach((line, index) => {
  const regex = /const\s+\[(\w+),\s*set(\w+)\]\s*=\s*useState/g;
  let m;
  while ((m = regex.exec(line)) !== null) {
    const stateName = m[1];
    const setterName = `set${m[2]}`;
    states.push({
      name: stateName,
      setter: setterName,
      line: index + 1
    });
    stateLines.set(stateName, index + 1);
  }
});

console.log(`📈 Total de estados encontrados: ${states.length}\n`);

// Analizar uso de cada estado
const analysis = {
  used: [],
  unused: [],
  readOnly: [], // Se lee pero nunca se actualiza
  writeOnly: [], // Se actualiza pero nunca se lee
  suspicious: [] // Patrones sospechosos
};

states.forEach(({ name, setter, line }) => {
  // Buscar uso del estado (excluyendo la declaración)
  const nameRegex = new RegExp(`\\b${name}\\b`, 'g');
  const setterRegex = new RegExp(`\\b${setter}\\b`, 'g');
  
  const nameMatches = content.match(nameRegex) || [];
  const setterMatches = content.match(setterRegex) || [];
  
  // Contar ocurrencias (excluyendo declaración)
  const nameCount = nameMatches.length;
  const setterCount = setterMatches.length;
  
  // Verificar si se usa en JSX
  const usedInJSX = content.includes(`{${name}}`) || 
                    content.includes(`{${name}.`) ||
                    content.includes(`[${name}]`);
  
  // Verificar si se pasa como prop
  const passedAsProp = new RegExp(`\\b${name}\\s*[=:]`, 'g').test(content);
  
  const usage = {
    name,
    setter,
    line,
    readCount: nameCount - 1, // Excluir declaración
    writeCount: setterCount - 1, // Excluir declaración
    usedInJSX,
    passedAsProp,
    totalUsage: nameCount + setterCount - 2
  };
  
  if (usage.readCount === 0 && usage.writeCount === 0) {
    analysis.unused.push(usage);
  } else if (usage.readCount > 0 && usage.writeCount === 0) {
    analysis.readOnly.push(usage);
  } else if (usage.readCount === 0 && usage.writeCount > 0) {
    analysis.writeOnly.push(usage);
  } else if (usage.readCount === 1 && usage.writeCount === 1) {
    analysis.suspicious.push(usage); // Probablemente no necesario
  } else {
    analysis.used.push(usage);
  }
});

// Mostrar resultados
console.log('📊 RESULTADOS DEL ANÁLISIS\n');
console.log('═'.repeat(60));

console.log(`\n✅ Estados USADOS: ${analysis.used.length}`);
if (analysis.used.length > 0 && analysis.used.length <= 10) {
  analysis.used.forEach(s => {
    console.log(`   - ${s.name} (línea ${s.line}) - leído ${s.readCount}x, escrito ${s.writeCount}x`);
  });
}

console.log(`\n❌ Estados NO USADOS: ${analysis.unused.length}`);
if (analysis.unused.length > 0) {
  console.log('\n   Estados que pueden ELIMINARSE:');
  analysis.unused.slice(0, 20).forEach(s => {
    console.log(`   - ${s.name} (línea ${s.line})`);
  });
  if (analysis.unused.length > 20) {
    console.log(`   ... y ${analysis.unused.length - 20} más`);
  }
}

console.log(`\n⚠️  Estados SOLO LECTURA: ${analysis.readOnly.length}`);
if (analysis.readOnly.length > 0) {
  console.log('   Estados que se leen pero nunca se actualizan:');
  analysis.readOnly.slice(0, 10).forEach(s => {
    console.log(`   - ${s.name} (línea ${s.line}) - leído ${s.readCount}x`);
  });
}

console.log(`\n⚠️  Estados SOLO ESCRITURA: ${analysis.writeOnly.length}`);
if (analysis.writeOnly.length > 0) {
  console.log('   Estados que se actualizan pero nunca se leen:');
  analysis.writeOnly.slice(0, 10).forEach(s => {
    console.log(`   - ${s.name} (línea ${s.line}) - escrito ${s.writeCount}x`);
  });
}

console.log(`\n🔍 Estados SOSPECHOSOS: ${analysis.suspicious.length}`);
if (analysis.suspicious.length > 0) {
  console.log('   Estados con uso mínimo (probablemente innecesarios):');
  analysis.suspicious.slice(0, 10).forEach(s => {
    console.log(`   - ${s.name} (línea ${s.line})`);
  });
}

// Estadísticas
console.log('\n' + '═'.repeat(60));
console.log('\n📈 ESTADÍSTICAS\n');

const totalStates = states.length;
const unusedPercentage = ((analysis.unused.length / totalStates) * 100).toFixed(1);
const suspiciousPercentage = ((analysis.suspicious.length / totalStates) * 100).toFixed(1);

console.log(`Total de estados: ${totalStates}`);
console.log(`Estados usados: ${analysis.used.length} (${((analysis.used.length / totalStates) * 100).toFixed(1)}%)`);
console.log(`Estados no usados: ${analysis.unused.length} (${unusedPercentage}%)`);
console.log(`Estados sospechosos: ${analysis.suspicious.length} (${suspiciousPercentage}%)`);

console.log(`\n💡 Potencial de reducción: ${analysis.unused.length + analysis.suspicious.length} estados (${(((analysis.unused.length + analysis.suspicious.length) / totalStates) * 100).toFixed(1)}%)`);

// Generar reporte en archivo
const reportPath = path.join(__dirname, '../ChatInterface_STATE_ANALYSIS_REPORT.txt');
const report = `
ANÁLISIS DE ESTADOS - ChatInterface.tsx
Generado: ${new Date().toISOString()}

TOTAL DE ESTADOS: ${totalStates}

ESTADOS USADOS (${analysis.used.length}):
${analysis.used.map(s => `  - ${s.name} (línea ${s.line})`).join('\n')}

ESTADOS NO USADOS - ELIMINAR (${analysis.unused.length}):
${analysis.unused.map(s => `  - ${s.name} (línea ${s.line})`).join('\n')}

ESTADOS SOLO LECTURA (${analysis.readOnly.length}):
${analysis.readOnly.map(s => `  - ${s.name} (línea ${s.line})`).join('\n')}

ESTADOS SOLO ESCRITURA (${analysis.writeOnly.length}):
${analysis.writeOnly.map(s => `  - ${s.name} (línea ${s.line})`).join('\n')}

ESTADOS SOSPECHOSOS (${analysis.suspicious.length}):
${analysis.suspicious.map(s => `  - ${s.name} (línea ${s.line})`).join('\n')}
`;

fs.writeFileSync(reportPath, report);
console.log(`\n📄 Reporte guardado en: ${reportPath}`);

// Sugerencias
console.log('\n💡 SUGERENCIAS:\n');
if (analysis.unused.length > 50) {
  console.log('   ⚠️  Tienes MÁS DE 50 estados no usados!');
  console.log('   → Prioridad ALTA: Eliminar estados no usados');
  console.log('   → Esto mejorará performance y mantenibilidad significativamente');
}

if (analysis.suspicious.length > 20) {
  console.log('   ⚠️  Tienes muchos estados con uso mínimo');
  console.log('   → Considera consolidar estados relacionados');
  console.log('   → Usa useReducer para estados relacionados');
}

console.log('\n✅ Análisis completado!\n');




