#!/usr/bin/env node

/**
 * Script para analizar performance del componente
 * Identifica problemas de performance potenciales
 * 
 * Uso: node performance-analyzer.js
 */

const fs = require('fs');
const path = require('path');

console.log('⚡ Analizando performance de ChatInterface.tsx...\n');

const filePath = path.join(__dirname, '../ChatInterface.tsx');

if (!fs.existsSync(filePath)) {
  console.error(`❌ Archivo no encontrado: ${filePath}`);
  process.exit(1);
}

const content = fs.readFileSync(filePath, 'utf-8');
const lines = content.split('\n');

const issues = [];
const warnings = [];
const suggestions = [];

// 1. Analizar re-renders innecesarios
console.log('🔄 Analizando re-renders...');

// Buscar componentes sin memo
const componentDefinitions = content.match(/function\s+\w+\([^)]*\)\s*\{/g) || [];
const memoizedComponents = (content.match(/memo\(/g) || []).length;
const totalComponents = componentDefinitions.length;

if (memoizedComponents < totalComponents * 0.5) {
  warnings.push(`⚠️  Solo ${memoizedComponents}/${totalComponents} componentes están memoizados`);
  suggestions.push('💡 Considerar usar React.memo() para componentes que reciben props');
}

// Buscar funciones sin useCallback
const functionDefinitions = (content.match(/const\s+\w+\s*=\s*\([^)]*\)\s*=>/g) || []).length;
const useCallbackCount = (content.match(/useCallback\(/g) || []).length;

if (functionDefinitions > useCallbackCount * 2) {
  warnings.push(`⚠️  Muchas funciones sin useCallback (${functionDefinitions} funciones, ${useCallbackCount} useCallback)`);
  suggestions.push('💡 Usar useCallback para funciones pasadas como props');
}

// 2. Analizar cálculos costosos
console.log('🧮 Analizando cálculos costosos...');

// Buscar operaciones costosas sin useMemo
const expensiveOps = [
  /\.filter\(/g,
  /\.map\(/g,
  /\.reduce\(/g,
  /\.sort\(/g,
  /\.find\(/g,
  /\.some\(/g,
  /\.every\(/g
];

let expensiveOpsCount = 0;
expensiveOps.forEach(regex => {
  const matches = content.match(regex) || [];
  expensiveOpsCount += matches.length;
});

const useMemoCount = (content.match(/useMemo\(/g) || []).length;

if (expensiveOpsCount > useMemoCount * 3) {
  warnings.push(`⚠️  Muchas operaciones costosas sin useMemo (${expensiveOpsCount} operaciones, ${useMemoCount} useMemo)`);
  suggestions.push('💡 Usar useMemo para cálculos costosos que se repiten');
}

// 3. Analizar efectos
console.log('🎯 Analizando useEffect...');

const useEffectCount = (content.match(/useEffect\(/g) || []).length;
const useEffectWithEmptyDeps = (content.match(/useEffect\([^,]*,\s*\[\s*\]/g) || []).length;
const useEffectWithManyDeps = content.match(/useEffect\([^,]*,\s*\[[^\]]{50,}\]/g) || [];

if (useEffectCount > 20) {
  warnings.push(`⚠️  Muchos useEffect (${useEffectCount}) - considerar consolidar`);
  suggestions.push('💡 Consolidar efectos relacionados en un solo useEffect');
}

if (useEffectWithManyDeps.length > 0) {
  warnings.push(`⚠️  ${useEffectWithManyDeps.length} useEffect con muchas dependencias`);
  suggestions.push('💡 Revisar si todas las dependencias son necesarias');
}

// 4. Analizar estados
console.log('📊 Analizando estados...');

const useStateCount = (content.match(/useState\(/g) || []).length;

if (useStateCount > 50) {
  issues.push(`❌ Demasiados useState (${useStateCount}) - considerar useReducer`);
  suggestions.push('💡 Consolidar estados relacionados con useReducer');
} else if (useStateCount > 30) {
  warnings.push(`⚠️  Muchos useState (${useStateCount}) - considerar consolidar`);
}

// Buscar estados que se actualizan juntos
const stateUpdates = content.match(/set\w+\(/g) || [];
const groupedUpdates = new Map();

stateUpdates.forEach(update => {
  const stateName = update.match(/set(\w+)/)?.[1];
  if (stateName) {
    const lineNum = content.substring(0, content.indexOf(update)).split('\n').length;
    const context = lines[lineNum - 1];
    
    // Buscar otros setState en la misma línea o cercanos
    const nearbyUpdates = stateUpdates.filter(u => {
      const uLineNum = content.substring(0, content.indexOf(u)).split('\n').length;
      return Math.abs(uLineNum - lineNum) <= 2;
    });
    
    if (nearbyUpdates.length > 1) {
      groupedUpdates.set(stateName, nearbyUpdates.length);
    }
  }
});

if (groupedUpdates.size > 0) {
  warnings.push(`⚠️  ${groupedUpdates.size} grupos de estados que se actualizan juntos`);
  suggestions.push('💡 Considerar consolidar estos estados en un objeto o useReducer');
}

// 5. Analizar imports y bundle size
console.log('📦 Analizando imports...');

const importLines = content.match(/^import\s+.*from\s+['"](.*?)['"]/gm) || [];
const externalImports = importLines.filter(line => {
  const match = line.match(/from\s+['"](.*?)['"]/);
  return match && !match[1].startsWith('.') && !match[1].startsWith('/');
});

const heavyLibraries = ['lodash', 'moment', 'axios', 'rxjs'];
const usedHeavyLibs = heavyLibraries.filter(lib => 
  content.includes(`from '${lib}'`) || content.includes(`from "${lib}"`)
);

if (usedHeavyLibs.length > 0) {
  warnings.push(`⚠️  Librerías pesadas detectadas: ${usedHeavyLibs.join(', ')}`);
  suggestions.push('💡 Considerar imports específicos (ej: import debounce from "lodash/debounce")');
  suggestions.push('💡 Considerar alternativas más ligeras');
}

// 6. Analizar listas grandes
console.log('📋 Analizando renderizado de listas...');

const mapRenders = (content.match(/\.map\([^)]*=>/g) || []).length;
const hasVirtualScrolling = content.includes('virtual') || content.includes('Virtual');

if (mapRenders > 5 && !hasVirtualScrolling) {
  warnings.push(`⚠️  ${mapRenders} renders de listas sin virtual scrolling`);
  suggestions.push('💡 Considerar usar react-window o react-virtual para listas grandes');
}

// 7. Analizar imágenes y assets
console.log('🖼️  Analizando assets...');

const imageImports = (content.match(/import.*\.(jpg|jpeg|png|gif|svg|webp)/gi) || []).length;
const hasLazyLoading = content.includes('lazy') || content.includes('Lazy');

if (imageImports > 10 && !hasLazyLoading) {
  warnings.push(`⚠️  ${imageImports} imágenes sin lazy loading`);
  suggestions.push('💡 Usar lazy loading para imágenes fuera del viewport');
}

// 8. Analizar code splitting
console.log('✂️  Analizando code splitting...');

const hasLazyImports = content.includes('lazy(') || content.includes('React.lazy');
const hasSuspense = content.includes('Suspense');

if (!hasLazyImports && lines.length > 1000) {
  warnings.push('⚠️  Componente grande sin code splitting');
  suggestions.push('💡 Usar React.lazy() y Suspense para cargar componentes pesados');
}

// Resumen
console.log('\n' + '═'.repeat(60));
console.log('📊 RESUMEN DE PERFORMANCE\n');

if (issues.length > 0) {
  console.log('❌ PROBLEMAS CRÍTICOS:');
  issues.forEach(msg => console.log(`   ${msg}`));
  console.log('');
}

if (warnings.length > 0) {
  console.log('⚠️  ADVERTENCIAS:');
  warnings.forEach(msg => console.log(`   ${msg}`));
  console.log('');
}

if (suggestions.length > 0) {
  console.log('💡 SUGERENCIAS:');
  suggestions.forEach(msg => console.log(`   ${msg}`));
  console.log('');
}

// Métricas
console.log('📈 MÉTRICAS:');
console.log(`   - useState: ${useStateCount}`);
console.log(`   - useEffect: ${useEffectCount}`);
console.log(`   - useCallback: ${useCallbackCount}`);
console.log(`   - useMemo: ${useMemoCount}`);
console.log(`   - Componentes: ${totalComponents}`);
console.log(`   - Componentes memoizados: ${memoizedComponents}`);
console.log(`   - Operaciones costosas: ${expensiveOpsCount}`);
console.log(`   - Renders de listas: ${mapRenders}`);

// Score
const totalIssues = issues.length * 3 + warnings.length;
const score = totalIssues === 0 ? 100 : Math.max(0, 100 - (totalIssues * 5));

console.log(`\n📊 Performance Score: ${score}/100`);

if (score >= 80) {
  console.log('🎉 ¡Excelente performance!');
} else if (score >= 60) {
  console.log('👍 Performance aceptable, hay espacio para mejoras');
} else {
  console.log('⚠️  Performance necesita mejoras significativas');
}

// Generar reporte
const reportPath = path.join(__dirname, '../ChatInterface_PERFORMANCE_REPORT.txt');
const report = `
REPORTE DE PERFORMANCE - ChatInterface.tsx
Generado: ${new Date().toISOString()}

SCORE: ${score}/100

MÉTRICAS:
- useState: ${useStateCount}
- useEffect: ${useEffectCount}
- useCallback: ${useCallbackCount}
- useMemo: ${useMemoCount}
- Componentes: ${totalComponents}
- Componentes memoizados: ${memoizedComponents}
- Operaciones costosas: ${expensiveOpsCount}
- Renders de listas: ${mapRenders}

PROBLEMAS CRÍTICOS (${issues.length}):
${issues.map(i => `  ❌ ${i}`).join('\n')}

ADVERTENCIAS (${warnings.length}):
${warnings.map(w => `  ⚠️  ${w}`).join('\n')}

SUGERENCIAS (${suggestions.length}):
${suggestions.map(s => `  💡 ${s}`).join('\n')}

PRIORIDADES:
1. ${issues.length > 0 ? issues[0] : 'Ninguna crítica'}
2. ${warnings.length > 0 ? warnings[0] : 'Ninguna'}
3. ${suggestions.length > 0 ? suggestions[0] : 'Ninguna'}
`;

fs.writeFileSync(reportPath, report);
console.log(`\n📄 Reporte guardado en: ${reportPath}\n`);




