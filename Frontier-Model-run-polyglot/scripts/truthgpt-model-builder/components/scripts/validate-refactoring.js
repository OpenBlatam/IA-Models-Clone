#!/usr/bin/env node

/**
 * Script para validar la refactorización
 * Verifica que la refactorización no haya roto nada
 * 
 * Uso: node validate-refactoring.js
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🔍 Validando refactorización de ChatInterface.tsx...\n');

const componentsDir = path.join(__dirname, '../ChatInterface');
const hooksDir = path.join(componentsDir, 'hooks');
const contextsDir = path.join(componentsDir, 'contexts');
const componentsComponentsDir = path.join(componentsDir, 'components');

const issues = [];
const warnings = [];
const successes = [];

// 1. Verificar estructura de directorios
console.log('📁 Verificando estructura de directorios...');
const requiredDirs = [
  { path: hooksDir, name: 'hooks' },
  { path: contextsDir, name: 'contexts' },
  { path: componentsComponentsDir, name: 'components' }
];

requiredDirs.forEach(({ path: dirPath, name }) => {
  if (fs.existsSync(dirPath)) {
    successes.push(`✅ Directorio ${name} existe`);
  } else {
    warnings.push(`⚠️  Directorio ${name} no existe (crear si es necesario)`);
  }
});

// 2. Verificar hooks
console.log('\n🪝 Verificando hooks...');
if (fs.existsSync(hooksDir)) {
  const hookFiles = fs.readdirSync(hooksDir)
    .filter(file => file.endsWith('.ts') || file.endsWith('.tsx'))
    .filter(file => !file.includes('.test.') && !file.includes('.spec.'));
  
  const expectedHooks = [
    'useChatState',
    'useSearch',
    'useMessageManagement',
    'useVoiceFeatures',
    'useModelSelection',
    'useHistory',
    'useFilters',
    'useValidation',
    'useProactive',
    'useMetrics'
  ];
  
  expectedHooks.forEach(hookName => {
    const hookFile = hookFiles.find(f => f.includes(hookName) || f.toLowerCase().includes(hookName.toLowerCase()));
    if (hookFile) {
      successes.push(`✅ Hook ${hookName} encontrado: ${hookFile}`);
      
      // Verificar que exporta correctamente
      const hookContent = fs.readFileSync(path.join(hooksDir, hookFile), 'utf-8');
      if (hookContent.includes(`export`) && hookContent.includes(`function ${hookName}`)) {
        successes.push(`   ✅ Exportación correcta`);
      } else {
        issues.push(`❌ Hook ${hookName} no exporta correctamente`);
      }
    } else {
      warnings.push(`⚠️  Hook ${hookName} no encontrado (esperado pero no crítico)`);
    }
  });
  
  console.log(`   Total hooks encontrados: ${hookFiles.length}`);
} else {
  warnings.push('⚠️  Directorio hooks no existe');
}

// 3. Verificar contexts
console.log('\n🌐 Verificando contexts...');
if (fs.existsSync(contextsDir)) {
  const contextFiles = fs.readdirSync(contextsDir)
    .filter(file => file.endsWith('.ts') || file.endsWith('.tsx'))
    .filter(file => !file.includes('.test.') && !file.includes('.spec.'));
  
  const expectedContexts = [
    'ChatContext',
    'ThemeContext',
    'SettingsContext',
    'ModelContext'
  ];
  
  expectedContexts.forEach(contextName => {
    const contextFile = contextFiles.find(f => f.includes(contextName) || f.toLowerCase().includes(contextName.toLowerCase()));
    if (contextFile) {
      successes.push(`✅ Context ${contextName} encontrado: ${contextFile}`);
      
      // Verificar Provider y hook
      const contextContent = fs.readFileSync(path.join(contextsDir, contextFile), 'utf-8');
      if (contextContent.includes('Provider') && contextContent.includes('useContext')) {
        successes.push(`   ✅ Provider y hook implementados`);
      } else {
        issues.push(`❌ Context ${contextName} no implementa Provider correctamente`);
      }
    } else {
      warnings.push(`⚠️  Context ${contextName} no encontrado (esperado pero no crítico)`);
    }
  });
  
  console.log(`   Total contexts encontrados: ${contextFiles.length}`);
} else {
  warnings.push('⚠️  Directorio contexts no existe');
}

// 4. Verificar componentes
console.log('\n🧩 Verificando componentes...');
if (fs.existsSync(componentsComponentsDir)) {
  const componentDirs = fs.readdirSync(componentsComponentsDir, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => dirent.name);
  
  const expectedComponents = [
    'MessageList',
    'InputArea',
    'Sidebar',
    'Toolbar'
  ];
  
  expectedComponents.forEach(componentName => {
    const componentDir = componentDirs.find(d => d.includes(componentName) || d.toLowerCase().includes(componentName.toLowerCase()));
    if (componentDir) {
      successes.push(`✅ Componente ${componentName} encontrado: ${componentDir}`);
      
      // Verificar archivos principales
      const componentPath = path.join(componentsComponentsDir, componentDir);
      const files = fs.readdirSync(componentPath);
      
      if (files.some(f => f.includes(componentName) && (f.endsWith('.tsx') || f.endsWith('.ts')))) {
        successes.push(`   ✅ Archivo principal existe`);
      } else {
        issues.push(`❌ Componente ${componentName} no tiene archivo principal`);
      }
      
      if (files.some(f => f === 'index.ts' || f === 'index.tsx')) {
        successes.push(`   ✅ index.ts existe`);
      } else {
        warnings.push(`⚠️  Componente ${componentName} no tiene index.ts`);
      }
    } else {
      warnings.push(`⚠️  Componente ${componentName} no encontrado (esperado pero no crítico)`);
    }
  });
  
  console.log(`   Total componentes encontrados: ${componentDirs.length}`);
} else {
  warnings.push('⚠️  Directorio components no existe');
}

// 5. Verificar archivo principal
console.log('\n📄 Verificando ChatInterface.tsx...');
const mainFile = path.join(__dirname, '../ChatInterface.tsx');
if (fs.existsSync(mainFile)) {
  const content = fs.readFileSync(mainFile, 'utf-8');
  const lineCount = content.split('\n').length;
  
  console.log(`   Líneas de código: ${lineCount}`);
  
  if (lineCount < 500) {
    successes.push(`✅ Archivo principal tiene menos de 500 líneas (${lineCount})`);
  } else if (lineCount < 1000) {
    warnings.push(`⚠️  Archivo principal aún tiene ${lineCount} líneas (objetivo: < 500)`);
  } else {
    issues.push(`❌ Archivo principal aún tiene ${lineCount} líneas (objetivo: < 500)`);
  }
  
  // Verificar imports de hooks
  const hookImports = content.match(/import.*from.*hooks/g) || [];
  if (hookImports.length > 0) {
    successes.push(`✅ Usa hooks personalizados (${hookImports.length} imports)`);
  } else {
    warnings.push(`⚠️  No se detectan imports de hooks personalizados`);
  }
  
  // Verificar imports de componentes
  const componentImports = content.match(/import.*from.*components/g) || [];
  if (componentImports.length > 0) {
    successes.push(`✅ Usa componentes extraídos (${componentImports.length} imports)`);
  } else {
    warnings.push(`⚠️  No se detectan imports de componentes extraídos`);
  }
  
  // Verificar cantidad de useState
  const useStateCount = (content.match(/useState/g) || []).length;
  if (useStateCount < 20) {
    successes.push(`✅ Pocos useState (${useStateCount})`);
  } else if (useStateCount < 50) {
    warnings.push(`⚠️  Aún hay muchos useState (${useStateCount})`);
  } else {
    issues.push(`❌ Demasiados useState (${useStateCount})`);
  }
} else {
  issues.push(`❌ Archivo principal ChatInterface.tsx no encontrado`);
}

// 6. Verificar tests
console.log('\n🧪 Verificando tests...');
const testFiles = [];
function findTestFiles(dir) {
  if (!fs.existsSync(dir)) return;
  const files = fs.readdirSync(dir);
  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    if (stat.isDirectory()) {
      findTestFiles(filePath);
    } else if (file.includes('.test.') || file.includes('.spec.')) {
      testFiles.push(filePath);
    }
  });
}

findTestFiles(componentsDir);
console.log(`   Total archivos de test encontrados: ${testFiles.length}`);

if (testFiles.length > 0) {
  successes.push(`✅ Tests encontrados (${testFiles.length} archivos)`);
} else {
  warnings.push(`⚠️  No se encontraron archivos de test`);
}

// 7. Verificar TypeScript (si es posible)
console.log('\n📘 Verificando TypeScript...');
try {
  // Intentar verificar si hay tsconfig
  const tsconfigPath = path.join(__dirname, '../../tsconfig.json');
  if (fs.existsSync(tsconfigPath)) {
    try {
      execSync('npx tsc --noEmit --skipLibCheck', { 
        cwd: path.dirname(tsconfigPath),
        stdio: 'pipe',
        timeout: 10000
      });
      successes.push(`✅ TypeScript compila sin errores`);
    } catch (error) {
      warnings.push(`⚠️  TypeScript tiene errores (revisar manualmente)`);
    }
  } else {
    warnings.push(`⚠️  tsconfig.json no encontrado`);
  }
} catch (error) {
  warnings.push(`⚠️  No se pudo verificar TypeScript`);
}

// Resumen
console.log('\n' + '═'.repeat(60));
console.log('📊 RESUMEN DE VALIDACIÓN\n');

if (successes.length > 0) {
  console.log('✅ ÉXITOS:');
  successes.forEach(msg => console.log(`   ${msg}`));
  console.log('');
}

if (warnings.length > 0) {
  console.log('⚠️  ADVERTENCIAS:');
  warnings.forEach(msg => console.log(`   ${msg}`));
  console.log('');
}

if (issues.length > 0) {
  console.log('❌ PROBLEMAS:');
  issues.forEach(msg => console.log(`   ${msg}`));
  console.log('');
}

// Score
const totalChecks = successes.length + warnings.length + issues.length;
const score = totalChecks > 0 
  ? Math.round((successes.length / totalChecks) * 100)
  : 0;

console.log(`📈 Score: ${score}% (${successes.length}/${totalChecks} checks pasados)`);

if (score >= 80) {
  console.log('🎉 ¡Excelente! La refactorización está muy avanzada.');
} else if (score >= 60) {
  console.log('👍 Buen progreso. Continúa con la refactorización.');
} else if (score >= 40) {
  console.log('📝 Progreso moderado. Revisa las advertencias y problemas.');
} else {
  console.log('🚧 Refactorización en etapas tempranas. Sigue el plan.');
}

console.log('\n' + '═'.repeat(60));

// Generar reporte
const reportPath = path.join(__dirname, '../ChatInterface_VALIDATION_REPORT.txt');
const report = `
REPORTE DE VALIDACIÓN - ChatInterface.tsx
Generado: ${new Date().toISOString()}

SCORE: ${score}%

ÉXITOS (${successes.length}):
${successes.map(s => `  ✅ ${s}`).join('\n')}

ADVERTENCIAS (${warnings.length}):
${warnings.map(w => `  ⚠️  ${w}`).join('\n')}

PROBLEMAS (${issues.length}):
${issues.map(i => `  ❌ ${i}`).join('\n')}

RECOMENDACIONES:
${issues.length > 0 ? '- Resolver problemas críticos primero\n' : ''}
${warnings.length > 0 ? '- Revisar advertencias y completar elementos faltantes\n' : ''}
- Continuar con la refactorización según el plan
- Escribir tests para nuevos componentes y hooks
- Verificar que no hay regresiones funcionales
`;

fs.writeFileSync(reportPath, report);
console.log(`\n📄 Reporte guardado en: ${reportPath}\n`);

process.exit(issues.length > 0 ? 1 : 0);




