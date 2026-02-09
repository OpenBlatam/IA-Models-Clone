#!/usr/bin/env node

/**
 * Script para generar template de hook desde estados existentes
 * 
 * Uso: node extract-hook-template.js [nombre-del-hook] [estados...]
 * Ejemplo: node extract-hook-template.js useSearch searchQuery filteredMessages
 */

const fs = require('fs');
const path = require('path');

const hookName = process.argv[2];
const stateNames = process.argv.slice(3);

if (!hookName) {
  console.error('❌ Error: Debes proporcionar un nombre para el hook');
  console.log('Uso: node extract-hook-template.js [nombre-hook] [estado1] [estado2] ...');
  process.exit(1);
}

if (stateNames.length === 0) {
  console.error('❌ Error: Debes proporcionar al menos un estado');
  console.log('Uso: node extract-hook-template.js [nombre-hook] [estado1] [estado2] ...');
  process.exit(1);
}

console.log(`📝 Generando template para hook: ${hookName}`);
console.log(`📦 Estados a incluir: ${stateNames.join(', ')}\n`);

// Generar template del hook
const hookTemplate = `import { useState, useCallback, useMemo, useEffect } from 'react'

/**
 * Hook: ${hookName}
 * 
 * Descripción: [Describe qué hace este hook]
 * 
 * Estados incluidos:
${stateNames.map(name => ` * - ${name}`).join('\n')}
 */

interface ${hookName}State {
${stateNames.map(name => `  ${name}: any // TODO: Tipar correctamente`).join('\n')}
}

interface ${hookName}Actions {
${stateNames.map(name => {
  const setterName = `set${name.charAt(0).toUpperCase() + name.slice(1)}`;
  return `  ${setterName}: (value: any) => void // TODO: Tipar correctamente`;
}).join('\n')}
  // Agregar más acciones según sea necesario
}

export function ${hookName}(): ${hookName}State & ${hookName}Actions {
${stateNames.map(name => {
  const setterName = `set${name.charAt(0).toUpperCase() + name.slice(1)}`;
  return `  const [${name}, ${setterName}] = useState<any>(null) // TODO: Valor inicial correcto`;
}).join('\n')}
  
  // TODO: Agregar lógica del hook aquí
  // - useMemo para cálculos derivados
  // - useCallback para funciones
  // - useEffect para efectos secundarios
  
  return {
    // State
${stateNames.map(name => `    ${name},`).join('\n')}
    
    // Actions
${stateNames.map(name => {
  const setterName = `set${name.charAt(0).toUpperCase() + name.slice(1)}`;
  return `    ${setterName},`;
}).join('\n')}
  }
}
`;

// Generar template de test
const testTemplate = `import { renderHook, act } from '@testing-library/react'
import { ${hookName} } from './${hookName}'

describe('${hookName}', () => {
  it('should initialize with default values', () => {
    const { result } = renderHook(() => ${hookName}())
    
    // TODO: Verificar valores iniciales
${stateNames.map(name => `    expect(result.current.${name}).toBeDefined()`).join('\n')}
  })
  
  it('should update state correctly', () => {
    const { result } = renderHook(() => ${hookName}())
    
    // TODO: Agregar tests de actualización
    // act(() => {
    //   result.current.set${stateNames[0].charAt(0).toUpperCase() + stateNames[0].slice(1)}(newValue)
    // })
    // 
    // expect(result.current.${stateNames[0]}).toBe(newValue)
  })
  
  // TODO: Agregar más tests según la lógica del hook
})
`;

// Crear directorio si no existe
const hooksDir = path.join(__dirname, '../ChatInterface/hooks');
if (!fs.existsSync(hooksDir)) {
  fs.mkdirSync(hooksDir, { recursive: true });
}

// Guardar hook
const hookPath = path.join(hooksDir, `${hookName}.ts`);
fs.writeFileSync(hookPath, hookTemplate);
console.log(`✅ Hook creado: ${hookPath}`);

// Crear directorio de tests si no existe
const testsDir = path.join(__dirname, '../ChatInterface/hooks/__tests__');
if (!fs.existsSync(testsDir)) {
  fs.mkdirSync(testsDir, { recursive: true });
}

// Guardar test
const testPath = path.join(testsDir, `${hookName}.test.ts`);
fs.writeFileSync(testPath, testTemplate);
console.log(`✅ Test creado: ${testPath}`);

console.log('\n📝 Próximos pasos:');
console.log('   1. Completar tipos TypeScript');
console.log('   2. Implementar lógica del hook');
console.log('   3. Escribir tests');
console.log('   4. Integrar en componente principal\n');




