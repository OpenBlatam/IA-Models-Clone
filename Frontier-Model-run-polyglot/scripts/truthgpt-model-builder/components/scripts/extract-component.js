#!/usr/bin/env node

/**
 * Script para extraer componentes de ChatInterface.tsx
 * Automatiza la extracción de JSX a componentes separados
 * 
 * Uso: node extract-component.js <component-name> <start-line> <end-line> [props...]
 * 
 * Ejemplo:
 * node extract-component.js MessageList 500 800 message messages onMessageClick
 */

const fs = require('fs');
const path = require('path');

const [componentName, startLine, endLine, ...props] = process.argv.slice(2);

if (!componentName || !startLine || !endLine) {
  console.error('❌ Uso: node extract-component.js <component-name> <start-line> <end-line> [props...]');
  console.error('');
  console.error('Ejemplo:');
  console.error('  node extract-component.js MessageList 500 800 messages onMessageClick');
  process.exit(1);
}

const filePath = path.join(__dirname, '../ChatInterface.tsx');
const outputDir = path.join(__dirname, '../ChatInterface/components', componentName);

if (!fs.existsSync(filePath)) {
  console.error(`❌ Archivo no encontrado: ${filePath}`);
  process.exit(1);
}

console.log(`🔧 Extrayendo componente: ${componentName}`);
console.log(`📄 Líneas: ${startLine}-${endLine}`);
console.log(`📦 Props: ${props.length > 0 ? props.join(', ') : 'ninguna'}\n`);

// Leer archivo
const content = fs.readFileSync(filePath, 'utf-8');
const lines = content.split('\n');

// Extraer código
const start = parseInt(startLine) - 1;
const end = parseInt(endLine);
const extractedLines = lines.slice(start, end);
const extractedCode = extractedLines.join('\n');

// Analizar imports necesarios
const imports = new Set();
const usedHooks = new Set();
const usedTypes = new Set();

// Buscar imports en el código extraído
extractedCode.match(/import\s+.*?from\s+['"](.*?)['"]/g)?.forEach(imp => {
  imports.add(imp);
});

// Buscar hooks usados
['useState', 'useEffect', 'useCallback', 'useMemo', 'useRef', 'useContext'].forEach(hook => {
  if (extractedCode.includes(hook)) {
    usedHooks.add(hook);
  }
});

// Buscar tipos
extractedCode.match(/(?:interface|type)\s+(\w+)/g)?.forEach(match => {
  const typeName = match.match(/(?:interface|type)\s+(\w+)/)?.[1];
  if (typeName) usedTypes.add(typeName);
});

// Crear directorio
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Generar props interface
const propsInterface = props.length > 0
  ? `interface ${componentName}Props {
${props.map(prop => `  ${prop}: ${inferType(prop, extractedCode)}`).join('\n')}
}`
  : `interface ${componentName}Props {
  // Agregar props aquí
}`;

// Generar imports necesarios
const importsCode = Array.from(usedHooks).length > 0
  ? `import { ${Array.from(usedHooks).join(', ')} } from 'react'`
  : `import React from 'react'`;

// Generar componente
const componentCode = `${importsCode}
${usedTypes.size > 0 ? `// TODO: Importar tipos necesarios` : ''}

${propsInterface}

export function ${componentName}({ ${props.join(', ')} }: ${componentName}Props) {
${extractedCode
  .split('\n')
  .map(line => {
    // Remover indentación excesiva
    const trimmed = line.trim();
    if (trimmed === '') return '';
    // Mantener indentación relativa pero reducir
    const indent = line.match(/^(\s*)/)?.[1] || '';
    const reducedIndent = indent.length > 2 ? indent.slice(2) : '';
    return reducedIndent + trimmed;
  })
  .filter(line => {
    // Filtrar líneas que no son parte del componente
    return !line.includes('export default function ChatInterface') &&
           !line.includes('function ChatInterface');
  })
  .join('\n')
  .split('\n')
  .map((line, idx) => {
    // Ajustar indentación para el cuerpo del componente
    if (idx === 0) return line;
    if (line.trim() === '') return '';
    return '  ' + line;
  })
  .join('\n')}
}
`;

// Generar archivo del componente
const componentFilePath = path.join(outputDir, `${componentName}.tsx`);
fs.writeFileSync(componentFilePath, componentCode);

// Generar index.ts
const indexCode = `export { ${componentName} } from './${componentName}';
export type { ${componentName}Props } from './${componentName}';
`;
fs.writeFileSync(path.join(outputDir, 'index.ts'), indexCode);

// Generar archivo de tests template
const testCode = `import { render, screen } from '@testing-library/react';
import { ${componentName} } from './${componentName}';

describe('${componentName}', () => {
  const defaultProps = {
${props.map(prop => `    ${prop}: ${getDefaultValue(prop)}`).join(',\n')}
  };

  it('should render correctly', () => {
    render(<${componentName} {...defaultProps} />);
    // TODO: Agregar assertions
  });

  // TODO: Agregar más tests
});
`;
fs.writeFileSync(path.join(outputDir, `${componentName}.test.tsx`), testCode);

// Generar README para el componente
const readmeCode = `# ${componentName}

## Descripción
Componente extraído de ChatInterface.tsx (líneas ${startLine}-${endLine})

## Props
${props.length > 0 
  ? props.map(prop => `- \`${prop}\`: ${inferType(prop, extractedCode)}`).join('\n')
  : 'Ninguna prop definida aún'
}

## Uso
\`\`\`tsx
import { ${componentName} } from './ChatInterface/components/${componentName}';

<${componentName} ${props.map(p => `${p}={...}`).join(' ')} />
\`\`\`

## Tests
Ejecutar tests: \`npm test ${componentName}\`

## Notas
- Componente extraído automáticamente
- Revisar y ajustar según sea necesario
- Agregar documentación adicional
`;
fs.writeFileSync(path.join(outputDir, 'README.md'), readmeCode);

console.log('✅ Componente extraído exitosamente!\n');
console.log(`📁 Ubicación: ${outputDir}`);
console.log(`📄 Archivos creados:`);
console.log(`   - ${componentName}.tsx (componente)`);
console.log(`   - index.ts (exports)`);
console.log(`   - ${componentName}.test.tsx (tests template)`);
console.log(`   - README.md (documentación)`);
console.log('\n⚠️  PRÓXIMOS PASOS:');
console.log('1. Revisar y ajustar el código extraído');
console.log('2. Verificar imports y tipos');
console.log('3. Completar tests');
console.log('4. Integrar en ChatInterface.tsx');
console.log(`5. Reemplazar código original (líneas ${startLine}-${endLine}) con:`);
console.log(`   <${componentName} ${props.map(p => `${p}={${p}}`).join(' ')} />`);

// Funciones auxiliares
function inferType(propName, code) {
  // Intentar inferir tipo del código
  if (propName.includes('messages') || propName.includes('Message')) {
    return 'Message[]';
  }
  if (propName.includes('on') || propName.includes('handle')) {
    return '() => void';
  }
  if (propName.includes('is') || propName.includes('show') || propName.includes('has')) {
    return 'boolean';
  }
  if (propName.includes('count') || propName.includes('index') || propName.includes('id')) {
    return 'number';
  }
  return 'any';
}

function getDefaultValue(propName) {
  if (propName.includes('on') || propName.includes('handle')) {
    return '() => {}';
  }
  if (propName.includes('is') || propName.includes('show') || propName.includes('has')) {
    return 'false';
  }
  if (propName.includes('messages') || propName.includes('items') || propName.includes('list')) {
    return '[]';
  }
  if (propName.includes('count') || propName.includes('index') || propName.includes('id')) {
    return '0';
  }
  return 'null';
}




