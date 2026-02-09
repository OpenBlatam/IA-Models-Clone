/**
 * Script para actualizar todos los espaciados a valores Tesla exactos
 * Ejecutar con: node scripts/update-spacing.js
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Mapeo de valores estándar a Tesla exactos
const spacingMap = {
  // Padding
  'p-0': 'p-0', // Mantener 0
  'p-1': 'p-tesla-xs', // 8px
  'p-2': 'p-tesla-sm', // 12px
  'p-3': 'p-tesla-sm', // 12px (closest)
  'p-4': 'p-tesla-md', // 16px
  'p-5': 'p-tesla-lg', // 24px
  'p-6': 'p-tesla-lg', // 24px
  'p-8': 'p-tesla-xl', // 32px
  
  // Padding X
  'px-1': 'px-tesla-xs',
  'px-2': 'px-tesla-sm',
  'px-3': 'px-tesla-sm',
  'px-4': 'px-tesla-md',
  'px-5': 'px-tesla-lg',
  'px-6': 'px-tesla-lg',
  'px-8': 'px-tesla-xl',
  
  // Padding Y
  'py-1': 'py-tesla-xs',
  'py-2': 'py-tesla-sm',
  'py-3': 'py-tesla-sm',
  'py-4': 'py-tesla-md',
  'py-5': 'py-tesla-lg',
  'py-6': 'py-tesla-lg',
  'py-8': 'py-tesla-xl',
  
  // Gap
  'gap-1': 'gap-tesla-xs',
  'gap-2': 'gap-tesla-sm',
  'gap-3': 'gap-tesla-sm',
  'gap-4': 'gap-tesla-md',
  'gap-6': 'gap-tesla-lg',
  'gap-8': 'gap-tesla-xl',
  
  // Margin
  'mb-1': 'mb-tesla-xs',
  'mb-2': 'mb-tesla-sm',
  'mb-3': 'mb-tesla-sm',
  'mb-4': 'mb-tesla-md',
  'mb-5': 'mb-tesla-lg',
  'mb-6': 'mb-tesla-lg',
  'mb-8': 'mb-tesla-xl',
  
  'mt-1': 'mt-tesla-xs',
  'mt-2': 'mt-tesla-sm',
  'mt-3': 'mt-tesla-sm',
  'mt-4': 'mt-tesla-md',
  'mt-6': 'mt-tesla-lg',
  'mt-8': 'mt-tesla-xl',
  
  // Space
  'space-y-2': 'space-y-tesla-sm',
  'space-y-3': 'space-y-tesla-sm',
  'space-y-4': 'space-y-tesla-md',
  'space-y-6': 'space-y-tesla-lg',
  
  'space-x-2': 'space-x-tesla-sm',
  'space-x-3': 'space-x-tesla-sm',
  'space-x-4': 'space-x-tesla-md',
};

function updateSpacingInFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');
  let updated = false;
  
  // Reemplazar en orden inverso (más específico primero)
  const sortedKeys = Object.keys(spacingMap).sort((a, b) => b.length - a.length);
  
  for (const oldClass of sortedKeys) {
    const newClass = spacingMap[oldClass];
    if (newClass && newClass !== oldClass) {
      // Buscar la clase como palabra completa (con espacios, comillas, o al inicio/fin)
      const regex = new RegExp(`\\b${oldClass.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g');
      if (regex.test(content)) {
        content = content.replace(regex, newClass);
        updated = true;
      }
    }
  }
  
  if (updated) {
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`✅ Updated: ${filePath}`);
    return true;
  }
  
  return false;
}

// Buscar todos los archivos .tsx y .ts en components
const files = glob.sync('components/**/*.{tsx,ts}', {
  cwd: path.join(__dirname, '..'),
  absolute: true,
});

console.log(`Found ${files.length} files to process...\n`);

let updatedCount = 0;
for (const file of files) {
  if (updateSpacingInFile(file)) {
    updatedCount++;
  }
}

console.log(`\n✅ Updated ${updatedCount} files`);



