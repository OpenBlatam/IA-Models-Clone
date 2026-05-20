/**
 * Advanced Code Quality Testing
 * 
 * Comprehensive tests for code quality including
 * complexity analysis, code smells, and best practices.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('Advanced Code Quality Testing', () => {
  describe('Cyclomatic Complexity', () => {
    it('should measure function complexity', () => {
      const calculateComplexity = (code: string) => {
        const complexity = 1; // Base complexity
        const complexityIncreasers = [
          /if\s*\(/g,
          /else\s+if\s*\(/g,
          /for\s*\(/g,
          /while\s*\(/g,
          /switch\s*\(/g,
          /catch\s*\(/g,
          /\?\s*.*\s*:/g, // Ternary operators
        ];
        
        complexityIncreasers.forEach(pattern => {
          const matches = code.match(pattern);
          if (matches) {
            return matches.length;
          }
        });
        
        return complexity;
      };

      const simpleCode = 'return value;';
      const complexCode = 'if (a) { if (b) { if (c) { return; } } }';
      
      expect(calculateComplexity(simpleCode)).toBe(1);
      expect(calculateComplexity(complexCode)).toBeGreaterThan(1);
    });

    it('should flag high complexity functions', () => {
      const isComplex = (complexity: number, threshold: number = 10) => {
        return complexity > threshold;
      };

      expect(isComplex(5, 10)).toBe(false);
      expect(isComplex(15, 10)).toBe(true);
    });
  });

  describe('Code Duplication', () => {
    it('should detect duplicate code blocks', () => {
      const detectDuplicates = (codeBlocks: string[]) => {
        const seen = new Set<string>();
        const duplicates: string[] = [];
        
        codeBlocks.forEach(block => {
          const normalized = block.trim().replace(/\s+/g, ' ');
          if (seen.has(normalized)) {
            duplicates.push(block);
          } else {
            seen.add(normalized);
          }
        });
        
        return duplicates;
      };

      const blocks = [
        'function test() { return true; }',
        'function test() { return true; }',
        'function other() { return false; }',
      ];
      
      const duplicates = detectDuplicates(blocks);
      expect(duplicates.length).toBeGreaterThan(0);
    });

    it('should suggest code extraction for duplicates', () => {
      const shouldExtract = (duplicateCount: number, threshold: number = 3) => {
        return duplicateCount >= threshold;
      };

      expect(shouldExtract(5, 3)).toBe(true);
      expect(shouldExtract(2, 3)).toBe(false);
    });
  });

  describe('Naming Conventions', () => {
    it('should validate function naming', () => {
      const isValidFunctionName = (name: string) => {
        const regex = /^[a-z][a-zA-Z0-9]*$/;
        return regex.test(name) && name.length > 0;
      };

      expect(isValidFunctionName('getUserData')).toBe(true);
      expect(isValidFunctionName('GetUserData')).toBe(false); // Should be camelCase
      expect(isValidFunctionName('123invalid')).toBe(false);
    });

    it('should validate constant naming', () => {
      const isValidConstantName = (name: string) => {
        const regex = /^[A-Z][A-Z0-9_]*$/;
        return regex.test(name);
      };

      expect(isValidConstantName('MAX_RETRIES')).toBe(true);
      expect(isValidConstantName('maxRetries')).toBe(false);
    });
  });

  describe('Function Length', () => {
    it('should measure function length', () => {
      const measureFunctionLength = (code: string) => {
        const lines = code.split('\n').filter(line => line.trim().length > 0);
        return lines.length;
      };

      const shortFunction = 'function test() {\n  return true;\n}';
      const longFunction = Array(100).fill('  console.log("line");').join('\n');
      
      expect(measureFunctionLength(shortFunction)).toBeLessThan(10);
      expect(measureFunctionLength(longFunction)).toBeGreaterThan(50);
    });

    it('should flag long functions', () => {
      const isTooLong = (lineCount: number, maxLines: number = 50) => {
        return lineCount > maxLines;
      };

      expect(isTooLong(30, 50)).toBe(false);
      expect(isTooLong(60, 50)).toBe(true);
    });
  });

  describe('Dependency Analysis', () => {
    it('should detect circular dependencies', () => {
      const detectCircular = (dependencies: Record<string, string[]>) => {
        const visited = new Set<string>();
        const recursionStack = new Set<string>();
        
        const hasCycle = (module: string): boolean => {
          if (recursionStack.has(module)) return true;
          if (visited.has(module)) return false;
          
          visited.add(module);
          recursionStack.add(module);
          
          const deps = dependencies[module] || [];
          for (const dep of deps) {
            if (hasCycle(dep)) return true;
          }
          
          recursionStack.delete(module);
          return false;
        };
        
        return Object.keys(dependencies).some(module => hasCycle(module));
      };

      const circular = {
        'A': ['B'],
        'B': ['C'],
        'C': ['A'],
      };
      
      const nonCircular = {
        'A': ['B'],
        'B': ['C'],
        'C': [],
      };
      
      expect(detectCircular(circular)).toBe(true);
      expect(detectCircular(nonCircular)).toBe(false);
    });

    it('should detect unused dependencies', () => {
      const detectUnused = (imported: string[], used: string[]) => {
        return imported.filter(imp => !used.includes(imp));
      };

      const imported = ['react', 'lodash', 'axios'];
      const used = ['react', 'axios'];
      
      const unused = detectUnused(imported, used);
      expect(unused).toContain('lodash');
    });
  });

  describe('Error Handling Quality', () => {
    it('should have proper error handling', () => {
      const hasErrorHandling = (code: string) => {
        return code.includes('try') && code.includes('catch') ||
               code.includes('.catch(') ||
               code.includes('error') && code.includes('handle');
      };

      const withHandling = 'try { code(); } catch (error) { handle(error); }';
      const withoutHandling = 'code();';
      
      expect(hasErrorHandling(withHandling)).toBe(true);
      expect(hasErrorHandling(withoutHandling)).toBe(false);
    });

    it('should provide meaningful error messages', () => {
      const hasMeaningfulMessage = (error: Error) => {
        return error.message && 
               error.message.length > 10 &&
               !error.message.includes('undefined') &&
               !error.message.includes('null');
      };

      const goodError = new Error('Failed to fetch user data: Network timeout');
      const badError = new Error('Error');
      
      expect(hasMeaningfulMessage(goodError)).toBe(true);
      expect(hasMeaningfulMessage(badError)).toBe(false);
    });
  });

  describe('Type Safety', () => {
    it('should use TypeScript types', () => {
      const hasTypes = (code: string) => {
        return code.includes(':') && (
          code.includes('string') ||
          code.includes('number') ||
          code.includes('boolean') ||
          code.includes('interface') ||
          code.includes('type ')
        );
      };

      const typed = 'function test(value: string): boolean { return true; }';
      const untyped = 'function test(value) { return true; }';
      
      expect(hasTypes(typed)).toBe(true);
      expect(hasTypes(untyped)).toBe(false);
    });

    it('should avoid any types', () => {
      const hasAnyType = (code: string) => {
        return code.includes(': any') || code.includes('<any>');
      };

      const withAny = 'const data: any = {};';
      const withoutAny = 'const data: Record<string, unknown> = {};';
      
      expect(hasAnyType(withAny)).toBe(true);
      expect(hasAnyType(withoutAny)).toBe(false);
    });
  });

  describe('Code Organization', () => {
    it('should have proper file structure', () => {
      const isValidStructure = (files: string[]) => {
        const hasComponents = files.some(f => f.includes('components'));
        const hasUtils = files.some(f => f.includes('utils'));
        const hasTypes = files.some(f => f.includes('types'));
        return hasComponents && hasUtils && hasTypes;
      };

      const goodStructure = [
        'components/Button.tsx',
        'utils/helpers.ts',
        'types/index.ts',
      ];
      
      expect(isValidStructure(goodStructure)).toBe(true);
    });

    it('should have consistent import ordering', () => {
      const isOrdered = (imports: string[]) => {
        let lastType = '';
        for (const imp of imports) {
          let currentType = 'external';
          if (imp.startsWith('.')) currentType = 'internal';
          if (imp.startsWith('@/')) currentType = 'alias';
          
          if (lastType && currentType < lastType) {
            return false;
          }
          lastType = currentType;
        }
        return true;
      };

      const ordered = ['react', 'lodash', './utils', '@/components'];
      const unordered = ['./utils', 'react', '@/components'];
      
      expect(isOrdered(ordered)).toBe(true);
      expect(isOrdered(unordered)).toBe(false);
    });
  });
});

