/**
 * Unit tests para verificar la resolución de URLs
 */
import { describe, it, expect } from 'vitest';
import { getApiEndpoint } from '../../app/api/tasks/utils/task-context';

describe('URL Resolution', () => {
  it('debería construir la URL correcta para diferentes puertos', () => {
    const testCases = [
      { baseUrl: 'http://localhost:3000', expected: 'http://localhost:3000/api/deepseek/stream' },
      { baseUrl: 'http://localhost:3001', expected: 'http://localhost:3001/api/deepseek/stream' },
      { baseUrl: 'https://example.com', expected: 'https://example.com/api/deepseek/stream' },
    ];

    testCases.forEach(({ baseUrl, expected }) => {
      const endpoint = getApiEndpoint('deepseek-chat', baseUrl);
      expect(endpoint).toBe(expected);
    });
  });

  it('debería usar el puerto correcto según la variable de entorno', () => {
    const originalEnv = process.env.NEXT_PUBLIC_APP_URL;
    
    // Test con puerto 3000
    process.env.NEXT_PUBLIC_APP_URL = 'http://localhost:3000';
    const endpoint3000 = getApiEndpoint('deepseek-chat', process.env.NEXT_PUBLIC_APP_URL);
    expect(endpoint3000).toBe('http://localhost:3000/api/deepseek/stream');

    // Test con puerto 3001
    process.env.NEXT_PUBLIC_APP_URL = 'http://localhost:3001';
    const endpoint3001 = getApiEndpoint('deepseek-chat', process.env.NEXT_PUBLIC_APP_URL);
    expect(endpoint3001).toBe('http://localhost:3001/api/deepseek/stream');

    // Restaurar
    if (originalEnv) {
      process.env.NEXT_PUBLIC_APP_URL = originalEnv;
    } else {
      delete process.env.NEXT_PUBLIC_APP_URL;
    }
  });
});

