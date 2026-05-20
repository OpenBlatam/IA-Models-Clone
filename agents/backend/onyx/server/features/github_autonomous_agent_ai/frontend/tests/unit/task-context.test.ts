/**
 * Unit tests para task-context
 */
import { describe, it, expect } from 'vitest';
import { buildTaskContext, getApiEndpoint } from '../../app/api/tasks/utils/task-context';

describe('TaskContext', () => {
  describe('buildTaskContext', () => {
    it('debería construir contexto correctamente cuando hay repoInfo', () => {
      const task = {
        repoInfo: {
          name: 'test-repo',
          full_name: 'user/test-repo',
          description: 'Test repository',
          language: 'TypeScript',
          default_branch: 'main',
          html_url: 'https://github.com/user/test-repo',
          id: 123,
        },
      };

      const context = buildTaskContext(task);

      expect(context.repo_info).toBeDefined();
      expect(context.repo_info?.name).toBe('test-repo');
      expect(context.repo_info?.full_name).toBe('user/test-repo');
      expect(context.metadata?.repository_id).toBe(123);
    });

    it('debería devolver objeto vacío cuando no hay repoInfo', () => {
      const task = {};
      const context = buildTaskContext(task);
      expect(context).toEqual({});
    });
  });

  describe('getApiEndpoint', () => {
    it('debería devolver endpoint de DeepSeek para modelos deepseek', () => {
      const endpoint = getApiEndpoint('deepseek-chat', 'http://localhost:3001');
      expect(endpoint).toBe('http://localhost:3001/api/deepseek/stream');
    });

    it('debería devolver endpoint de OpenRouter para modelos GPT', () => {
      const endpoint = getApiEndpoint('gpt-4', 'http://localhost:3001');
      expect(endpoint).toBe('http://localhost:3001/api/openrouter/stream');
    });

    it('debería devolver endpoint de OpenRouter para modelos Claude', () => {
      const endpoint = getApiEndpoint('claude-3', 'http://localhost:3001');
      expect(endpoint).toBe('http://localhost:3001/api/openrouter/stream');
    });
  });
});

