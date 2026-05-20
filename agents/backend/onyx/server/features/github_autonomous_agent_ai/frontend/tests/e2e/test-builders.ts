/**
 * Test Data Builders
 * 
 * Builders para crear datos de prueba de forma fluida y reutilizable
 */
import { Page } from '@playwright/test';
import { createTaskViaApi, type ApiTask } from './helpers';
import { generateTestId } from './test-helpers';

// ============================================================================
// Task Builder
// ============================================================================

export class TaskBuilder {
  private instruction: string = 'Test instruction';
  private repository: string = 'test/repo';
  private status: string = 'pending';
  private model: string = 'deepseek-chat';
  private repoInfo: {
    name: string;
    full_name: string;
    default_branch: string;
  } = {
    name: 'test',
    full_name: 'test/repo',
    default_branch: 'main',
  };

  withInstruction(instruction: string): this {
    this.instruction = instruction;
    return this;
  }

  withRepository(repository: string): this {
    this.repository = repository;
    const [owner, repo] = repository.split('/');
    this.repoInfo = {
      name: repo || 'repo',
      full_name: repository,
      default_branch: 'main',
    };
    return this;
  }

  withStatus(status: string): this {
    this.status = status;
    return this;
  }

  withModel(model: string): this {
    this.model = model;
    return this;
  }

  withDefaultBranch(branch: string): this {
    this.repoInfo.default_branch = branch;
    return this;
  }

  build(): {
    instruction: string;
    repository: string;
    status: string;
    repoInfo: typeof this.repoInfo;
    model: string;
  } {
    return {
      instruction: this.instruction,
      repository: this.repository,
      status: this.status,
      repoInfo: this.repoInfo,
      model: this.model,
    };
  }

  async createViaApi(page: Page): Promise<ApiTask> {
    return await createTaskViaApi(
      page,
      this.instruction,
      this.repository,
      this.model
    );
  }
}

// ============================================================================
// Test Instruction Builder
// ============================================================================

export class InstructionBuilder {
  private action: string = 'Crea';
  private target: string = 'un archivo';
  private filename: string = 'README.md';
  private content: string = 'Hello World';

  create(): this {
    this.action = 'Crea';
    return this;
  }

  update(): this {
    this.action = 'Actualiza';
    return this;
  }

  delete(): this {
    this.action = 'Elimina';
    return this;
  }

  file(filename: string): this {
    this.filename = filename;
    this.target = 'un archivo';
    return this;
  }

  folder(folderName: string): this {
    this.target = 'una carpeta';
    this.filename = folderName;
    return this;
  }

  withContent(content: string): this {
    this.content = content;
    return this;
  }

  build(): string {
    if (this.action === 'Elimina') {
      return `${this.action} ${this.target} ${this.filename}`;
    }
    return `${this.action} ${this.target} ${this.filename} con "${this.content}"`;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Crea un builder de tarea con valores por defecto
 */
export function taskBuilder(): TaskBuilder {
  return new TaskBuilder();
}

/**
 * Crea un builder de instrucción con valores por defecto
 */
export function instructionBuilder(): InstructionBuilder {
  return new InstructionBuilder();
}

/**
 * Crea una tarea de prueba simple
 */
export function createSimpleTask(): TaskBuilder {
  return taskBuilder()
    .withInstruction('Test instruction')
    .withRepository('test/repo');
}

/**
 * Crea una tarea de prueba para crear archivo
 */
export function createFileTask(filename: string = 'README.md'): TaskBuilder {
  return taskBuilder()
    .withInstruction(
      instructionBuilder().file(filename).withContent('Hello World').build()
    )
    .withRepository('test/repo');
}

/**
 * Crea una tarea de prueba con ID único
 */
export function createUniqueTask(): TaskBuilder {
  const testId = generateTestId('task');
  return taskBuilder()
    .withInstruction(`Test instruction ${testId}`)
    .withRepository(`test/repo-${testId}`);
}



