/**
 * Macro System for automation
 * @module robot-3d-view/lib/macro-system
 */

import type { SceneConfig } from '../schemas/validation-schemas';

/**
 * Macro step
 */
export interface MacroStep {
  type: 'command' | 'delay' | 'config' | 'wait';
  data: unknown;
  description?: string;
}

/**
 * Macro definition
 */
export interface Macro {
  id: string;
  name: string;
  description?: string;
  steps: MacroStep[];
  enabled: boolean;
  createdAt: number;
  updatedAt: number;
}

/**
 * Macro execution context
 */
export interface MacroContext {
  config: SceneConfig;
  onConfigChange?: (config: SceneConfig) => void;
  onCommand?: (command: string) => Promise<unknown>;
  onDelay?: (ms: number) => Promise<void>;
}

/**
 * Step handler configuration
 */
interface StepHandlerConfig {
  expectedType: 'string' | 'number' | 'object';
  contextKey: keyof MacroContext;
  execute: (data: unknown, context: MacroContext) => Promise<void>;
}

/**
 * Macro Manager class
 */
export class MacroManager {
  private macros: Map<string, Macro> = new Map();
  private isExecuting = false;
  private currentMacro: string | null = null;

  /**
   * Step handler configuration map - defines how each step type is processed
   * This replaces the switch statement with a loop-based approach
   */
  private readonly stepHandlers: Map<MacroStep['type'], StepHandlerConfig> = new Map([
    [
      'command',
      {
        expectedType: 'string',
        contextKey: 'onCommand',
        execute: async (data: unknown, context: MacroContext) => {
          if (context.onCommand && typeof data === 'string') {
            await context.onCommand(data);
          }
        },
      },
    ],
    [
      'delay',
      {
        expectedType: 'number',
        contextKey: 'onDelay',
        execute: async (data: unknown, context: MacroContext) => {
          if (context.onDelay && typeof data === 'number') {
            await context.onDelay(data);
          }
        },
      },
    ],
    [
      'config',
      {
        expectedType: 'object',
        contextKey: 'onConfigChange',
        execute: async (data: unknown, context: MacroContext) => {
          if (context.onConfigChange && typeof data === 'object' && data !== null) {
            context.onConfigChange(data as SceneConfig);
          }
        },
      },
    ],
  ]);

  /**
   * Creates a macro
   */
  createMacro(name: string, steps: MacroStep[], description?: string): Macro {
    const id = `macro-${Date.now()}`;
    const macro: Macro = {
      id,
      name,
      description,
      steps,
      enabled: true,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.macros.set(id, macro);
    return macro;
  }

  /**
   * Updates a macro
   */
  updateMacro(id: string, updates: Partial<Macro>): boolean {
    const macro = this.macros.get(id);
    if (!macro) return false;

    this.macros.set(id, {
      ...macro,
      ...updates,
      updatedAt: Date.now(),
    });
    return true;
  }

  /**
   * Deletes a macro
   */
  deleteMacro(id: string): boolean {
    return this.macros.delete(id);
  }

  /**
   * Gets a macro
   */
  getMacro(id: string): Macro | undefined {
    return this.macros.get(id);
  }

  /**
   * Gets macros with optional filter
   * Unified method that replaces getAllMacros and getEnabledMacros
   */
  getMacros(filter?: (macro: Macro) => boolean): Macro[] {
    const allMacros = Array.from(this.macros.values());
    return filter ? allMacros.filter(filter) : allMacros;
  }

  /**
   * Gets all macros
   */
  getAllMacros(): Macro[] {
    return this.getMacros();
  }

  /**
   * Gets enabled macros
   */
  getEnabledMacros(): Macro[] {
    return this.getMacros((m) => m.enabled);
  }

  /**
   * Executes a macro
   */
  async executeMacro(id: string, context: MacroContext): Promise<void> {
    // Validation checks - consolidated into a loop-based approach
    const validationErrors: string[] = [];

    if (this.isExecuting) {
      validationErrors.push('Another macro is already executing');
    }

    const macro = this.macros.get(id);
    if (!macro) {
      validationErrors.push(`Macro not found: ${id}`);
    } else if (!macro.enabled) {
      validationErrors.push(`Macro is disabled: ${id}`);
    }

    // Throw first validation error if any exist
    if (validationErrors.length > 0) {
      throw new Error(validationErrors[0]);
    }

    this.isExecuting = true;
    this.currentMacro = id;

    try {
      // Loop through all steps and execute them sequentially
      for (const step of macro!.steps) {
        await this.executeStep(step, context);
      }
    } finally {
      this.isExecuting = false;
      this.currentMacro = null;
    }
  }

  /**
   * Executes a macro step using loop-based handler lookup
   * Replaces the switch statement with a configuration-driven approach
   */
  private async executeStep(step: MacroStep, context: MacroContext): Promise<void> {
    // Handle 'wait' type separately as it doesn't follow the standard pattern
    if (step.type === 'wait') {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      return;
    }

    // Look up handler configuration for this step type
    const handler = this.stepHandlers.get(step.type);

    if (handler) {
      // Validate data type matches expected type
      const typeCheck = this.validateDataType(step.data, handler.expectedType);
      
      if (typeCheck.isValid) {
        // Execute the handler's execute function
        await handler.execute(step.data, context);
      }
    }
  }

  /**
   * Validates that data matches the expected type
   */
  private validateDataType(data: unknown, expectedType: string): { isValid: boolean } {
    const actualType = typeof data;
    
    // Special handling for object type (must be non-null)
    if (expectedType === 'object') {
      return { isValid: actualType === 'object' && data !== null };
    }
    
    return { isValid: actualType === expectedType };
  }

  /**
   * Exports a macro
   */
  exportMacro(id: string): string {
    const macro = this.macros.get(id);
    if (!macro) {
      throw new Error(`Macro not found: ${id}`);
    }

    return JSON.stringify(macro, null, 2);
  }

  /**
   * Imports a macro
   */
  importMacro(json: string): Macro {
    const macro = JSON.parse(json) as Macro;
    this.macros.set(macro.id, macro);
    return macro;
  }

  /**
   * Checks if a macro is executing
   */
  isMacroExecuting(): boolean {
    return this.isExecuting;
  }

  /**
   * Gets current executing macro
   */
  getCurrentMacro(): string | null {
    return this.currentMacro;
  }
}

/**
 * Global macro manager instance
 */
export const macroManager = new MacroManager();



