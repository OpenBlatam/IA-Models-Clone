/**
 * Page Object Model para la página de control del agente
 * 
 * Encapsula todas las interacciones con la página de control
 */
import { Page, Locator } from '@playwright/test';
import {
  navigateToAgentControl,
  createTask,
  waitForTaskToAppear,
  type TaskInfo,
  getLastTask,
} from '../helpers';

export class AgentControlPage {
  constructor(private readonly page: Page) {}

  /**
   * Navega a la página de control del agente
   */
  async goto(): Promise<void> {
    await navigateToAgentControl(this.page);
  }

  /**
   * Crea una nueva tarea desde el formulario
   */
  async createTask(instruction: string): Promise<void> {
    await createTask(this.page, instruction);
  }

  /**
   * Espera a que aparezca una tarea en la página
   */
  async waitForTask(timeout?: number): Promise<void> {
    await waitForTaskToAppear(this.page, timeout);
  }

  /**
   * Obtiene la última tarea visible
   */
  async getLastTask(): Promise<TaskInfo | null> {
    return await getLastTask(this.page);
  }

  /**
   * Obtiene el textarea de instrucciones
   */
  getInstructionTextarea(): Locator {
    return this.page.locator('textarea[name="instruction"]');
  }

  /**
   * Obtiene el botón de crear/procesar
   */
  getCreateButton(): Locator {
    return this.page.getByRole('button', { name: /crear|procesar/i });
  }

  /**
   * Verifica que la página esté cargada
   */
  async isLoaded(): Promise<boolean> {
    try {
      await this.getInstructionTextarea().waitFor({ timeout: 5000 });
      return true;
    } catch {
      return false;
    }
  }
}



