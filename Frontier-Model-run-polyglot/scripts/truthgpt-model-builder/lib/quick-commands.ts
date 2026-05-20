/**
 * Quick Commands
 * Sistema de comandos rápidos
 */

export interface QuickCommand {
  id: string
  name: string
  description: string
  shortcut?: string
  icon?: string
  action: () => void | Promise<void>
  category?: string
  tags?: string[]
}

export class QuickCommands {
  private commands: Map<string, QuickCommand> = new Map()
  private categories: Map<string, QuickCommand[]> = new Map()

  /**
   * Registrar comando
   */
  registerCommand(command: QuickCommand): void {
    this.commands.set(command.id, command)

    const category = command.category || 'general'
    if (!this.categories.has(category)) {
      this.categories.set(category, [])
    }
    this.categories.get(category)!.push(command)
  }

  /**
   * Eliminar comando
   */
  unregisterCommand(id: string): void {
    const command = this.commands.get(id)
    if (command) {
      const category = command.category || 'general'
      const categoryCommands = this.categories.get(category)
      if (categoryCommands) {
        this.categories.set(category, categoryCommands.filter(c => c.id !== id))
      }
      this.commands.delete(id)
    }
  }

  /**
   * Ejecutar comando
   */
  async executeCommand(id: string): Promise<boolean> {
    const command = this.commands.get(id)
    if (!command) return false

    try {
      await command.action()
      return true
    } catch (error) {
      console.error(`Error executing command ${id}:`, error)
      return false
    }
  }

  /**
   * Ejecutar comando por shortcut
   */
  async executeByShortcut(shortcut: string): Promise<boolean> {
    const command = Array.from(this.commands.values()).find(
      c => c.shortcut && c.shortcut.toLowerCase() === shortcut.toLowerCase()
    )

    if (command) {
      return await this.executeCommand(command.id)
    }

    return false
  }

  /**
   * Obtener comando
   */
  getCommand(id: string): QuickCommand | undefined {
    return this.commands.get(id)
  }

  /**
   * Obtener todos los comandos
   */
  getAllCommands(): QuickCommand[] {
    return Array.from(this.commands.values())
  }

  /**
   * Obtener comandos por categoría
   */
  getCommandsByCategory(category: string): QuickCommand[] {
    return this.categories.get(category) || []
  }

  /**
   * Buscar comandos
   */
  searchCommands(query: string): QuickCommand[] {
    const queryLower = query.toLowerCase()
    return Array.from(this.commands.values()).filter(
      command =>
        command.name.toLowerCase().includes(queryLower) ||
        command.description.toLowerCase().includes(queryLower) ||
        command.tags?.some(tag => tag.toLowerCase().includes(queryLower))
    )
  }

  /**
   * Obtener categorías
   */
  getCategories(): string[] {
    return Array.from(this.categories.keys()).sort()
  }

  /**
   * Limpiar comandos
   */
  clear(): void {
    this.commands.clear()
    this.categories.clear()
  }
}

// Singleton instance
let quickCommandsInstance: QuickCommands | null = null

export function getQuickCommands(): QuickCommands {
  if (!quickCommandsInstance) {
    quickCommandsInstance = new QuickCommands()
  }
  return quickCommandsInstance
}

export default QuickCommands










