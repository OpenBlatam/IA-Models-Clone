/**
 * Command System for CLI-like interface
 * @module robot-3d-view/lib/command-system
 */

/**
 * Command handler function
 */
export type CommandHandler = (
  args: string[],
  context: CommandContext
) => Promise<CommandResult> | CommandResult;

/**
 * Command context
 */
export interface CommandContext {
  [key: string]: unknown;
}

/**
 * Command result
 */
export interface CommandResult {
  success: boolean;
  message?: string;
  data?: unknown;
  error?: string;
}

/**
 * Command definition
 */
export interface Command {
  name: string;
  aliases?: string[];
  description: string;
  usage?: string;
  handler: CommandHandler;
  args?: CommandArg[];
}

/**
 * Command argument
 */
export interface CommandArg {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'position';
  required?: boolean;
  description?: string;
}

/**
 * Command Manager class
 */
export class CommandManager {
  private commands: Map<string, Command> = new Map();
  private history: string[] = [];
  private maxHistorySize = 100;

  /**
   * Registers a command
   */
  register(command: Command): void {
    this.commands.set(command.name.toLowerCase(), command);
    if (command.aliases) {
      command.aliases.forEach((alias) => {
        this.commands.set(alias.toLowerCase(), command);
      });
    }
  }

  /**
   * Unregisters a command
   */
  unregister(name: string): boolean {
    return this.commands.delete(name.toLowerCase());
  }

  /**
   * Gets a command
   */
  getCommand(name: string): Command | undefined {
    return this.commands.get(name.toLowerCase());
  }

  /**
   * Gets all commands
   */
  getAllCommands(): Command[] {
    const uniqueCommands = new Set<Command>();
    this.commands.forEach((cmd) => uniqueCommands.add(cmd));
    return Array.from(uniqueCommands);
  }

  /**
   * Executes a command
   */
  async execute(
    input: string,
    context: CommandContext = {}
  ): Promise<CommandResult> {
    const trimmed = input.trim();
    if (!trimmed) {
      return {
        success: false,
        error: 'Empty command',
      };
    }

    // Add to history
    this.addToHistory(trimmed);

    // Parse command
    const parts = trimmed.split(/\s+/);
    const commandName = parts[0];
    const args = parts.slice(1);

    const command = this.getCommand(commandName);
    if (!command) {
      return {
        success: false,
        error: `Command not found: ${commandName}. Type 'help' for available commands.`,
      };
    }

    // Validate arguments
    if (command.args) {
      const validation = this.validateArgs(args, command.args);
      if (!validation.valid) {
        return {
          success: false,
          error: validation.error || 'Invalid arguments',
          message: command.usage,
        };
      }
    }

    // Execute command
    try {
      const result = await command.handler(args, context);
      return result;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Validates command arguments
   */
  private validateArgs(
    args: string[],
    argDefs: CommandArg[]
  ): { valid: boolean; error?: string } {
    const requiredArgs = argDefs.filter((arg) => arg.required !== false);
    if (args.length < requiredArgs.length) {
      return {
        valid: false,
        error: `Missing required arguments. Expected ${requiredArgs.length}, got ${args.length}`,
      };
    }

    // Type validation
    for (let i = 0; i < Math.min(args.length, argDefs.length); i++) {
      const arg = args[i];
      const def = argDefs[i];

      if (def.type === 'number' && isNaN(Number(arg))) {
        return {
          valid: false,
          error: `Argument ${def.name} must be a number`,
        };
      }

      if (def.type === 'boolean' && arg !== 'true' && arg !== 'false') {
        return {
          valid: false,
          error: `Argument ${def.name} must be true or false`,
        };
      }
    }

    return { valid: true };
  }

  /**
   * Adds command to history
   */
  private addToHistory(command: string): void {
    this.history.push(command);
    if (this.history.length > this.maxHistorySize) {
      this.history.shift();
    }
  }

  /**
   * Gets command history
   */
  getHistory(): readonly string[] {
    return [...this.history];
  }

  /**
   * Clears history
   */
  clearHistory(): void {
    this.history = [];
  }

  /**
   * Autocompletes command
   */
  autocomplete(input: string): string[] {
    const trimmed = input.trim().toLowerCase();
    if (!trimmed) {
      return Array.from(this.commands.keys());
    }

    const matches: string[] = [];
    this.commands.forEach((cmd, key) => {
      if (key.startsWith(trimmed)) {
        matches.push(cmd.name);
      }
    });

    return matches;
  }
}

/**
 * Global command manager instance
 */
export const commandManager = new CommandManager();

// Register default commands
commandManager.register({
  name: 'help',
  description: 'Shows available commands',
  handler: async (args, context) => {
    const commands = commandManager.getAllCommands();
    const commandList = commands
      .map((cmd) => {
        const usage = cmd.usage || cmd.name;
        return `  ${cmd.name.padEnd(20)} - ${cmd.description}\n    Usage: ${usage}`;
      })
      .join('\n');

    return {
      success: true,
      message: `Available commands:\n\n${commandList}\n\nType 'help <command>' for detailed information.`,
    };
  },
});

commandManager.register({
  name: 'clear',
  aliases: ['cls'],
  description: 'Clears the command history',
  handler: () => {
    commandManager.clearHistory();
    return {
      success: true,
      message: 'History cleared',
    };
  },
});

commandManager.register({
  name: 'history',
  aliases: ['hist'],
  description: 'Shows command history',
  handler: () => {
    const history = commandManager.getHistory();
    const historyList = history
      .map((cmd, index) => `  ${index + 1}. ${cmd}`)
      .join('\n');

    return {
      success: true,
      message: historyList || 'No history',
    };
  },
});



