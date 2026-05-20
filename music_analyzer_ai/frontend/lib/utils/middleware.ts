/**
 * Middleware utility functions.
 * Provides helper functions for middleware pattern.
 */

/**
 * Middleware function type.
 */
export type Middleware<TContext = any> = (
  context: TContext,
  next: () => Promise<void> | void
) => Promise<void> | void;

/**
 * Middleware chain class.
 */
export class MiddlewareChain<TContext = any> {
  private middlewares: Middleware<TContext>[] = [];

  /**
   * Adds middleware to chain.
   */
  use(middleware: Middleware<TContext>): this {
    this.middlewares.push(middleware);
    return this;
  }

  /**
   * Executes middleware chain.
   */
  async execute(context: TContext): Promise<void> {
    let index = 0;

    const next = async (): Promise<void> => {
      if (index < this.middlewares.length) {
        const middleware = this.middlewares[index++];
        await middleware(context, next);
      }
    };

    await next();
  }
}

/**
 * Creates a middleware chain.
 */
export function createMiddlewareChain<TContext = any>(): MiddlewareChain<TContext> {
  return new MiddlewareChain<TContext>();
}

