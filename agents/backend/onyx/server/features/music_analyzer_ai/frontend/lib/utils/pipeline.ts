/**
 * Pipeline utility functions.
 * Provides helper functions for pipeline processing.
 */

/**
 * Pipeline stage.
 */
export type PipelineStage<T, R> = (input: T) => R | Promise<R>;

/**
 * Pipeline class.
 */
export class Pipeline<T> {
  private stages: Array<(input: any) => any> = [];

  /**
   * Adds a stage to the pipeline.
   */
  pipe<R>(stage: PipelineStage<T, R>): Pipeline<R> {
    const newPipeline = new Pipeline<R>();
    newPipeline.stages = [...this.stages, stage];
    return newPipeline as any;
  }

  /**
   * Executes the pipeline.
   */
  async execute(input: T): Promise<any> {
    let result: any = input;

    for (const stage of this.stages) {
      result = await stage(result);
    }

    return result;
  }

  /**
   * Executes the pipeline synchronously.
   */
  executeSync(input: T): any {
    let result: any = input;

    for (const stage of this.stages) {
      result = stage(result);
    }

    return result;
  }
}

/**
 * Creates a pipeline.
 */
export function pipeline<T>(): Pipeline<T> {
  return new Pipeline<T>();
}

