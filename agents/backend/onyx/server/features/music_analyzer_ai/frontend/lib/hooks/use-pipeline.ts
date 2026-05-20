/**
 * Custom hook for pipeline processing.
 * Provides reactive pipeline functionality.
 */

import { useMemo, useCallback } from 'react';
import { Pipeline, pipeline, PipelineStage } from '../utils/pipeline';

/**
 * Custom hook for pipeline processing.
 * Provides reactive pipeline functionality.
 *
 * @param stages - Pipeline stages
 * @returns Pipeline instance and execute function
 */
export function usePipeline<T, R>(
  stages: PipelineStage<any, any>[]
) {
  const pipelineInstance = useMemo(() => {
    let p: Pipeline<any> = pipeline<T>();
    for (const stage of stages) {
      p = p.pipe(stage);
    }
    return p as Pipeline<R>;
  }, [stages]);

  const execute = useCallback(
    async (input: T): Promise<R> => {
      return pipelineInstance.execute(input);
    },
    [pipelineInstance]
  );

  const executeSync = useCallback(
    (input: T): R => {
      return pipelineInstance.executeSync(input);
    },
    [pipelineInstance]
  );

  return {
    pipeline: pipelineInstance,
    execute,
    executeSync,
  };
}

