/**
 * Proactive Model Builder
 * Construye modelos continuamente adaptados a optimization_core
 * Sin parar, genera y adapta modelos basados en descripciones
 */

import { ModelSpec } from './modules/management'
import { adaptiveAnalyze } from './adaptive-analyzer'
import { 
  adaptToOptimizationCore, 
  generateOptimizationCoreYAML,
  generateTrainingScript,
  OptimizationCoreConfig
} from './optimization-core-adapter'
import { createGitHubRepository } from './github-service'
import { sanitizeModelName, generateModelId } from './utils'
import fs from 'fs'
import path from 'path'

export interface ProactiveBuildOptions {
  modelName: string
  description: string
  spec: ModelSpec
  optimizationCorePath: string
  outputDir?: string
  createGitHubRepo?: boolean
  continuous?: boolean
}

export interface ProactiveBuildResult {
  modelId: string
  modelName: string
  configPath: string
  scriptPath: string
  githubUrl?: string
  config: OptimizationCoreConfig
  status: 'creating' | 'completed' | 'failed'
  error?: string
}

export class ProactiveModelBuilder {
  private optimizationCorePath: string
  private outputBaseDir: string
  private isBuilding: boolean = false
  private buildQueue: ProactiveBuildOptions[] = []
  private currentBuild: ProactiveBuildOptions | null = null

  constructor(optimizationCorePath: string, outputBaseDir: string = './generated_models') {
    this.optimizationCorePath = optimizationCorePath
    this.outputBaseDir = outputBaseDir
    
    // Asegurar que el directorio base existe
    if (!fs.existsSync(this.outputBaseDir)) {
      fs.mkdirSync(this.outputBaseDir, { recursive: true })
    }
  }

  /**
   * Construye un modelo de forma proactiva y continua
   */
  async buildModelProactively(options: ProactiveBuildOptions): Promise<ProactiveBuildResult> {
    const modelId = generateModelId()
    const modelName = options.modelName || `truthgpt-${sanitizeModelName(options.description)}`
    
    // Crear directorio para el modelo
    const modelDir = path.join(this.outputBaseDir, modelName)
    if (!fs.existsSync(modelDir)) {
      fs.mkdirSync(modelDir, { recursive: true })
    }

    try {
      // 1. Adaptar especificación a optimization_core
      const config = adaptToOptimizationCore(options.spec, modelName, options.description)
      
      // 2. Generar archivo YAML de configuración
      const configPath = path.join(modelDir, 'config.yaml')
      generateOptimizationCoreYAML(config, configPath)
      
      // 3. Generar script de entrenamiento
      const scriptPath = path.join(modelDir, 'train.py')
      const trainingScript = generateTrainingScript(
        configPath,
        modelName,
        this.optimizationCorePath
      )
      fs.writeFileSync(scriptPath, trainingScript, 'utf-8')
      
      // 4. Generar README con instrucciones
      const readmePath = path.join(modelDir, 'README.md')
      const readmeContent = this.generateREADME(modelName, options.description, config)
      fs.writeFileSync(readmePath, readmeContent, 'utf-8')
      
      // 5. Crear repositorio en GitHub si está habilitado
      let githubUrl: string | undefined
      if (options.createGitHubRepo) {
        try {
          githubUrl = await createGitHubRepository(modelName, options.description)
        } catch (error) {
          console.error('Error creating GitHub repository:', error)
          // No fallar si GitHub falla
        }
      }
      
      const result: ProactiveBuildResult = {
        modelId,
        modelName,
        configPath,
        scriptPath,
        githubUrl,
        config,
        status: 'completed',
      }
      
      return result
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Error desconocido'
      
      return {
        modelId,
        modelName,
        configPath: '',
        scriptPath: '',
        config: {} as OptimizationCoreConfig,
        status: 'failed',
        error: errorMessage,
      }
    }
  }

  /**
   * Construye modelos continuamente desde una cola
   */
  async buildContinuously(
    descriptions: string[],
    onProgress?: (result: ProactiveBuildResult) => void,
    onComplete?: (results: ProactiveBuildResult[]) => void
  ): Promise<ProactiveBuildResult[]> {
    const results: ProactiveBuildResult[] = []
    
    for (const description of descriptions) {
      try {
        // Analizar descripción
        const spec = adaptiveAnalyze(description)
        const modelName = `truthgpt-${sanitizeModelName(description)}`
        
        const options: ProactiveBuildOptions = {
          modelName,
          description,
          spec,
          optimizationCorePath: this.optimizationCorePath,
          createGitHubRepo: true,
          continuous: true,
        }
        
        // Construir modelo
        const result = await this.buildModelProactively(options)
        results.push(result)
        
        // Notificar progreso
        if (onProgress) {
          onProgress(result)
        }
        
        // Pequeña pausa entre modelos
        await new Promise(resolve => setTimeout(resolve, 1000))
      } catch (error) {
        console.error(`Error building model for "${description}":`, error)
        const errorResult: ProactiveBuildResult = {
          modelId: generateModelId(),
          modelName: `error-${Date.now()}`,
          configPath: '',
          scriptPath: '',
          config: {} as OptimizationCoreConfig,
          status: 'failed',
          error: error instanceof Error ? error.message : 'Error desconocido',
        }
        results.push(errorResult)
      }
    }
    
    // Notificar completado
    if (onComplete) {
      onComplete(results)
    }
    
    return results
  }

  /**
   * Genera README para el modelo
   */
  private generateREADME(
    modelName: string,
    description: string,
    config: OptimizationCoreConfig
  ): string {
    return `# ${modelName}

## Descripción
${description}

## Arquitectura
- **Modelo Base**: ${config.model.name_or_path}
- **Atención**: ${config.model.attention.backend}
- **KV Cache**: ${config.model.kv_cache.type}
- **LoRA**: ${config.model.lora?.enabled ? 'Habilitado' : 'Deshabilitado'}
- **Precisión**: ${config.training.mixed_precision}
- **Optimizador**: ${config.optimizer.type} (${config.optimizer.fused ? 'fused' : 'standard'})

## Configuración de Entrenamiento
- **Épocas**: ${config.training.epochs}
- **Batch Size**: ${config.training.train_batch_size}
- **Learning Rate**: ${config.training.learning_rate}
- **Max Sequence Length**: ${config.data.max_seq_len}

## Uso

### Entrenar el modelo

\`\`\`bash
cd ${this.optimizationCorePath}
python train.py --config ${path.join(this.outputBaseDir, modelName, 'config.yaml')}
\`\`\`

O usando el script generado:

\`\`\`bash
python ${path.join(this.outputBaseDir, modelName, 'train.py')}
\`\`\`

## Archivos Generados
- \`config.yaml\`: Configuración de optimization_core
- \`train.py\`: Script de entrenamiento
- \`README.md\`: Este archivo

## Notas
Este modelo fue generado automáticamente por el TruthGPT Model Builder y está adaptado para trabajar con TruthGPT optimization_core.

## Integración con optimization_core
Este modelo utiliza la siguiente configuración de optimization_core:

\`\`\`yaml
# Ver config.yaml para la configuración completa
\`\`\`

### Características de optimization_core utilizadas:
- ✅ Gradient Checkpointing: ${config.model.gradient_checkpointing}
- ✅ Flash/SDPA Attention: ${config.model.attention.backend}
- ✅ KV Cache: ${config.model.kv_cache.type}
- ✅ Memory Management: ${config.model.memory.policy}
- ✅ Mixed Precision: ${config.training.mixed_precision}
- ✅ Torch Compile: ${config.training.torch_compile}
- ✅ Fused AdamW: ${config.training.fused_adamw}
- ✅ EMA: ${config.ema.enabled}
`
  }

  /**
   * Obtiene el estado de construcción actual
   */
  getBuildingStatus(): { isBuilding: boolean; queueLength: number; currentModel?: string } {
    return {
      isBuilding: this.isBuilding,
      queueLength: this.buildQueue.length,
      currentModel: this.currentBuild?.modelName,
    }
  }

  /**
   * Limpia la cola de construcción
   */
  clearQueue(): void {
    this.buildQueue = []
    this.isBuilding = false
    this.currentBuild = null
  }
}

// Instancia singleton
let builderInstance: ProactiveModelBuilder | null = null

export function getProactiveModelBuilder(
  optimizationCorePath: string,
  outputBaseDir?: string
): ProactiveModelBuilder {
  if (!builderInstance) {
    builderInstance = new ProactiveModelBuilder(optimizationCorePath, outputBaseDir)
  }
  return builderInstance
}

export default ProactiveModelBuilder

