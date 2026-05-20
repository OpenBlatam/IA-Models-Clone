/**
 * Model Creation Service - Handles the model creation workflow
 */

import * as path from 'path'
import * as fs from 'fs/promises'
import { logger, retry } from '../modules/utilities'
import { verifyTruthGPTPath, ensureGeneratedModelsDir } from '../modules/utilities'
import { analyzeModelDescription } from '../modules/analysis'
import { validateModelSpec } from '../modules/validation'
import { optimizeModelSpec } from '../modules/optimization'
import { adaptToTruthGPT, generateTruthGPTCode } from '../modules/adaptation'
import { integrateWithTruthGPT } from '../modules/adaptation'
import { Cache } from '../modules/storage'
import { CONFIG } from '../core/config'
import { generateReadme } from '../utils/readme-generator'
import { toPascalCase } from '../utils/code-generators'
import type { ModelSpec } from '../core/types'

const specCache = new Cache<ModelSpec>()

export interface CreationResult {
  modelId: string
  modelName: string
  modelDir: string
  spec: ModelSpec
}

/**
 * Analyzes and prepares model specification
 */
export async function analyzeAndPrepareSpec(
  description: string
): Promise<ModelSpec> {
  // Check cache
  let spec = specCache.get(description)
  if (spec) {
    logger.debug('Using cached spec', { description })
    return spec
  }

  // Analyze with enhanced analyzer
  logger.debug('Analyzing model description', { description })
  spec = analyzeModelDescription(description)

  // Validate
  const validation = validateModelSpec(spec)
  if (!validation.isValid) {
    logger.warn('Model spec validation failed', { errors: validation.errors })
  }

  // Optimize
  const optimization = optimizeModelSpec(spec)
  logger.info('Model spec optimized', {
    improvements: optimization.improvements.length,
  })
  spec = optimization.optimized

  // Cache
  specCache.set(description, spec, CONFIG.SPEC_CACHE_TTL)

  return spec
}

/**
 * Prepares model directory structure
 */
export async function prepareModelDirectory(
  modelName: string
): Promise<string> {
  const truthgptPath = path.resolve(process.cwd(), CONFIG.TRUTHGPT_PATH)

  // Verify path with retry
  const pathExists = await retry(
    () => verifyTruthGPTPath(truthgptPath),
    {
      maxAttempts: CONFIG.RETRY_MAX_ATTEMPTS,
      initialDelay: CONFIG.RETRY_INITIAL_DELAY,
      retryable: (error) => {
        logger.warn('Failed to verify TruthGPT path, retrying...', {
          error: error.message,
        })
        return true
      },
    }
  )

  if (!pathExists) {
    throw new Error(`TruthGPT path not found: ${truthgptPath}`)
  }

  // Ensure directory exists
  await retry(
    () => ensureGeneratedModelsDir(truthgptPath),
    {
      maxAttempts: CONFIG.RETRY_MAX_ATTEMPTS,
      retryable: (error) => {
        logger.warn('Failed to ensure generated_models directory, retrying...', {
          error: error.message,
        })
        return true
      },
    }
  )

  // Create model directory
  const modelDir = path.join(
    truthgptPath,
    CONFIG.GENERATED_MODELS_DIR,
    modelName
  )
  await fs.mkdir(modelDir, { recursive: true })
  logger.debug('Created model directory', { modelDir })

  return modelDir
}

/**
 * Generates model files
 */
export async function generateModelFiles(
  modelDir: string,
  modelName: string,
  description: string,
  spec: ModelSpec
): Promise<void> {
  // Adapt to TruthGPT
  const truthgptSpec = adaptToTruthGPT(spec)
  if (!truthgptSpec.compatible) {
    logger.warn('Spec not fully compatible with TruthGPT', { spec })
  }

  // Generate code
  const modelCode = generateTruthGPTCode(truthgptSpec, modelName, description)

  // Write model file with retry
  await retry(
    async () => {
      await fs.writeFile(
        path.join(modelDir, `${modelName}.py`),
        modelCode,
        'utf-8'
      )
    },
    {
      maxAttempts: CONFIG.RETRY_MAX_ATTEMPTS,
      retryable: (error) => {
        logger.warn('Failed to write model file, retrying...', {
          error: error.message,
        })
        return true
      },
    }
  )

  logger.info('Model file written successfully', {
    fileName: `${modelName}.py`,
  })
}

/**
 * Generates supporting files (README, requirements, etc.)
 */
export async function generateSupportingFiles(
  modelDir: string,
  modelName: string,
  description: string,
  spec: ModelSpec
): Promise<void> {
  // Generate README
  const readme = generateReadme(modelName, description, spec)
  await fs.writeFile(path.join(modelDir, 'README.md'), readme)

  // Generate requirements.txt
  const requirements = `torch>=2.0.0
numpy>=1.24.0
truthgpt-api
`
  await fs.writeFile(
    path.join(modelDir, 'requirements.txt'),
    requirements
  )

  // Generate __init__.py
  const className = toPascalCase(modelName)
  await fs.writeFile(
    path.join(modelDir, '__init__.py'),
    `from .${modelName} import ${className}

__all__ = ['${className}']
`
  )
}

/**
 * Integrates with TruthGPT core
 */
export async function performTruthGPTIntegration(
  modelDir: string,
  modelName: string
): Promise<void> {
  try {
    logger.debug('Integrating with TruthGPT core', { modelName })
    await retry(
      () =>
        integrateWithTruthGPT(modelDir, modelName, {
          optimizationLevel: 'enhanced',
          useTruthGPTCore: true,
          applyOptimizations: true,
        }),
      {
        maxAttempts: 2,
        retryable: (error) => {
          logger.warn('TruthGPT integration failed, retrying...', {
            error: error.message,
          })
          return true
        },
      }
    )
    logger.info('TruthGPT integration completed', { modelName })
  } catch (error) {
    logger.error(
      'Error integrating with TruthGPT',
      error instanceof Error ? error : new Error(String(error)),
      { modelName }
    )
    // Continue without integration
  }
}

// Helper functions
function generateReadme(
  modelName: string,
  description: string,
  spec: any
): string {
  const className = toPascalCase(modelName)

  return `# ${className}

![TruthGPT](https://img.shields.io/badge/TruthGPT-Powered-purple?style=for-the-badge&logo=github)

TruthGPT Model Generated Automatically

## Description

${description}

## Model Specifications

- **Type**: ${spec.type}
- **Architecture**: ${spec.architecture}
- **Layers**: ${spec.layers.join(', ')}
- **Activation**: ${spec.activation}
- **Optimizer**: ${spec.optimizer}
- **Learning Rate**: ${spec.learningRate}
- **Loss Function**: ${spec.loss}
- **Metrics**: ${spec.metrics.join(', ')}

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`python
from ${modelName} import ${className}
import truthgpt as tg
import numpy as np

# Create model instance
model = ${className}()

# Compile model
model.compile(
    optimizer=tg.optimizers.${spec.optimizer.charAt(0).toUpperCase() + spec.optimizer.slice(1)}(learning_rate=${spec.learningRate}),
    loss=tg.losses.${toPascalCaseLoss(spec.loss)}(),
    metrics=['${spec.metrics.join("', '")}']
)

# Train model
history = model.fit(x_train, y_train, epochs=${spec.epochs}, batch_size=${spec.batchSize})

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Make predictions
predictions = model.predict(x_test)
\`\`\`

## Generated by

TruthGPT Model Builder
`
}

function toPascalCase(str: string): string {
  return str
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join('')
}

function toPascalCaseLoss(loss: string): string {
  return loss
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join('')
}

