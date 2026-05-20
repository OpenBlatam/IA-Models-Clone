/**
 * Optimization Core Adapter
 * Adapta modelos para trabajar con TruthGPT optimization_core
 * Genera configuraciones YAML compatibles con optimization_core
 */

import { ModelSpec } from './modules/management'
import fs from 'fs'
import path from 'path'

export interface OptimizationCoreConfig {
  seed: number
  run_name: string
  output_dir: string
  model: {
    name_or_path: string
    gradient_checkpointing: boolean
    lora?: {
      enabled: boolean
      r: number
      alpha: number
      dropout: number
    }
    attention: {
      backend: 'sdpa' | 'flash' | 'triton'
    }
    kv_cache: {
      type: 'none' | 'paged'
      block_size: number
    }
    memory: {
      policy: 'adaptive' | 'static'
    }
  }
  training: {
    epochs: number
    train_batch_size: number
    eval_batch_size: number
    grad_accum_steps: number
    max_grad_norm: number
    learning_rate: number
    weight_decay: number
    warmup_ratio: number
    scheduler: string
    mixed_precision: 'bf16' | 'fp16' | 'fp32'
    early_stopping_patience: number
    log_interval: number
    eval_interval: number
    allow_tf32: boolean
    torch_compile: boolean
    compile_mode: string
    fused_adamw: boolean
    detect_anomaly: boolean
    use_profiler: boolean
    save_safetensors: boolean
    callbacks: string[]
  }
  logging: {
    project: string
    run_name: string
    dir: string
  }
  eval: {
    metrics: string[]
    select_best_by: string
  }
  checkpoint: {
    interval_steps: number
    keep_last: number
  }
  ema: {
    enabled: boolean
    decay: number
  }
  resume: {
    enabled: boolean
    checkpoint_dir: string | null
  }
  optimizer: {
    type: 'adamw' | 'lion' | 'adafactor'
    fused: boolean
  }
  data: {
    source: 'hf' | 'jsonl' | 'webdataset'
    dataset: string
    subset: string | null
    path: string | null
    text_field: string
    streaming: boolean
    collate: 'lm' | 'cv' | 'audio'
    max_seq_len: number
    bucket_by_length: boolean
    bucket_bins: number[]
    num_workers: number
    prefetch_factor: number
    persistent_workers: boolean
  }
  hardware: {
    device: string
    ddp: boolean
  }
}

/**
 * Adapta un ModelSpec a una configuración de optimization_core
 */
export function adaptToOptimizationCore(
  spec: ModelSpec,
  modelName: string,
  description: string
): OptimizationCoreConfig {
  // Determinar configuración basada en el tipo de modelo
  const isLargeModel = spec.layers.reduce((sum, l) => sum + l, 0) > 1000
  const isNLP = spec.type === 'nlp' || spec.type === 'text' || spec.type === 'language'
  const isVision = spec.type === 'vision' || spec.type === 'image' || spec.type === 'cv'
  const isAudio = spec.type === 'audio' || spec.type === 'speech'
  
  // Determinar modelo base
  let baseModel = 'gpt2'
  if (isVision) {
    baseModel = 'gpt2' // TruthGPT optimization_core principalmente para LLM
  } else if (isNLP) {
    baseModel = 'gpt2' // Por defecto GPT-2, pero se puede cambiar
  }

  // Configuración de atención
  let attentionBackend: 'sdpa' | 'flash' | 'triton' = 'sdpa'
  if (isLargeModel) {
    attentionBackend = 'flash' // Flash attention para modelos grandes
  }

  // Configuración de KV cache
  let kvCacheType: 'none' | 'paged' = 'none'
  if (isLargeModel || spec.maxSequenceLength > 512) {
    kvCacheType = 'paged'
  }

  // Determinar batch size
  const trainBatchSize = isLargeModel ? 4 : 8
  const evalBatchSize = isLargeModel ? 4 : 8

  // Determinar precision
  let mixedPrecision: 'bf16' | 'fp16' | 'fp32' = 'bf16'
  if (!isLargeModel) {
    mixedPrecision = 'fp16'
  }

  // Determinar collate function
  let collate: 'lm' | 'cv' | 'audio' = 'lm'
  if (isVision) {
    collate = 'cv'
  } else if (isAudio) {
    collate = 'audio'
  }

  // Configuración de LoRA
  const useLoRA = isLargeModel || spec.useDropout

  const config: OptimizationCoreConfig = {
    seed: 42,
    run_name: modelName,
    output_dir: `runs/${modelName}`,
    model: {
      name_or_path: baseModel,
      gradient_checkpointing: isLargeModel,
      ...(useLoRA && {
        lora: {
          enabled: true,
          r: 16,
          alpha: 32,
          dropout: spec.dropoutRate || 0.05,
        },
      }),
      attention: {
        backend: attentionBackend,
      },
      kv_cache: {
        type: kvCacheType,
        block_size: 128,
      },
      memory: {
        policy: 'adaptive',
      },
    },
    training: {
      epochs: spec.epochs || 3,
      train_batch_size: trainBatchSize,
      eval_batch_size: evalBatchSize,
      grad_accum_steps: isLargeModel ? 4 : 2,
      max_grad_norm: 1.0,
      learning_rate: spec.learningRate || 5.0e-5,
      weight_decay: 0.01,
      warmup_ratio: 0.06,
      scheduler: 'cosine',
      mixed_precision: mixedPrecision,
      early_stopping_patience: 2,
      log_interval: 50,
      eval_interval: 500,
      allow_tf32: true,
      torch_compile: isLargeModel,
      compile_mode: 'default',
      fused_adamw: true,
      detect_anomaly: false,
      use_profiler: false,
      save_safetensors: true,
      callbacks: ['print'],
    },
    logging: {
      project: 'truthgpt',
      run_name: modelName,
      dir: 'runs',
    },
    eval: {
      metrics: ['ppl'],
      select_best_by: 'ppl',
    },
    checkpoint: {
      interval_steps: 1000,
      keep_last: 3,
    },
    ema: {
      enabled: true,
      decay: 0.999,
    },
    resume: {
      enabled: false,
      checkpoint_dir: null,
    },
    optimizer: {
      type: 'adamw',
      fused: true,
    },
    data: {
      source: 'hf',
      dataset: 'wikitext',
      subset: 'wikitext-2-raw-v1',
      path: null,
      text_field: 'text',
      streaming: false,
      collate: collate,
      max_seq_len: spec.maxSequenceLength || 512,
      bucket_by_length: false,
      bucket_bins: [64, 128, 256, 512],
      num_workers: 4,
      prefetch_factor: 2,
      persistent_workers: true,
    },
    hardware: {
      device: 'auto',
      ddp: false,
    },
  }

  return config
}

/**
 * Genera un archivo YAML de configuración para optimization_core
 */
export function generateOptimizationCoreYAML(
  config: OptimizationCoreConfig,
  outputPath: string
): string {
  const yamlContent = generateYAMLString(config)
  
  // Asegurar que el directorio existe
  const dir = path.dirname(outputPath)
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true })
  }

  // Escribir archivo
  fs.writeFileSync(outputPath, yamlContent, 'utf-8')
  
  return outputPath
}

/**
 * Genera string YAML a partir de la configuración
 */
function generateYAMLString(config: OptimizationCoreConfig): string {
  const indent = (level: number) => '  '.repeat(level)
  
  let yaml = ''
  
  // seed
  yaml += `seed: ${config.seed}\n`
  yaml += `run_name: ${config.run_name}\n`
  yaml += `output_dir: ${config.output_dir}\n\n`
  
  // model
  yaml += `model:\n`
  yaml += `${indent(1)}name_or_path: ${config.model.name_or_path}\n`
  yaml += `${indent(1)}gradient_checkpointing: ${config.model.gradient_checkpointing}\n`
  
  if (config.model.lora) {
    yaml += `${indent(1)}lora:\n`
    yaml += `${indent(2)}enabled: ${config.model.lora.enabled}\n`
    yaml += `${indent(2)}r: ${config.model.lora.r}\n`
    yaml += `${indent(2)}alpha: ${config.model.lora.alpha}\n`
    yaml += `${indent(2)}dropout: ${config.model.lora.dropout}\n`
  }
  
  yaml += `${indent(1)}attention:\n`
  yaml += `${indent(2)}backend: ${config.model.attention.backend}\n`
  
  yaml += `${indent(1)}kv_cache:\n`
  yaml += `${indent(2)}type: ${config.model.kv_cache.type}\n`
  yaml += `${indent(2)}block_size: ${config.model.kv_cache.block_size}\n`
  
  yaml += `${indent(1)}memory:\n`
  yaml += `${indent(2)}policy: ${config.model.memory.policy}\n\n`
  
  // training
  yaml += `training:\n`
  yaml += `${indent(1)}epochs: ${config.training.epochs}\n`
  yaml += `${indent(1)}train_batch_size: ${config.training.train_batch_size}\n`
  yaml += `${indent(1)}eval_batch_size: ${config.training.eval_batch_size}\n`
  yaml += `${indent(1)}grad_accum_steps: ${config.training.grad_accum_steps}\n`
  yaml += `${indent(1)}max_grad_norm: ${config.training.max_grad_norm}\n`
  yaml += `${indent(1)}learning_rate: ${config.training.learning_rate}\n`
  yaml += `${indent(1)}weight_decay: ${config.training.weight_decay}\n`
  yaml += `${indent(1)}warmup_ratio: ${config.training.warmup_ratio}\n`
  yaml += `${indent(1)}scheduler: ${config.training.scheduler}\n`
  yaml += `${indent(1)}mixed_precision: ${config.training.mixed_precision}\n`
  yaml += `${indent(1)}early_stopping_patience: ${config.training.early_stopping_patience}\n`
  yaml += `${indent(1)}log_interval: ${config.training.log_interval}\n`
  yaml += `${indent(1)}eval_interval: ${config.training.eval_interval}\n`
  yaml += `${indent(1)}allow_tf32: ${config.training.allow_tf32}\n`
  yaml += `${indent(1)}torch_compile: ${config.training.torch_compile}\n`
  yaml += `${indent(1)}compile_mode: ${config.training.compile_mode}\n`
  yaml += `${indent(1)}fused_adamw: ${config.training.fused_adamw}\n`
  yaml += `${indent(1)}detect_anomaly: ${config.training.detect_anomaly}\n`
  yaml += `${indent(1)}use_profiler: ${config.training.use_profiler}\n`
  yaml += `${indent(1)}save_safetensors: ${config.training.save_safetensors}\n`
  yaml += `${indent(1)}callbacks:\n`
  config.training.callbacks.forEach(cb => {
    yaml += `${indent(2)}- ${cb}\n`
  })
  yaml += `\n`
  
  // logging
  yaml += `logging:\n`
  yaml += `${indent(1)}project: ${config.logging.project}\n`
  yaml += `${indent(1)}run_name: ${config.logging.run_name}\n`
  yaml += `${indent(1)}dir: ${config.logging.dir}\n\n`
  
  // eval
  yaml += `eval:\n`
  yaml += `${indent(1)}metrics: [${config.eval.metrics.join(', ')}]\n`
  yaml += `${indent(1)}select_best_by: ${config.eval.select_best_by}\n\n`
  
  // checkpoint
  yaml += `checkpoint:\n`
  yaml += `${indent(1)}interval_steps: ${config.checkpoint.interval_steps}\n`
  yaml += `${indent(1)}keep_last: ${config.checkpoint.keep_last}\n\n`
  
  // ema
  yaml += `ema:\n`
  yaml += `${indent(1)}enabled: ${config.ema.enabled}\n`
  yaml += `${indent(1)}decay: ${config.ema.decay}\n\n`
  
  // resume
  yaml += `resume:\n`
  yaml += `${indent(1)}enabled: ${config.resume.enabled}\n`
  yaml += `${indent(1)}checkpoint_dir: ${config.resume.checkpoint_dir}\n\n`
  
  // optimizer
  yaml += `optimizer:\n`
  yaml += `${indent(1)}type: ${config.optimizer.type}\n`
  yaml += `${indent(1)}fused: ${config.optimizer.fused}\n\n`
  
  // data
  yaml += `data:\n`
  yaml += `${indent(1)}source: ${config.data.source}\n`
  yaml += `${indent(1)}dataset: ${config.data.dataset}\n`
  if (config.data.subset) {
    yaml += `${indent(1)}subset: ${config.data.subset}\n`
  }
  yaml += `${indent(1)}path: ${config.data.path}\n`
  yaml += `${indent(1)}text_field: ${config.data.text_field}\n`
  yaml += `${indent(1)}streaming: ${config.data.streaming}\n`
  yaml += `${indent(1)}collate: ${config.data.collate}\n`
  yaml += `${indent(1)}max_seq_len: ${config.data.max_seq_len}\n`
  yaml += `${indent(1)}bucket_by_length: ${config.data.bucket_by_length}\n`
  yaml += `${indent(1)}bucket_bins: [${config.data.bucket_bins.join(', ')}]\n`
  yaml += `${indent(1)}num_workers: ${config.data.num_workers}\n`
  yaml += `${indent(1)}prefetch_factor: ${config.data.prefetch_factor}\n`
  yaml += `${indent(1)}persistent_workers: ${config.data.persistent_workers}\n\n`
  
  // hardware
  yaml += `hardware:\n`
  yaml += `${indent(1)}device: ${config.hardware.device}\n`
  yaml += `${indent(1)}ddp: ${config.hardware.ddp}\n`
  
  return yaml
}

/**
 * Genera un script Python para entrenar el modelo con optimization_core
 */
export function generateTrainingScript(
  configPath: string,
  modelName: string,
  optimizationCorePath: string
): string {
  const script = `"""
Script de entrenamiento generado para ${modelName}
Usa TruthGPT optimization_core para entrenar el modelo
"""

import sys
import os

# Agregar optimization_core al path
sys.path.insert(0, '${optimizationCorePath}')

from train_llm import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="${configPath}")
    parser.add_argument("--max_seq_len", type=int, default=512)
    
    args = parser.parse_args()
    
    # Modificar sys.argv para que train_llm.py pueda leer los argumentos
    sys.argv = ["train_llm.py", "--config", args.config]
    
    if args.max_seq_len:
        sys.argv.extend(["--max_seq_len", str(args.max_seq_len)])
    
    main()
`
  
  return script
}

