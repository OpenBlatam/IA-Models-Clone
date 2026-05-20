/**
 * Cost estimation for model training and deployment
 */

export interface CostEstimate {
  training: {
    compute: number
    storage: number
    total: number
  }
  deployment: {
    hosting: number
    api: number
    total: number
  }
  total: number
  currency: string
  breakdown: CostBreakdown[]
}

export interface CostBreakdown {
  category: string
  item: string
  quantity: number
  unitCost: number
  total: number
}

interface ModelSpec {
  architecture: string
  layers: number[]
  epochs: number
  batchSize: number
  optimizer: string
}

const HOURLY_COMPUTE_COST = 0.5 // $0.50 per hour
const STORAGE_COST_PER_GB = 0.1 // $0.10 per GB per month
const HOSTING_COST_PER_MONTH = 10 // $10 per month
const API_COST_PER_1000 = 0.01 // $0.01 per 1000 API calls

export function estimateCost(spec: ModelSpec): CostEstimate {
  const trainingHours = estimateTrainingTime(spec)
  const modelSizeGB = estimateModelSize(spec)
  const apiCallsPerMonth = 10000 // Default estimate

  const computeCost = trainingHours * HOURLY_COMPUTE_COST
  const storageCost = modelSizeGB * STORAGE_COST_PER_GB
  const hostingCost = HOSTING_COST_PER_MONTH
  const apiCost = (apiCallsPerMonth / 1000) * API_COST_PER_1000

  const breakdown: CostBreakdown[] = [
    {
      category: 'Training',
      item: 'Compute (GPU/CPU)',
      quantity: trainingHours,
      unitCost: HOURLY_COMPUTE_COST,
      total: computeCost,
    },
    {
      category: 'Storage',
      item: 'Model Storage',
      quantity: modelSizeGB,
      unitCost: STORAGE_COST_PER_GB,
      total: storageCost,
    },
    {
      category: 'Deployment',
      item: 'Hosting',
      quantity: 1,
      unitCost: HOSTING_COST_PER_MONTH,
      total: hostingCost,
    },
    {
      category: 'Deployment',
      item: 'API Calls',
      quantity: apiCallsPerMonth,
      unitCost: API_COST_PER_1000 / 1000,
      total: apiCost,
    },
  ]

  return {
    training: {
      compute: computeCost,
      storage: storageCost,
      total: computeCost + storageCost,
    },
    deployment: {
      hosting: hostingCost,
      api: apiCost,
      total: hostingCost + apiCost,
    },
    total: computeCost + storageCost + hostingCost + apiCost,
    currency: 'USD',
    breakdown,
  }
}

function estimateTrainingTime(spec: ModelSpec): number {
  // Base time: 1 hour
  let baseTime = 1

  // Architecture multiplier
  const archMultiplier: Record<string, number> = {
    dense: 1,
    cnn: 1.5,
    lstm: 2,
    transformer: 3,
  }

  const multiplier = archMultiplier[spec.architecture] || 1

  // Epochs factor
  const epochsFactor = spec.epochs / 10 // Normalize to 10 epochs

  // Batch size factor (smaller = longer)
  const batchFactor = 32 / spec.batchSize

  return baseTime * multiplier * epochsFactor * batchFactor
}

function estimateModelSize(spec: ModelSpec): number {
  // Estimate model size in GB
  let totalParams = 0

  // Estimate parameters based on layers
  for (let i = 0; i < spec.layers.length - 1; i++) {
    const inputSize = spec.layers[i]
    const outputSize = spec.layers[i + 1]
    totalParams += inputSize * outputSize + outputSize // weights + bias
  }

  // Each parameter is 4 bytes (float32)
  const sizeBytes = totalParams * 4
  const sizeGB = sizeBytes / (1024 * 1024 * 1024)

  // Add overhead (20%)
  return sizeGB * 1.2
}

export function formatCost(cost: number, currency: string = 'USD'): string {
  const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
  return formatter.format(cost)
}


