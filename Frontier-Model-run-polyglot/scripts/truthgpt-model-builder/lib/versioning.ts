/**
 * Model versioning system
 */

export interface ModelVersion {
  version: string
  modelId: string
  parentVersion?: string
  changes: string[]
  createdAt: Date
  spec: any
  code: string
  githubUrl?: string
}

export interface VersionHistory {
  modelId: string
  currentVersion: string
  versions: ModelVersion[]
}

const VERSION_HISTORY_KEY = 'truthgpt-model-versions'

export function saveVersion(version: ModelVersion): void {
  if (typeof window === 'undefined') return

  try {
    const history = getVersionHistory(version.modelId)
    history.versions.push(version)
    history.currentVersion = version.version

    const allHistory = getAllVersionHistory()
    allHistory[version.modelId] = history
    localStorage.setItem(VERSION_HISTORY_KEY, JSON.stringify(allHistory))
  } catch (error) {
    console.error('Error saving version:', error)
  }
}

export function getVersionHistory(modelId: string): VersionHistory {
  if (typeof window === 'undefined') {
    return { modelId, currentVersion: '1.0.0', versions: [] }
  }

  try {
    const allHistoryStr = localStorage.getItem(VERSION_HISTORY_KEY)
    if (!allHistoryStr) {
      return { modelId, currentVersion: '1.0.0', versions: [] }
    }

    const allHistory = JSON.parse(allHistoryStr)
    const history = allHistory[modelId]

    if (!history) {
      return { modelId, currentVersion: '1.0.0', versions: [] }
    }

    return {
      ...history,
      versions: history.versions.map((v: any) => ({
        ...v,
        createdAt: new Date(v.createdAt),
      })),
    }
  } catch (error) {
    console.error('Error getting version history:', error)
    return { modelId, currentVersion: '1.0.0', versions: [] }
  }
}

export function getAllVersionHistory(): Record<string, VersionHistory> {
  if (typeof window === 'undefined') return {}

  try {
    const historyStr = localStorage.getItem(VERSION_HISTORY_KEY)
    if (!historyStr) return {}

    const history = JSON.parse(historyStr)
    return Object.keys(history).reduce((acc, key) => {
      acc[key] = {
        ...history[key],
        versions: history[key].versions.map((v: any) => ({
          ...v,
          createdAt: new Date(v.createdAt),
        })),
      }
      return acc
    }, {} as Record<string, VersionHistory>)
  } catch (error) {
    console.error('Error getting all version history:', error)
    return {}
  }
}

export function generateNextVersion(
  modelId: string,
  changeType: 'major' | 'minor' | 'patch' = 'minor'
): string {
  const history = getVersionHistory(modelId)
  const currentVersion = history.currentVersion || '0.0.0'
  const [major, minor, patch] = currentVersion.split('.').map(Number)

  switch (changeType) {
    case 'major':
      return `${major + 1}.0.0`
    case 'minor':
      return `${major}.${minor + 1}.0`
    case 'patch':
      return `${major}.${minor}.${patch + 1}`
    default:
      return `${major}.${minor + 1}.0`
  }
}

export function cloneModelAsVersion(
  modelId: string,
  changes: string[],
  spec: any,
  code: string,
  changeType: 'major' | 'minor' | 'patch' = 'minor'
): ModelVersion {
  const history = getVersionHistory(modelId)
  const parentVersion = history.currentVersion
  const newVersion = generateNextVersion(modelId, changeType)

  const version: ModelVersion = {
    version: newVersion,
    modelId: `${modelId}-v${newVersion}`,
    parentVersion,
    changes,
    createdAt: new Date(),
    spec,
    code,
  }

  saveVersion(version)
  return version
}


