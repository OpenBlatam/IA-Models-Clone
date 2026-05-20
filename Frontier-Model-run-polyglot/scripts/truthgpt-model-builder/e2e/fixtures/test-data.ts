/**
 * E2E Test Data Fixtures
 */

export const testModels = [
  {
    description: 'classification model for text categorization',
    expectedKeywords: ['classification', 'categorization'],
  },
  {
    description: 'regression model for price prediction',
    expectedKeywords: ['regression', 'prediction'],
  },
  {
    description: 'sentiment analysis model for reviews',
    expectedKeywords: ['sentiment', 'analysis'],
  },
  {
    description: 'neural network with 3 hidden layers and dropout',
    expectedKeywords: ['neural', 'network', 'layers'],
  },
]

export const testTemplates = [
  'classification',
  'regression',
  'nlp',
  'computer-vision',
]

export const testSearchQueries = [
  'classification',
  'regression',
  'sentiment',
  'neural',
]

export const waitForModelCreation = async (page: any, timeout: number = 30000) => {
  await page.waitForSelector(
    'text=/completado|success|creado|error|failed/i',
    { timeout }
  )
}

export const waitForPanel = async (page: any, panelText: string, timeout: number = 3000) => {
  await page.waitForSelector(`text=/${panelText}/i`, { timeout })
}

export const clickButton = async (page: any, buttonText: string) => {
  const button = page.locator(`button:has-text("${buttonText}")`).first()
  if (await button.isVisible()) {
    await button.click()
    return true
  }
  return false
}










