'use client'

import React from 'react'
import PageLayout from '@/components/layout/PageLayout'
import CodeImprover from '@/components/features/CodeImprover'

const CodeImprovePage: React.FC = () => {
  return (
    <PageLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Improve Code</h1>
          <p className="mt-2 text-gray-600">
            Enhance your code using knowledge from research papers
          </p>
        </div>

        <CodeImprover />
      </div>
    </PageLayout>
  )
}

export default CodeImprovePage




