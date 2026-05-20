'use client'

import ChatInterface from '@/components/ChatInterface'
import Header from '@/components/Header'
import ErrorBoundary from '@/components/ErrorBoundary'

export default function Home() {
  return (
    <ErrorBoundary>
      <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">                                                              
        <Header />
        <div className="container mx-auto px-4 py-8">
          <ChatInterface devMode={false} />
        </div>
      </main>
    </ErrorBoundary>
  )
}

