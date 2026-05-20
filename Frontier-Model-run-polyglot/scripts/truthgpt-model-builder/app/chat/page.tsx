'use client'

import { useSearchParams } from 'next/navigation'
import { Suspense } from 'react'
import ChatInterface from '@/components/ChatInterface'
import Header from '@/components/Header'
import ErrorBoundary from '@/components/ErrorBoundary'

function ChatContent() {
  const searchParams = useSearchParams()
  const devMode = searchParams.get('dev_mode') === 'true'

  return <ChatInterface devMode={devMode} />
}

export default function ChatPage() {
  return (
    <ErrorBoundary>
      <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <Header />
        <div className="container mx-auto px-4 py-8">
          <Suspense fallback={<div className="text-white">Cargando...</div>}>
            <ChatContent />
          </Suspense>
        </div>
      </main>
    </ErrorBoundary>
  )
}

