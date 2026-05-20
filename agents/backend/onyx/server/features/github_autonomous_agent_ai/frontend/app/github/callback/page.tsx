'use client';

import { Suspense } from 'react';
import GithubCallback from '../../components/GithubCallback';

export default function CallbackPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    }>
      <GithubCallback />
    </Suspense>
  );
}



