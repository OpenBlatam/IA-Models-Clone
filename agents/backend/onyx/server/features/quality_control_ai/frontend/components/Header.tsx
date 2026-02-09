'use client';

import { memo } from 'react';
import { Camera, Shield } from 'lucide-react';
import { APP_CONFIG } from '@/config/app.config';

const Header = memo((): JSX.Element => {
  return (
    <header className="bg-white shadow-sm border-b" role="banner">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-primary-600 p-2 rounded-lg" aria-hidden="true">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{APP_CONFIG.NAME}</h1>
              <p className="text-sm text-gray-500">{APP_CONFIG.DESCRIPTION}</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <Camera className="w-4 h-4" aria-hidden="true" />
            <span>Real-time Inspection</span>
          </div>
        </div>
      </div>
    </header>
  );
});

Header.displayName = 'Header';

export default Header;
