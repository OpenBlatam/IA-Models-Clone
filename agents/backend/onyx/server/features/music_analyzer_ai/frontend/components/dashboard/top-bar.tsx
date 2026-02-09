/**
 * Dashboard Top Bar component.
 * Top navigation bar with logo, credits, filters, and controls.
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';
import {
  Search,
  Filter,
  Heart,
  Globe,
  Upload,
  Music,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { Button } from '@/components/ui';

interface TopBarProps {
  activeView: 'simple' | 'custom';
  onViewChange: (view: 'simple' | 'custom') => void;
}

/**
 * Dashboard Top Bar component.
 * Provides top navigation with logo, credits, and filters.
 *
 * @param props - Component props
 * @returns Top Bar component
 */
export function DashboardTopBar({ activeView, onViewChange }: TopBarProps) {
  const [filtersCount] = useState(3);

  return (
    <div className="h-16 bg-black border-b border-white/10 flex items-center justify-between px-6 z-10">
      {/* Left Section - Logo and Credits */}
      <div className="flex items-center gap-6">
        <Link href="/dashboard" className="flex items-center gap-2">
          <span className="text-2xl font-bold text-white tracking-tight uppercase">
            SUNO
          </span>
        </Link>

        <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-lg">
          <Music className="w-4 h-4 text-orange-400" />
          <span className="text-sm font-medium text-white">50</span>
        </div>

        {/* View Toggle */}
        <div className="flex items-center gap-1 bg-white/5 rounded-lg p-1">
          <button
            onClick={() => onViewChange('simple')}
            className={`
              px-3 py-1.5 text-sm font-medium rounded transition-colors
              ${
                activeView === 'simple'
                  ? 'bg-white/10 text-white'
                  : 'text-gray-400 hover:text-white'
              }
            `}
          >
            Simple
          </button>
          <button
            onClick={() => onViewChange('custom')}
            className={`
              px-3 py-1.5 text-sm font-medium rounded transition-colors
              ${
                activeView === 'custom'
                  ? 'bg-white/10 text-white'
                  : 'text-gray-400 hover:text-white'
              }
            `}
          >
            Custom
          </button>
        </div>

        {/* Version Dropdown */}
        <select className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-orange-500">
          <option value="v4.5-all">v4.5-all</option>
          <option value="v5-preview">v5 Preview</option>
        </select>
      </div>

      {/* Center Section - Breadcrumbs */}
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <span>Workspaces</span>
        <span>/</span>
        <span className="text-white">My Workspace</span>
      </div>

      {/* Right Section - Search, Filters, and Controls */}
      <div className="flex items-center gap-3">
        <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
          <Search className="w-5 h-5 text-gray-400" />
        </button>

        <button className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg transition-colors">
          <Filter className="w-4 h-4 text-gray-400" />
          <span className="text-sm text-gray-400">Filters ({filtersCount})</span>
        </button>

        <select className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-orange-500">
          <option value="newest">Newest</option>
          <option value="oldest">Oldest</option>
          <option value="popular">Popular</option>
        </select>

        <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
          <Heart className="w-5 h-5 text-gray-400" />
        </button>

        <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
          <Globe className="w-5 h-5 text-gray-400" />
        </button>

        <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
          <Upload className="w-5 h-5 text-gray-400" />
        </button>

        {/* Pagination */}
        <div className="flex items-center gap-1 ml-2">
          <button className="p-1.5 hover:bg-white/10 rounded transition-colors">
            <ChevronLeft className="w-4 h-4 text-gray-400" />
          </button>
          <span className="px-2 py-1 text-sm text-white">1</span>
          <button className="p-1.5 hover:bg-white/10 rounded transition-colors">
            <ChevronRight className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>
    </div>
  );
}

