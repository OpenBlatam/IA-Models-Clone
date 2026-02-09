/**
 * Dashboard Sidebar component.
 * Left navigation sidebar with user profile and menu items.
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';
import {
  Home,
  Music,
  Library,
  Search,
  Radio,
  Bell,
  Sparkles,
  TrendingUp,
  User,
  Plus,
} from 'lucide-react';
import { Button } from '@/components/ui';

interface SidebarItem {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  href?: string;
  active?: boolean;
  badge?: number;
  showCreateButton?: boolean;
}

/**
 * Dashboard Sidebar component.
 * Provides navigation menu and user information.
 *
 * @returns Sidebar component
 */
export function DashboardSidebar() {
  const [activeItem, setActiveItem] = useState('create');

  const menuItems: SidebarItem[] = [
    { id: 'home', label: 'Home', icon: Home, href: '/dashboard' },
    { id: 'create', label: 'Create', icon: Music, active: true },
    { id: 'studio', label: 'Studio', icon: Library, href: '/dashboard/studio' },
    { id: 'library', label: 'Library', icon: Library, href: '/dashboard/library' },
    { id: 'search', label: 'Search', icon: Search, href: '/dashboard/search' },
    {
      id: 'hooks',
      label: 'Hooks',
      icon: Sparkles,
      href: '/dashboard/hooks',
      showCreateButton: true,
    },
    { id: 'explore', label: 'Explore', icon: TrendingUp, href: '/dashboard/explore' },
    { id: 'radio', label: 'Radio', icon: Radio, href: '/dashboard/radio' },
    { id: 'notifications', label: 'Notifications', icon: Bell, badge: 20 },
  ];

  return (
    <div className="w-64 bg-black border-r border-white/10 flex flex-col h-full">
      {/* User Profile Section */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-500 via-orange-400 to-yellow-500 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-white truncate">folmeca3</p>
          </div>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeItem === item.id || item.active;

          const content = (
            <div
              className={`
                flex items-center gap-3 px-3 py-2 rounded transition-colors
                ${
                  isActive
                    ? 'bg-white/10 text-white'
                    : 'text-gray-400 hover:bg-white/5 hover:text-white'
                }
              `}
            >
              <Icon className="w-5 h-5" />
              <span className="flex-1 text-sm font-medium">{item.label}</span>
              {item.showCreateButton && (
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    // Handle create hook
                  }}
                  className="ml-auto px-2 py-1 text-xs bg-white/10 hover:bg-white/20 rounded transition-colors flex items-center gap-1"
                >
                  <Plus className="w-3 h-3" />
                  Create
                </button>
              )}
              {item.badge && (
                <span className="px-2 py-0.5 text-xs bg-orange-500 text-white rounded-full">
                  {item.badge}
                </span>
              )}
            </div>
          );

          if (item.href) {
            return (
              <Link
                key={item.id}
                href={item.href}
                onClick={() => setActiveItem(item.id)}
              >
                {content}
              </Link>
            );
          }

          return (
            <button
              key={item.id}
              onClick={() => setActiveItem(item.id)}
              className="w-full text-left"
            >
              {content}
            </button>
          );
        })}
      </nav>

      {/* Credits Section */}
      <div className="p-4 border-t border-white/10">
        <div className="flex items-center justify-between mb-4">
          <span className="text-sm text-gray-400">50 Credits</span>
        </div>

        {/* Go Pro Section */}
        <div className="bg-gradient-to-r from-red-900/30 to-orange-900/30 rounded-lg p-4 border border-red-500/20">
          <h3 className="text-sm font-semibold text-white mb-1">Go Pro</h3>
          <p className="text-xs text-gray-300 mb-3">
            Unlock new features, more song credits, better models, and more!
          </p>
          <Button
            variant="primary"
            size="sm"
            className="w-full bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600"
          >
            Upgrade
          </Button>
        </div>

        {/* Additional Links */}
        <div className="mt-4 space-y-2">
          <button className="w-full text-left text-sm text-gray-400 hover:text-white transition-colors">
            Earn Credits
          </button>
          <button className="w-full text-left text-sm text-gray-400 hover:text-white transition-colors flex items-center justify-between">
            <span>What's new?</span>
            <span className="px-2 py-0.5 text-xs bg-orange-500 text-white rounded-full">
              20
            </span>
          </button>
          <button className="w-full text-left text-sm text-gray-400 hover:text-white transition-colors">
            More from Suno
          </button>
        </div>
      </div>
    </div>
  );
}

