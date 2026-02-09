import React from 'react';
import { AppVersion } from './AppVersion';

type Page = 'main' | 'agent-control' | 'kanban' | 'continuous-agent';

interface LayoutProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ currentPage, onNavigate, children }) => {
  const navItems = [
    { id: 'main' as Page, label: 'Inicio', href: 'main' },
    { id: 'agent-control' as Page, label: 'Agent Control', href: 'agent-control' },
    { id: 'continuous-agent' as Page, label: 'Agentes Continuos', href: 'continuous-agent' },
    { id: 'kanban' as Page, label: 'Kanban', href: 'kanban' },
  ];

  return (
    <div className="min-h-screen bg-white text-black flex flex-col">
      {/* Header */}
      <header className="relative z-10 bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-5 md:px-6 lg:px-8 py-3.5 md:py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={() => onNavigate('main')}
              className="flex items-center gap-2.5 text-base md:text-lg text-black hover:opacity-80 transition-opacity"
            >
              <div className="w-6 h-6 md:w-7 md:h-7 flex items-center justify-center flex-shrink-0">
                <svg viewBox="0 0 24 24" className="w-full h-full">
                  <defs>
                    <linearGradient id="gradient-header" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#8800ff" />
                      <stop offset="16.66%" stopColor="#0000ff" />
                      <stop offset="33.33%" stopColor="#0088ff" />
                      <stop offset="50%" stopColor="#00ff00" />
                      <stop offset="66.66%" stopColor="#ffdd00" />
                      <stop offset="83.33%" stopColor="#ff8800" />
                      <stop offset="100%" stopColor="#ff0000" />
                    </linearGradient>
                  </defs>
                  <path d="M7 20L12 4L17 20H14.5L12 12.5L9.5 20H7Z" fill="url(#gradient-header)"/>
                </svg>
              </div>
              <span className="font-normal">
                <span className="font-light">GitHub</span> <span className="font-normal">Autonomous Agent AI</span>
              </span>
            </button>
            
            <nav className="flex items-center gap-4">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`text-black transition-opacity font-normal text-sm ${
                    currentPage === item.id
                      ? 'font-semibold border-b-2 border-black'
                      : 'hover:opacity-70'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </nav>

            {/* App Version */}
            <div className="hidden md:flex items-center">
              <AppVersion />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
};

