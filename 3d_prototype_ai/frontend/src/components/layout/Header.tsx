'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useAuthStore } from '@/store/auth.store'
import { Sparkles, User, LogOut, Menu, X } from 'lucide-react'
import { useState } from 'react'

export function Header() {
  const pathname = usePathname()
  const { user, isAuthenticated, logout } = useAuthStore()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const navLinks = [
    { href: '/dashboard', label: 'Dashboard' },
    { href: '/prototypes', label: 'Prototipos' },
    { href: '/prototypes/create', label: 'Crear' },
    { href: '/marketplace', label: 'Marketplace' },
    { href: '/analytics', label: 'Analytics' },
  ]

  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <Sparkles className="w-8 h-8 text-primary-600" />
            <span className="text-xl font-bold text-gray-900">3D Prototype AI</span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-6">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  pathname === link.href
                    ? 'bg-primary-100 text-primary-700'
                    : 'text-gray-700 hover:text-primary-600 hover:bg-gray-50'
                }`}
              >
                {link.label}
              </Link>
            ))}
          </nav>

          {/* User Menu */}
          <div className="flex items-center gap-4">
            {isAuthenticated ? (
              <>
                <div className="hidden md:flex items-center gap-2 text-sm text-gray-700">
                  <User className="w-4 h-4" />
                  <span>{user?.username || user?.email}</span>
                </div>
                <button
                  onClick={logout}
                  className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:text-red-600 transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                  <span className="hidden md:inline">Salir</span>
                </button>
              </>
            ) : (
              <Link
                href="/auth/login"
                className="px-4 py-2 bg-primary-600 text-white rounded-md text-sm font-medium hover:bg-primary-700 transition-colors"
              >
                Iniciar Sesión
              </Link>
            )}

            {/* Mobile Menu Button */}
            <button
              className="md:hidden p-2"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <nav className="md:hidden py-4 border-t border-gray-200">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`block px-4 py-2 rounded-md text-sm font-medium ${
                  pathname === link.href
                    ? 'bg-primary-100 text-primary-700'
                    : 'text-gray-700 hover:bg-gray-50'
                }`}
                onClick={() => setMobileMenuOpen(false)}
              >
                {link.label}
              </Link>
            ))}
          </nav>
        )}
      </div>
    </header>
  )
}



