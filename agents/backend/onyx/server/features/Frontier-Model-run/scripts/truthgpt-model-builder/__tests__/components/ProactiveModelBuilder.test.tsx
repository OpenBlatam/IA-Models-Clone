/**
 * Component Tests - ProactiveModelBuilder
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'
import ProactiveModelBuilder from '@/components/ProactiveModelBuilder'

// Mock dependencies
jest.mock('@/lib/realtime-metrics')
jest.mock('@/lib/smart-cache')
jest.mock('@/lib/intelligent-alerts')
jest.mock('@/lib/smart-history')
jest.mock('@/lib/favorites-manager')
jest.mock('react-hot-toast', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    loading: jest.fn(),
  },
}))

describe('ProactiveModelBuilder', () => {
  const mockOnModelCreated = jest.fn()

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should render component', () => {
    render(<ProactiveModelBuilder onModelCreated={mockOnModelCreated} />)
    
    expect(screen.getByText(/constructor proactivo/i)).toBeInTheDocument()
  })

  it('should toggle proactive mode', async () => {
    render(<ProactiveModelBuilder onModelCreated={mockOnModelCreated} />)
    
    const toggleButton = screen.getByRole('button', { name: /proactivo/i })
    fireEvent.click(toggleButton)

    await waitFor(() => {
      expect(screen.getByText(/construyendo modelos/i)).toBeInTheDocument()
    })
  })

  it('should add model to queue', async () => {
    render(<ProactiveModelBuilder onModelCreated={mockOnModelCreated} />)
    
    const input = screen.getByPlaceholderText(/descripción del modelo/i)
    const addButton = screen.getByRole('button', { name: /agregar/i })

    fireEvent.change(input, { target: { value: 'classification model' } })
    fireEvent.click(addButton)

    await waitFor(() => {
      expect(screen.getByText(/classification model/i)).toBeInTheDocument()
    })
  })

  it('should display statistics', () => {
    render(<ProactiveModelBuilder onModelCreated={mockOnModelCreated} />)
    
    const statsButton = screen.getByRole('button', { name: /estadísticas/i })
    fireEvent.click(statsButton)

    expect(screen.getByText(/total/i)).toBeInTheDocument()
  })

  it('should show templates panel', () => {
    render(<ProactiveModelBuilder onModelCreated={mockOnModelCreated} />)
    
    const templatesButton = screen.getByRole('button', { name: /plantillas/i })
    fireEvent.click(templatesButton)

    expect(screen.getByText(/plantillas/i)).toBeInTheDocument()
  })

  it('should handle pause/play', async () => {
    render(<ProactiveModelBuilder onModelCreated={mockOnModelCreated} />)
    
    const startButton = screen.getByRole('button', { name: /iniciar/i })
    fireEvent.click(startButton)

    await waitFor(() => {
      const pauseButton = screen.getByRole('button', { name: /pausar/i })
      expect(pauseButton).toBeInTheDocument()
    })
  })
})










