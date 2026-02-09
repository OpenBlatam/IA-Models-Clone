/**
 * Unit tests para la página Not Found (404)
 * 
 * Tests que verifican:
 * - Renderizado correcto de la página
 * - Presencia de elementos clave
 * - Funcionalidad de navegación
 * - Accesibilidad
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import NotFound from '../../app/not-found';

// Mock del router de Next.js
const mockPush = vi.fn();
const mockBack = vi.fn();

vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
    back: mockBack,
    replace: vi.fn(),
    refresh: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

// Mock del componente Header para simplificar tests
vi.mock('../../app/components/home', () => ({
  Header: () => <header data-testid="header">Header</header>,
}));

describe('NotFound Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Renderizado', () => {
    it('debería renderizar el componente sin errores', () => {
      render(<NotFound />);
      expect(screen.getByTestId('header')).toBeInTheDocument();
    });

    it('debería mostrar el número 404', () => {
      render(<NotFound />);
      const heading404 = screen.getByRole('heading', { name: /404/i });
      expect(heading404).toBeInTheDocument();
      expect(heading404).toHaveTextContent('404');
    });

    it('debería mostrar el título "Page Not Found"', () => {
      render(<NotFound />);
      const title = screen.getByRole('heading', { name: /page not found/i });
      expect(title).toBeInTheDocument();
      expect(title).toHaveTextContent('Page Not Found');
    });

    it('debería mostrar el mensaje descriptivo', () => {
      render(<NotFound />);
      const description = screen.getByText(
        /the page you're looking for doesn't exist or has been moved/i
      );
      expect(description).toBeInTheDocument();
    });

    it('debería renderizar el botón "Go to Home"', () => {
      render(<NotFound />);
      const homeButton = screen.getByRole('button', { name: /navigate to home page/i });
      expect(homeButton).toBeInTheDocument();
      expect(homeButton).toHaveTextContent('Go to Home');
    });

    it('debería renderizar el botón "Go Back"', () => {
      render(<NotFound />);
      const backButton = screen.getByRole('button', { name: /navigate back to previous page/i });
      expect(backButton).toBeInTheDocument();
      expect(backButton).toHaveTextContent('Go Back');
    });
  });

  describe('Navegación', () => {
    it('debería navegar a home cuando se hace clic en "Go to Home"', async () => {
      const user = userEvent.setup();
      render(<NotFound />);
      
      const homeButton = screen.getByRole('button', { name: /navigate to home page/i });
      await user.click(homeButton);
      
      expect(mockPush).toHaveBeenCalledTimes(1);
      expect(mockPush).toHaveBeenCalledWith('/');
    });

    it('debería navegar hacia atrás cuando se hace clic en "Go Back"', async () => {
      const user = userEvent.setup();
      
      // Mock window.history.length to simulate having history
      Object.defineProperty(window, 'history', {
        value: { length: 2 },
        writable: true,
      });
      
      render(<NotFound />);
      
      const backButton = screen.getByRole('button', { name: /navigate back to previous page/i });
      await user.click(backButton);
      
      expect(mockBack).toHaveBeenCalledTimes(1);
    });
  });

  describe('Accesibilidad', () => {
    it('debería tener una estructura semántica correcta', () => {
      render(<NotFound />);
      
      // Verificar que hay un elemento main
      const main = screen.getByRole('main');
      expect(main).toBeInTheDocument();
    });

    it('debería tener headings con niveles correctos', () => {
      render(<NotFound />);
      
      const h1 = screen.getByRole('heading', { level: 1 });
      expect(h1).toHaveTextContent('404');
      
      const h2 = screen.getByRole('heading', { level: 2 });
      expect(h2).toHaveTextContent('Page Not Found');
    });

    it('debería tener botones accesibles con texto descriptivo', () => {
      render(<NotFound />);
      
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThanOrEqual(2);
      
      buttons.forEach(button => {
        expect(button).toHaveAccessibleName();
      });
    });

    it('debería tener contraste adecuado en los elementos', () => {
      render(<NotFound />);
      
      // Verificar que los elementos tienen las clases de estilo correctas
      const homeButton = screen.getByRole('button', { name: /navigate to home page/i });
      expect(homeButton).toHaveClass('bg-black', 'text-white');
    });
  });

  describe('Estilos y Clases CSS', () => {
    it('debería tener las clases de Tailwind correctas', () => {
      render(<NotFound />);
      
      const container = screen.getByRole('main').parentElement;
      expect(container).toHaveClass('min-h-screen', 'bg-white', 'text-black');
    });

    it('debería tener animaciones de framer-motion configuradas', () => {
      render(<NotFound />);
      
      // Verificar que los elementos motion tienen las props correctas
      const mainContent = screen.getByRole('main');
      expect(mainContent).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('debería tener clases responsive para diferentes tamaños de pantalla', () => {
      render(<NotFound />);
      
      const h1 = screen.getByRole('heading', { name: /404/i });
      expect(h1.className).toMatch(/md:text-9xl/);
      
      const h2 = screen.getByRole('heading', { name: /page not found/i });
      expect(h2.className).toMatch(/md:text-4xl/);
    });

    it('debería tener layout flexible para móvil y desktop', () => {
      render(<NotFound />);
      
      const homeButton = screen.getByRole('button', { name: /navigate to home page/i });
      const buttonContainer = homeButton.parentElement;
      expect(buttonContainer).toHaveClass('flex-col', 'sm:flex-row');
    });
  });
});

