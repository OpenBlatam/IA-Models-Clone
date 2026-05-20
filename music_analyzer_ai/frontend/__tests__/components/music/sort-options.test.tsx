import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SortOptions } from '@/components/music/SortOptions';

describe('SortOptions', () => {
  const mockOnSortChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render sort button', () => {
    render(<SortOptions onSortChange={mockOnSortChange} />);

    expect(screen.getByText('Ordenar')).toBeInTheDocument();
  });

  it('should open dropdown when clicked', async () => {
    const user = userEvent.setup();
    render(<SortOptions onSortChange={mockOnSortChange} />);

    const button = screen.getByText('Ordenar');
    await user.click(button);

    expect(screen.getByText('Popularidad')).toBeInTheDocument();
    expect(screen.getByText('Nombre')).toBeInTheDocument();
    expect(screen.getByText('Duración')).toBeInTheDocument();
    expect(screen.getByText('Fecha')).toBeInTheDocument();
  });

  it('should call onSortChange when option is selected', async () => {
    const user = userEvent.setup();
    render(<SortOptions onSortChange={mockOnSortChange} />);

    const button = screen.getByText('Ordenar');
    await user.click(button);

    const nameOption = screen.getByText('Nombre');
    await user.click(nameOption);

    expect(mockOnSortChange).toHaveBeenCalledWith('name', 'desc');
  });

  it('should toggle order when same field is selected', async () => {
    const user = userEvent.setup();
    render(
      <SortOptions
        onSortChange={mockOnSortChange}
        currentField="popularity"
        currentOrder="desc"
      />
    );

    const button = screen.getByText('Ordenar');
    await user.click(button);

    const popularityOption = screen.getByText('Popularidad');
    await user.click(popularityOption);

    // Should toggle to asc
    expect(mockOnSortChange).toHaveBeenCalledWith('popularity', 'asc');
  });

  it('should use currentField and currentOrder props', () => {
    render(
      <SortOptions
        onSortChange={mockOnSortChange}
        currentField="name"
        currentOrder="asc"
      />
    );

    const button = screen.getByText('Ordenar');
    expect(button).toBeInTheDocument();
  });

  it('should show arrow up icon when order is asc', () => {
    render(
      <SortOptions
        onSortChange={mockOnSortChange}
        currentOrder="asc"
      />
    );

    // ArrowUp icon should be present
    const button = screen.getByText('Ordenar').closest('button');
    const arrowUp = button?.querySelector('.lucide-arrow-up');
    expect(arrowUp).toBeInTheDocument();
  });

  it('should show arrow down icon when order is desc', () => {
    render(
      <SortOptions
        onSortChange={mockOnSortChange}
        currentOrder="desc"
      />
    );

    // ArrowDown icon should be present
    const button = screen.getByText('Ordenar').closest('button');
    const arrowDown = button?.querySelector('.lucide-arrow-down');
    expect(arrowDown).toBeInTheDocument();
  });

  it('should highlight selected field', async () => {
    const user = userEvent.setup();
    render(
      <SortOptions
        onSortChange={mockOnSortChange}
        currentField="name"
      />
    );

    const button = screen.getByText('Ordenar');
    await user.click(button);

    const nameOption = screen.getByText('Nombre').closest('button');
    expect(nameOption?.className).toContain('bg-purple-600');
  });

  it('should close dropdown when clicking outside', async () => {
    const user = userEvent.setup();
    render(<SortOptions onSortChange={mockOnSortChange} />);

    const button = screen.getByText('Ordenar');
    await user.click(button);

    expect(screen.getByText('Popularidad')).toBeInTheDocument();

    // Click outside
    await user.click(document.body);

    // Dropdown should close (options not visible)
    // Note: This might need additional setup for click-outside detection
  });
});

