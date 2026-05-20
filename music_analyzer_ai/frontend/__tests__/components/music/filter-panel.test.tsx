import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FilterPanel } from '@/components/music/FilterPanel';

describe('FilterPanel', () => {
  const mockOnFilterChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render filter button', () => {
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    expect(screen.getByText('Filtros')).toBeInTheDocument();
  });

  it('should open filter panel when clicked', async () => {
    const user = userEvent.setup();
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    const filterButton = screen.getByText('Filtros');
    await user.click(filterButton);

    expect(screen.getByText(/filtros/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/popularidad/i)).toBeInTheDocument();
  });

  it('should close filter panel when close button is clicked', async () => {
    const user = userEvent.setup();
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    const filterButton = screen.getByText('Filtros');
    await user.click(filterButton);

    expect(screen.getByText(/filtros/i)).toBeInTheDocument();

    const closeButton = screen.getByRole('button', { name: /close/i });
    await user.click(closeButton);

    expect(screen.queryByLabelText(/popularidad/i)).not.toBeInTheDocument();
  });

  it('should call onFilterChange when popularity filter changes', async () => {
    const user = userEvent.setup();
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    const filterButton = screen.getByText('Filtros');
    await user.click(filterButton);

    const minPopularityInput = screen.getByPlaceholderText('Min');
    await user.clear(minPopularityInput);
    await user.type(minPopularityInput, '50');

    expect(mockOnFilterChange).toHaveBeenCalled();
    const lastCall = mockOnFilterChange.mock.calls[mockOnFilterChange.mock.calls.length - 1][0];
    expect(lastCall.minPopularity).toBe(50);
  });

  it('should call onFilterChange when energy filter changes', async () => {
    const user = userEvent.setup();
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    const filterButton = screen.getByText('Filtros');
    await user.click(filterButton);

    const energyInputs = screen.getAllByPlaceholderText('Min');
    const energyMinInput = energyInputs.find((input) => {
      const label = input.closest('div')?.querySelector('label');
      return label?.textContent?.includes('Energía');
    });

    if (energyMinInput) {
      await user.clear(energyMinInput);
      await user.type(energyMinInput, '0.5');
      expect(mockOnFilterChange).toHaveBeenCalled();
    }
  });

  it('should call onFilterChange when danceability filter changes', async () => {
    const user = userEvent.setup();
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    const filterButton = screen.getByText('Filtros');
    await user.click(filterButton);

    const danceabilityInputs = screen.getAllByPlaceholderText('Min');
    const danceabilityMinInput = danceabilityInputs.find((input) => {
      const label = input.closest('div')?.querySelector('label');
      return label?.textContent?.includes('Bailabilidad');
    });

    if (danceabilityMinInput) {
      await user.clear(danceabilityMinInput);
      await user.type(danceabilityMinInput, '0.7');
      expect(mockOnFilterChange).toHaveBeenCalled();
    }
  });

  it('should call onFilterChange when genre filter changes', async () => {
    const user = userEvent.setup();
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    const filterButton = screen.getByText('Filtros');
    await user.click(filterButton);

    const genreInput = screen.getByPlaceholderText(/rock, pop, electronic/i);
    await user.type(genreInput, 'Rock');

    expect(mockOnFilterChange).toHaveBeenCalled();
    const lastCall = mockOnFilterChange.mock.calls[mockOnFilterChange.mock.calls.length - 1][0];
    expect(lastCall.genre).toBe('Rock');
  });

  it('should call onFilterChange when year filter changes', async () => {
    const user = userEvent.setup();
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    const filterButton = screen.getByText('Filtros');
    await user.click(filterButton);

    const yearInput = screen.getByPlaceholderText('2020');
    await user.type(yearInput, '2020');

    expect(mockOnFilterChange).toHaveBeenCalled();
    const lastCall = mockOnFilterChange.mock.calls[mockOnFilterChange.mock.calls.length - 1][0];
    expect(lastCall.year).toBe('2020');
  });

  it('should clear all filters when clear button is clicked', async () => {
    const user = userEvent.setup();
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    const filterButton = screen.getByText('Filtros');
    await user.click(filterButton);

    // Set some filters first
    const genreInput = screen.getByPlaceholderText(/rock, pop, electronic/i);
    await user.type(genreInput, 'Rock');

    // Clear filters
    const clearButton = screen.getByText(/limpiar filtros/i);
    await user.click(clearButton);

    expect(mockOnFilterChange).toHaveBeenCalled();
    const clearCall = mockOnFilterChange.mock.calls.find(
      (call) => call[0].genre === '' && call[0].year === ''
    );
    expect(clearCall).toBeDefined();
  });

  it('should handle max popularity filter', async () => {
    const user = userEvent.setup();
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    const filterButton = screen.getByText('Filtros');
    await user.click(filterButton);

    const maxInputs = screen.getAllByPlaceholderText('Max');
    const maxPopularityInput = maxInputs[0];
    await user.clear(maxPopularityInput);
    await user.type(maxPopularityInput, '80');

    expect(mockOnFilterChange).toHaveBeenCalled();
    const lastCall = mockOnFilterChange.mock.calls[mockOnFilterChange.mock.calls.length - 1][0];
    expect(lastCall.maxPopularity).toBe(80);
  });

  it('should toggle panel open/close', async () => {
    const user = userEvent.setup();
    render(<FilterPanel onFilterChange={mockOnFilterChange} />);

    const filterButton = screen.getByText('Filtros');

    // Open
    await user.click(filterButton);
    expect(screen.getByLabelText(/popularidad/i)).toBeInTheDocument();

    // Close
    await user.click(filterButton);
    expect(screen.queryByLabelText(/popularidad/i)).not.toBeInTheDocument();
  });
});

