import { render, screen } from '@testing-library/react';
import { ProgressIndicator } from '@/components/music/ProgressIndicator';

const mockSteps = [
  { id: '1', label: 'Step 1', status: 'completed' as const },
  { id: '2', label: 'Step 2', status: 'loading' as const },
  { id: '3', label: 'Step 3', status: 'pending' as const },
  { id: '4', label: 'Step 4', status: 'error' as const },
];

describe('ProgressIndicator', () => {
  it('should render all steps', () => {
    render(<ProgressIndicator steps={mockSteps} />);

    expect(screen.getByText('Step 1')).toBeInTheDocument();
    expect(screen.getByText('Step 2')).toBeInTheDocument();
    expect(screen.getByText('Step 3')).toBeInTheDocument();
    expect(screen.getByText('Step 4')).toBeInTheDocument();
  });

  it('should render title', () => {
    render(<ProgressIndicator steps={mockSteps} />);

    expect(screen.getByText(/progreso del análisis/i)).toBeInTheDocument();
  });

  it('should show completed status with check icon', () => {
    render(<ProgressIndicator steps={mockSteps} />);

    const step1 = screen.getByText('Step 1').closest('div');
    expect(step1).toBeInTheDocument();
    // Check icon should be present for completed status
    const checkIcon = step1?.querySelector('.text-green-400');
    expect(checkIcon).toBeInTheDocument();
  });

  it('should show loading status with spinner', () => {
    render(<ProgressIndicator steps={mockSteps} />);

    const step2 = screen.getByText('Step 2').closest('div');
    expect(step2).toBeInTheDocument();
    // Spinner should be present for loading status
    const spinner = step2?.querySelector('.animate-spin');
    expect(spinner).toBeInTheDocument();
  });

  it('should show pending status with empty circle', () => {
    render(<ProgressIndicator steps={mockSteps} />);

    const step3 = screen.getByText('Step 3').closest('div');
    expect(step3).toBeInTheDocument();
    // Empty circle should be present for pending status
    const emptyCircle = step3?.querySelector('.rounded-full.border-2');
    expect(emptyCircle).toBeInTheDocument();
  });

  it('should show error status with error icon', () => {
    render(<ProgressIndicator steps={mockSteps} />);

    const step4 = screen.getByText('Step 4').closest('div');
    expect(step4).toBeInTheDocument();
    // Error icon should be present for error status
    const errorIcon = step4?.querySelector('.text-red-400');
    expect(errorIcon).toBeInTheDocument();
  });

  it('should show progress bar for loading step', () => {
    render(<ProgressIndicator steps={mockSteps} />);

    const step2 = screen.getByText('Step 2').closest('div');
    const progressBar = step2?.querySelector('.bg-purple-400');
    expect(progressBar).toBeInTheDocument();
  });

  it('should apply correct text colors for each status', () => {
    render(<ProgressIndicator steps={mockSteps} />);

    const step1 = screen.getByText('Step 1');
    expect(step1.className).toContain('text-green-300');

    const step2 = screen.getByText('Step 2');
    expect(step2.className).toContain('text-purple-300');

    const step4 = screen.getByText('Step 4');
    expect(step4.className).toContain('text-red-300');

    const step3 = screen.getByText('Step 3');
    expect(step3.className).toContain('text-gray-400');
  });

  it('should handle empty steps array', () => {
    render(<ProgressIndicator steps={[]} />);

    expect(screen.getByText(/progreso del análisis/i)).toBeInTheDocument();
  });

  it('should handle single step', () => {
    const singleStep = [{ id: '1', label: 'Single Step', status: 'loading' as const }];
    render(<ProgressIndicator steps={singleStep} />);

    expect(screen.getByText('Single Step')).toBeInTheDocument();
  });

  it('should use currentStep prop if provided', () => {
    render(<ProgressIndicator steps={mockSteps} currentStep="2" />);

    // Component should still render all steps
    expect(screen.getByText('Step 1')).toBeInTheDocument();
    expect(screen.getByText('Step 2')).toBeInTheDocument();
  });
});

