import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ErrorBoundary } from '@/components/error-boundary';
import { getErrorMessage } from '@/lib/errors';

// Component that throws an error
const ThrowError = ({ shouldThrow }: { shouldThrow: boolean }) => {
  if (shouldThrow) {
    throw new Error('Test error');
  }
  return <div>No error</div>;
};

// Suppress console.error for error boundary tests
const originalError = console.error;
beforeAll(() => {
  console.error = jest.fn();
});

afterAll(() => {
  console.error = originalError;
});

describe('ErrorBoundary', () => {
  it('should render children when there is no error', () => {
    render(
      <ErrorBoundary>
        <div>Test content</div>
      </ErrorBoundary>
    );

    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('should catch and display error when child component throws', () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
  });

  it('should display error message', () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    const errorMessage = getErrorMessage(new Error('Test error'));
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('should show error details when expanded', async () => {
    const user = userEvent.setup();
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    const detailsButton = screen.getByText(/error details/i);
    await user.click(detailsButton);

    expect(screen.getByText(/test error/i)).toBeInTheDocument();
  });

  it('should call onError callback when error occurs', () => {
    const onError = jest.fn();

    render(
      <ErrorBoundary onError={onError}>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(onError).toHaveBeenCalled();
    expect(onError).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({
        componentStack: expect.any(String),
      })
    );
  });

  it('should render custom fallback when provided', () => {
    const customFallback = <div>Custom error message</div>;

    render(
      <ErrorBoundary fallback={customFallback}>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText('Custom error message')).toBeInTheDocument();
    expect(screen.queryByText(/something went wrong/i)).not.toBeInTheDocument();
  });

  it('should reset error state when reset button is clicked', async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();

    const resetButton = screen.getByText(/try again/i);
    await user.click(resetButton);

    // Re-render with error fixed
    rerender(
      <ErrorBoundary>
        <ThrowError shouldThrow={false} />
      </ErrorBoundary>
    );

    expect(screen.getByText('No error')).toBeInTheDocument();
  });

  it('should display error stack trace in details', async () => {
    const user = userEvent.setup();
    const testError = new Error('Test error');
    testError.stack = 'Error: Test error\n    at test.js:1:1';

    const ThrowSpecificError = () => {
      throw testError;
    };

    render(
      <ErrorBoundary>
        <ThrowSpecificError />
      </ErrorBoundary>
    );

    const detailsButton = screen.getByText(/error details/i);
    await user.click(detailsButton);

    expect(screen.getByText(/at test\.js:1:1/i)).toBeInTheDocument();
  });
});

