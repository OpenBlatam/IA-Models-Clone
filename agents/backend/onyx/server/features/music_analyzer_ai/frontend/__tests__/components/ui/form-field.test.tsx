import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FormField } from '@/components/ui/form-field';

describe('FormField', () => {
  it('should render label', () => {
    render(<FormField label="Test Label" />);
    expect(screen.getByText('Test Label')).toBeInTheDocument();
  });

  it('should render input', () => {
    render(<FormField label="Test" name="test" />);
    expect(screen.getByRole('textbox')).toBeInTheDocument();
  });

  it('should display error messages', () => {
    render(<FormField label="Test" name="test" errors={['Error 1', 'Error 2']} touched />);
    expect(screen.getByText('Error 1')).toBeInTheDocument();
    expect(screen.getByText('Error 2')).toBeInTheDocument();
  });

  it('should not display errors when not touched', () => {
    render(<FormField label="Test" name="test" errors={['Error']} touched={false} />);
    expect(screen.queryByText('Error')).not.toBeInTheDocument();
  });

  it('should handle input changes', async () => {
    const user = userEvent.setup();
    const onChange = jest.fn();

    render(<FormField label="Test" name="test" onChange={onChange} />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'test value');

    expect(onChange).toHaveBeenCalled();
  });

  it('should be disabled when disabled prop is true', () => {
    render(<FormField label="Test" name="test" disabled />);
    expect(screen.getByRole('textbox')).toBeDisabled();
  });

  it('should show required indicator', () => {
    render(<FormField label="Test" name="test" required />);
    const requiredIndicator = screen.getByText('*');
    expect(requiredIndicator).toBeInTheDocument();
  });

  it('should render helper text when no errors', () => {
    render(<FormField label="Test" name="test" helperText="Helper text" />);
    expect(screen.getByText('Helper text')).toBeInTheDocument();
  });

  it('should not show helper text when errors are present', () => {
    render(
      <FormField
        label="Test"
        name="test"
        helperText="Helper text"
        errors={['Error']}
        touched
      />
    );
    expect(screen.queryByText('Helper text')).not.toBeInTheDocument();
  });

  it('should handle different input types', () => {
    const { rerender } = render(
      <FormField label="Test" name="test" type="email" />
    );
    expect(screen.getByRole('textbox')).toHaveAttribute('type', 'email');

    rerender(<FormField label="Test" name="test" type="password" />);
    expect(screen.getByLabelText('Test')).toHaveAttribute('type', 'password');
  });

  it('should have correct aria attributes when error', () => {
    render(<FormField label="Test" name="test" errors={['Error']} touched />);
    const input = screen.getByRole('textbox');
    expect(input).toHaveAttribute('aria-invalid', 'true');
  });

  it('should generate field ID from label', () => {
    render(<FormField label="Test Field" name="test" />);
    const input = screen.getByLabelText('Test Field');
    expect(input).toHaveAttribute('id', 'field-test-field');
  });
});

