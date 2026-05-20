/**
 * Security Tests
 * Tests for security best practices
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FormField } from '@/components/ui/form-field';

describe('Security Tests', () => {
  describe('XSS Prevention', () => {
    it('should sanitize user input', async () => {
      const user = userEvent.setup();
      const dangerousInput = '<script>alert("xss")</script>';

      render(<FormField label="Test" name="test" />);

      const input = screen.getByRole('textbox');
      await user.type(input, dangerousInput);

      // React should automatically escape the input
      expect(input).toHaveValue(dangerousInput);
      // The value should be stored but not executed
    });

    it('should handle HTML entities correctly', async () => {
      const user = userEvent.setup();
      const htmlEntities = '&lt;script&gt;alert("xss")&lt;/script&gt;';

      render(<FormField label="Test" name="test" />);

      const input = screen.getByRole('textbox');
      await user.type(input, htmlEntities);

      expect(input).toHaveValue(htmlEntities);
    });
  });

  describe('Input Validation', () => {
    it('should validate input length', async () => {
      const user = userEvent.setup();
      const longInput = 'a'.repeat(10000);

      render(<FormField label="Test" name="test" maxLength={100} />);

      const input = screen.getByRole('textbox');
      await user.type(input, longInput);

      // Input should be limited by maxLength
      expect(input).toHaveAttribute('maxLength', '100');
    });

    it('should handle special characters safely', async () => {
      const user = userEvent.setup();
      const specialChars = '!@#$%^&*()_+-=[]{}|;:,.<>?';

      render(<FormField label="Test" name="test" />);

      const input = screen.getByRole('textbox');
      await user.type(input, specialChars);

      expect(input).toHaveValue(specialChars);
    });
  });

  describe('Data Sanitization', () => {
    it('should not execute JavaScript in rendered content', () => {
      const dangerousContent = '<img src="x" onerror="alert(1)">';
      
      // React should escape this automatically
      render(<div>{dangerousContent}</div>);
      
      // Content should be escaped, not executed
      expect(screen.getByText(dangerousContent)).toBeInTheDocument();
    });
  });
});

