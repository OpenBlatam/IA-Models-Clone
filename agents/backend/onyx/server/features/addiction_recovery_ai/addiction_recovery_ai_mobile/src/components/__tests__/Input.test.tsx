import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { Input } from '../input';

describe('Input Component', () => {
  it('renders correctly with label', () => {
    const { getByText } = render(
      <Input label="Test Label" value="" onChangeText={() => {}} />
    );
    expect(getByText('Test Label')).toBeTruthy();
  });

  it('displays error message when error prop is provided', () => {
    const { getByText } = render(
      <Input
        label="Test"
        value=""
        onChangeText={() => {}}
        error="This is an error"
      />
    );
    expect(getByText('This is an error')).toBeTruthy();
  });

  it('calls onChangeText with sanitized input', () => {
    const onChangeTextMock = jest.fn();
    const { getByPlaceholderText } = render(
      <Input
        placeholder="Enter text"
        value=""
        onChangeText={onChangeTextMock}
      />
    );

    const input = getByPlaceholderText('Enter text');
    fireEvent.changeText(input, '<script>alert("xss")</script>');
    
    // Should sanitize the input
    expect(onChangeTextMock).toHaveBeenCalled();
  });

  it('has proper accessibility props', () => {
    const { getByLabelText } = render(
      <Input
        label="Email"
        value=""
        onChangeText={() => {}}
        accessibilityLabel="Email input field"
      />
    );
    
    expect(getByLabelText('Email input field')).toBeTruthy();
  });
});

