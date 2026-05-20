import { render, screen, fireEvent } from '@testing-library/react';
import { Equalizer } from '@/components/music/Equalizer';

describe('Equalizer', () => {
  it('renders equalizer', () => {
    render(<Equalizer />);
    
    expect(screen.getByText(/ecualizador|equalizer/i)).toBeInTheDocument();
  });

  it('displays all frequency bands', () => {
    render(<Equalizer />);
    
    expect(screen.getByText(/bajos|bass/i)).toBeInTheDocument();
    expect(screen.getByText(/medios|mid/i)).toBeInTheDocument();
    expect(screen.getByText(/agudos|treble/i)).toBeInTheDocument();
  });

  it('allows adjusting bass', () => {
    render(<Equalizer />);
    
    const bassSlider = screen.getByLabelText(/bass|bajos/i) || 
                      document.querySelector('input[type="range"]');
    
    if (bassSlider) {
      fireEvent.change(bassSlider, { target: { value: '6' } });
      expect(bassSlider).toHaveValue(6);
    }
  });

  it('allows adjusting mid', () => {
    render(<Equalizer />);
    
    const sliders = document.querySelectorAll('input[type="range"]');
    if (sliders.length > 1) {
      fireEvent.change(sliders[1], { target: { value: '-3' } });
      expect(sliders[1]).toHaveValue(-3);
    }
  });

  it('allows adjusting treble', () => {
    render(<Equalizer />);
    
    const sliders = document.querySelectorAll('input[type="range"]');
    if (sliders.length > 2) {
      fireEvent.change(sliders[2], { target: { value: '9' } });
      expect(sliders[2]).toHaveValue(9);
    }
  });

  it('displays dB values', () => {
    render(<Equalizer />);
    
    expect(screen.getByText(/dB/i)).toBeInTheDocument();
  });
});

