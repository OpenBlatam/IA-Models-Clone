import { render, screen, fireEvent } from '@testing-library/react';
import { MusicSocial } from '@/components/music/MusicSocial';
import * as toast from 'react-hot-toast';

// Mock react-hot-toast
jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
  },
}));

describe('MusicSocial', () => {
  it('renders social feed with posts', () => {
    render(<MusicSocial />);
    
    expect(screen.getByText('Social')).toBeInTheDocument();
    expect(screen.getByText('MusicLover23')).toBeInTheDocument();
    expect(screen.getByText('BeatHunter')).toBeInTheDocument();
    expect(screen.getByText('SoundExplorer')).toBeInTheDocument();
  });

  it('displays post content correctly', () => {
    render(<MusicSocial />);
    
    expect(screen.getByText('Acabo de descubrir esta increíble canción!')).toBeInTheDocument();
    expect(screen.getByText('Bohemian Rhapsody')).toBeInTheDocument();
    expect(screen.getByText('Queen')).toBeInTheDocument();
  });

  it('handles like button click', () => {
    const toastSpy = jest.spyOn(toast, 'default');
    render(<MusicSocial />);
    
    const likeButtons = screen.getAllByText(/42|38|25/);
    if (likeButtons.length > 0) {
      fireEvent.click(likeButtons[0].closest('button')!);
      expect(toastSpy.success).toHaveBeenCalledWith('Like agregado');
    }
  });

  it('handles comment button click', () => {
    const toastSpy = jest.spyOn(toast, 'default');
    render(<MusicSocial />);
    
    const commentButtons = screen.getAllByText(/8|12|6/);
    if (commentButtons.length > 0) {
      fireEvent.click(commentButtons[0].closest('button')!);
      expect(toastSpy.info).toHaveBeenCalledWith('Abrir comentarios');
    }
  });

  it('handles share button click', () => {
    const toastSpy = jest.spyOn(toast, 'default');
    render(<MusicSocial />);
    
    const shareButtons = screen.getAllByText(/5|3|9/);
    if (shareButtons.length > 0) {
      fireEvent.click(shareButtons[0].closest('button')!);
      expect(toastSpy.success).toHaveBeenCalledWith('Compartido');
    }
  });

  it('handles follow button click', () => {
    const toastSpy = jest.spyOn(toast, 'default');
    render(<MusicSocial />);
    
    const followButtons = screen.getAllByText('Seguir');
    if (followButtons.length > 0) {
      fireEvent.click(followButtons[0]);
      expect(toastSpy.success).toHaveBeenCalledWith('Usuario seguido');
    }
  });

  it('displays timestamps', () => {
    render(<MusicSocial />);
    
    expect(screen.getByText('Hace 2 horas')).toBeInTheDocument();
    expect(screen.getByText('Hace 5 horas')).toBeInTheDocument();
    expect(screen.getByText('Hace 1 día')).toBeInTheDocument();
  });
});

