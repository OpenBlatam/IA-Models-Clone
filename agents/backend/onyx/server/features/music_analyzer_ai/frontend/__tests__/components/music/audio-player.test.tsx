import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { type Track } from '@/lib/api/music-api';

// Mock audio element
const mockPlay = jest.fn().mockResolvedValue(undefined);
const mockPause = jest.fn();
const mockLoad = jest.fn();
const mockAddEventListener = jest.fn();
const mockRemoveEventListener = jest.fn();

// Mock HTMLAudioElement
Object.defineProperty(global, 'HTMLAudioElement', {
  writable: true,
  value: class HTMLAudioElement {
    play = mockPlay;
    pause = mockPause;
    load = mockLoad;
    addEventListener = mockAddEventListener;
    removeEventListener = mockRemoveEventListener;
    currentTime = 0;
    duration = 100;
    volume = 1;
    paused = true;
  },
});

const mockTrack: Track = {
  id: '1',
  name: 'Test Track',
  artist: 'Test Artist',
  preview_url: 'https://example.com/preview.mp3',
  images: [{ url: 'https://example.com/image.jpg' }],
  duration_ms: 200000,
  album: { name: 'Test Album' },
};

const mockTrackWithoutPreview: Track = {
  ...mockTrack,
  preview_url: null,
};

describe('AudioPlayer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockPlay.mockResolvedValue(undefined);
  });

  it('should render track information', () => {
    render(<AudioPlayer track={mockTrack} />);

    expect(screen.getByText('Test Track')).toBeInTheDocument();
    expect(screen.getByText('Test Artist')).toBeInTheDocument();
  });

  it('should show message when preview is not available', () => {
    render(<AudioPlayer track={mockTrackWithoutPreview} />);

    expect(screen.getByText(/preview no disponible/i)).toBeInTheDocument();
  });

  it('should render play button initially', () => {
    render(<AudioPlayer track={mockTrack} />);

    const playButton = screen.getByRole('button', { name: /play/i });
    expect(playButton).toBeInTheDocument();
  });

  it('should toggle play/pause when play button is clicked', async () => {
    const user = userEvent.setup();
    render(<AudioPlayer track={mockTrack} />);

    const playButton = screen.getByRole('button', { name: /play/i });
    await user.click(playButton);

    await waitFor(() => {
      expect(mockPlay).toHaveBeenCalled();
    });
  });

  it('should call onNext when next button is clicked', async () => {
    const user = userEvent.setup();
    const onNext = jest.fn();

    render(<AudioPlayer track={mockTrack} onNext={onNext} />);

    const nextButton = screen.getByRole('button', { name: /next|skip forward/i });
    await user.click(nextButton);

    expect(onNext).toHaveBeenCalled();
  });

  it('should call onPrevious when previous button is clicked', async () => {
    const user = userEvent.setup();
    const onPrevious = jest.fn();

    render(<AudioPlayer track={mockTrack} onPrevious={onPrevious} />);

    const prevButton = screen.getByRole('button', { name: /previous|skip back/i });
    await user.click(prevButton);

    expect(onPrevious).toHaveBeenCalled();
  });

  it('should display track image when available', () => {
    render(<AudioPlayer track={mockTrack} />);

    const image = screen.getByAltText(/test track/i);
    expect(image).toHaveAttribute('src', 'https://example.com/image.jpg');
  });

  it('should handle volume changes', async () => {
    const user = userEvent.setup();
    render(<AudioPlayer track={mockTrack} />);

    const volumeSlider = screen.getByRole('slider', { name: /volume/i });
    await user.type(volumeSlider, '0.5');

    // Volume should be updated
    expect(volumeSlider).toHaveValue('0.5');
  });

  it('should format time correctly', () => {
    render(<AudioPlayer track={mockTrack} />);

    // Should display formatted time (0:00 format)
    const timeDisplay = screen.getByText(/0:00|0:00/);
    expect(timeDisplay).toBeInTheDocument();
  });

  it('should handle seek functionality', async () => {
    const user = userEvent.setup();
    render(<AudioPlayer track={mockTrack} />);

    const seekSlider = screen.getByRole('slider', { name: /progress|seek/i });
    await user.type(seekSlider, '50');

    // Seek should update current time
    expect(seekSlider).toHaveValue('50');
  });
});

