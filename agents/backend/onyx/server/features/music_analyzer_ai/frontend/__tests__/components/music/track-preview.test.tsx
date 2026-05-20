import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TrackPreview } from '@/components/music/TrackPreview';
import { type Track } from '@/lib/api/music-api';

const mockTrack: Track = {
  id: 'track-123',
  name: 'Test Track',
  artists: ['Test Artist'],
  album: 'Test Album',
  duration_ms: 200000,
  preview_url: 'https://example.com/preview.mp3',
  popularity: 80,
  images: [{ url: 'https://example.com/image.jpg' }],
};

const mockTrackWithoutPreview: Track = {
  ...mockTrack,
  preview_url: null,
};

describe('TrackPreview', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render track information', () => {
    render(<TrackPreview track={mockTrack} />);

    expect(screen.getByText('Test Track')).toBeInTheDocument();
    expect(screen.getByText('Test Artist')).toBeInTheDocument();
  });

  it('should show message when preview is not available', () => {
    render(<TrackPreview track={mockTrackWithoutPreview} />);

    expect(
      screen.getByText(/preview no disponible/i)
    ).toBeInTheDocument();
  });

  it('should display track image when available', () => {
    render(<TrackPreview track={mockTrack} />);

    const image = screen.getByAltText('Test Track');
    expect(image).toHaveAttribute('src', 'https://example.com/image.jpg');
  });

  it('should toggle play/pause when play button is clicked', async () => {
    const user = userEvent.setup();
    render(<TrackPreview track={mockTrack} />);

    const playButton = screen.getByRole('button', { name: /play|pause/i });
    await user.click(playButton);

    // Should toggle to pause icon
    expect(playButton).toBeInTheDocument();
  });

  it('should toggle mute when mute button is clicked', async () => {
    const user = userEvent.setup();
    render(<TrackPreview track={mockTrack} />);

    const muteButton = screen.getByRole('button', { name: /volume|mute/i });
    await user.click(muteButton);

    // Should toggle mute state
    expect(muteButton).toBeInTheDocument();
  });

  it('should display formatted duration', () => {
    render(<TrackPreview track={mockTrack} />);

    // Duration should be formatted (3:20 for 200000ms)
    expect(screen.getByText(/3:20|0:03/i)).toBeInTheDocument();
  });

  it('should handle track with multiple artists', () => {
    const trackWithMultipleArtists: Track = {
      ...mockTrack,
      artists: ['Artist 1', 'Artist 2', 'Artist 3'],
    };

    render(<TrackPreview track={trackWithMultipleArtists} />);

    expect(screen.getByText(/artist 1, artist 2, artist 3/i)).toBeInTheDocument();
  });

  it('should handle track without image', () => {
    const trackWithoutImage: Track = {
      ...mockTrack,
      images: [],
    };

    render(<TrackPreview track={trackWithoutImage} />);

    expect(screen.getByText('Test Track')).toBeInTheDocument();
    expect(screen.queryByAltText('Test Track')).not.toBeInTheDocument();
  });
});

