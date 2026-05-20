/**
 * Store-Component Integration Tests
 * Tests for integration between Zustand store and React components
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useMusicStore } from '@/lib/store/music-store';
import { type Track } from '@/lib/api/types';

// Mock component that uses the store
const TestComponent = () => {
  const {
    currentTrack,
    playlistQueue,
    isPlaying,
    setCurrentTrack,
    setIsPlaying,
    addToQueue,
  } = useMusicStore();

  const mockTrack: Track = {
    id: '1',
    name: 'Test Track',
    artists: ['Artist 1'],
    duration_ms: 200000,
    preview_url: 'https://example.com/preview.mp3',
    images: [],
  };

  return (
    <div>
      <div data-testid="current-track">
        {currentTrack ? currentTrack.name : 'No track'}
      </div>
      <div data-testid="queue-length">{playlistQueue.length}</div>
      <div data-testid="playing">{isPlaying ? 'Playing' : 'Paused'}</div>
      <button
        onClick={() => setCurrentTrack(mockTrack)}
        data-testid="set-track"
      >
        Set Track
      </button>
      <button
        onClick={() => setIsPlaying(!isPlaying)}
        data-testid="toggle-play"
      >
        Toggle Play
      </button>
      <button onClick={() => addToQueue(mockTrack)} data-testid="add-queue">
        Add to Queue
      </button>
    </div>
  );
};

describe('Store-Component Integration', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    // Reset store state
    useMusicStore.getState().clearQueue();
    useMusicStore.getState().setCurrentTrack(null);
    useMusicStore.getState().setIsPlaying(false);
  });

  it('should update component when store state changes', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <TestComponent />
      </QueryClientProvider>
    );

    expect(screen.getByTestId('current-track')).toHaveTextContent('No track');
    expect(screen.getByTestId('queue-length')).toHaveTextContent('0');
    expect(screen.getByTestId('playing')).toHaveTextContent('Paused');
  });

  it('should update store when component actions are triggered', async () => {
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={queryClient}>
        <TestComponent />
      </QueryClientProvider>
    );

    const setTrackButton = screen.getByTestId('set-track');
    await user.click(setTrackButton);

    await waitFor(() => {
      expect(screen.getByTestId('current-track')).toHaveTextContent(
        'Test Track'
      );
    });

    expect(useMusicStore.getState().currentTrack?.name).toBe('Test Track');
  });

  it('should handle play/pause toggle', async () => {
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={queryClient}>
        <TestComponent />
      </QueryClientProvider>
    );

    const toggleButton = screen.getByTestId('toggle-play');
    await user.click(toggleButton);

    await waitFor(() => {
      expect(screen.getByTestId('playing')).toHaveTextContent('Playing');
    });

    expect(useMusicStore.getState().playback.isPlaying).toBe(true);
  });

  it('should add tracks to queue', async () => {
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={queryClient}>
        <TestComponent />
      </QueryClientProvider>
    );

    const addButton = screen.getByTestId('add-queue');
    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByTestId('queue-length')).toHaveTextContent('1');
    });

    expect(useMusicStore.getState().playlistQueue.length).toBe(1);
  });

  it('should sync multiple components with same store', () => {
    const { container: container1 } = render(
      <QueryClientProvider client={queryClient}>
        <TestComponent />
      </QueryClientProvider>
    );

    const { container: container2 } = render(
      <QueryClientProvider client={queryClient}>
        <TestComponent />
      </QueryClientProvider>
    );

    // Update store from first component
    const setTrackButton1 = container1.querySelector(
      '[data-testid="set-track"]'
    ) as HTMLButtonElement;
    setTrackButton1?.click();

    // Both components should reflect the change
    waitFor(() => {
      const track1 = container1.querySelector('[data-testid="current-track"]');
      const track2 = container2.querySelector('[data-testid="current-track"]');
      expect(track1).toHaveTextContent('Test Track');
      expect(track2).toHaveTextContent('Test Track');
    });
  });

  it('should persist state across component unmounts', async () => {
    const user = userEvent.setup();
    const { unmount } = render(
      <QueryClientProvider client={queryClient}>
        <TestComponent />
      </QueryClientProvider>
    );

    const setTrackButton = screen.getByTestId('set-track');
    await user.click(setTrackButton);

    await waitFor(() => {
      expect(useMusicStore.getState().currentTrack).toBeTruthy();
    });

    unmount();

    // State should persist
    expect(useMusicStore.getState().currentTrack?.name).toBe('Test Track');

    // Re-render and verify state is still there
    render(
      <QueryClientProvider client={queryClient}>
        <TestComponent />
      </QueryClientProvider>
    );

    expect(screen.getByTestId('current-track')).toHaveTextContent(
      'Test Track'
    );
  });

  it('should handle complex state updates', async () => {
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={queryClient}>
        <TestComponent />
      </QueryClientProvider>
    );

    // Set track
    await user.click(screen.getByTestId('set-track'));
    await waitFor(() => {
      expect(screen.getByTestId('current-track')).toHaveTextContent(
        'Test Track'
      );
    });

    // Add to queue
    await user.click(screen.getByTestId('add-queue'));
    await waitFor(() => {
      expect(screen.getByTestId('queue-length')).toHaveTextContent('1');
    });

    // Toggle play
    await user.click(screen.getByTestId('toggle-play'));
    await waitFor(() => {
      expect(screen.getByTestId('playing')).toHaveTextContent('Playing');
    });

    // Verify all state
    const state = useMusicStore.getState();
    expect(state.currentTrack?.name).toBe('Test Track');
    expect(state.playlistQueue.length).toBe(1);
    expect(state.playback.isPlaying).toBe(true);
  });
});

