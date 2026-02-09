/**
 * Custom hook for voice features
 * Handles voice input, output, recording, and dictation
 */

import { useState, useCallback, useRef, useEffect } from 'react'

export interface VoiceFeaturesState {
  voiceInputEnabled: boolean
  voiceOutputEnabled: boolean
  isRecording: boolean
  dictationMode: boolean
  audioRecording: boolean
  videoRecording: boolean
  recordingDuration: number
  transcription: string
}

export interface VoiceFeaturesActions {
  setVoiceInputEnabled: (enabled: boolean) => void
  setVoiceOutputEnabled: (enabled: boolean) => void
  startRecording: () => Promise<void>
  stopRecording: () => void
  toggleDictation: () => void
  startAudioRecording: () => Promise<void>
  stopAudioRecording: () => void
  startVideoRecording: () => Promise<void>
  stopVideoRecording: () => void
  clearTranscription: () => void
}

export function useVoiceFeatures(): VoiceFeaturesState & VoiceFeaturesActions {
  const [voiceInputEnabled, setVoiceInputEnabled] = useState(false)
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [dictationMode, setDictationMode] = useState(false)
  const [audioRecording, setAudioRecording] = useState(false)
  const [videoRecording, setVideoRecording] = useState(false)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [transcription, setTranscription] = useState('')

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const recognitionRef = useRef<any>(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current)
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
    }
  }, [])

  const startRecording = useCallback(async () => {
    try {
      // Check for browser support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Voice recording not supported in this browser')
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder

      const chunks: Blob[] = []
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' })
        // Handle audio blob (e.g., send to transcription API)
        console.log('Recording stopped, blob size:', blob.size)
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingDuration(0)

      // Start duration counter
      durationIntervalRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1)
      }, 1000)
    } catch (error) {
      console.error('Error starting recording:', error)
      throw error
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current)
      durationIntervalRef.current = null
    }
    setIsRecording(false)
    setRecordingDuration(0)
  }, [])

  const toggleDictation = useCallback(() => {
    if (dictationMode) {
      // Stop dictation
      if (recognitionRef.current) {
        recognitionRef.current.stop()
        recognitionRef.current = null
      }
      setDictationMode(false)
      setTranscription('')
    } else {
      // Start dictation
      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
        const recognition = new SpeechRecognition()
        
        recognition.continuous = true
        recognition.interimResults = true
        recognition.lang = 'es-ES'

        recognition.onresult = (event: any) => {
          let interimTranscript = ''
          let finalTranscript = ''

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript
            if (event.results[i].isFinal) {
              finalTranscript += transcript + ' '
            } else {
              interimTranscript += transcript
            }
          }

          setTranscription(finalTranscript + interimTranscript)
        }

        recognition.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error)
        }

        recognition.onend = () => {
          if (dictationMode) {
            recognition.start() // Restart if still in dictation mode
          }
        }

        recognitionRef.current = recognition
        recognition.start()
        setDictationMode(true)
      } else {
        console.warn('Speech recognition not supported')
      }
    }
  }, [dictationMode])

  const startAudioRecording = useCallback(async () => {
    await startRecording()
    setAudioRecording(true)
  }, [startRecording])

  const stopAudioRecording = useCallback(() => {
    stopRecording()
    setAudioRecording(false)
  }, [stopRecording])

  const startVideoRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      streamRef.current = stream

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder

      const chunks: Blob[] = []
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' })
        console.log('Video recording stopped, blob size:', blob.size)
      }

      mediaRecorder.start()
      setVideoRecording(true)
      setRecordingDuration(0)

      durationIntervalRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1)
      }, 1000)
    } catch (error) {
      console.error('Error starting video recording:', error)
      throw error
    }
  }, [])

  const stopVideoRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current)
      durationIntervalRef.current = null
    }
    setVideoRecording(false)
    setRecordingDuration(0)
  }, [])

  const clearTranscription = useCallback(() => {
    setTranscription('')
  }, [])

  return {
    // State
    voiceInputEnabled,
    voiceOutputEnabled,
    isRecording,
    dictationMode,
    audioRecording,
    videoRecording,
    recordingDuration,
    transcription,
    // Actions
    setVoiceInputEnabled,
    setVoiceOutputEnabled,
    startRecording,
    stopRecording,
    toggleDictation,
    startAudioRecording,
    stopAudioRecording,
    startVideoRecording,
    stopVideoRecording,
    clearTranscription,
  }
}




