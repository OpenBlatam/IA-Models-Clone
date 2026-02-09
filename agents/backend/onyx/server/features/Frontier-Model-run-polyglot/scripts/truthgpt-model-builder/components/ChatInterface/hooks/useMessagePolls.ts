/**
 * Hook modular para encuestas y votaciones
 */

import { useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { PollService } from '../services/PollService'
import type { MessagePollsState } from '../types/state.types'

// MessagePollsState ahora se importa de types/state.types

export interface MessagePollsActions {
  createPoll: (messageId: string, question: string, options: string[]) => void
  votePoll: (messageId: string, optionIndex: number) => void
  rateMessage: (messageId: string, rating: number) => void
}

export function useMessagePolls(
  state: MessagePollsState,
  setState: React.Dispatch<React.SetStateAction<MessagePollsState>>
): MessagePollsActions {
  
  const createPoll = useCallback((messageId: string, question: string, options: string[]) => {
    setState(prev => ({
      ...prev,
      messagePolls: PollService.createPoll(prev.messagePolls, messageId, question, options)
    }))
    toast.success('Encuesta creada', { icon: '📊' })
  }, [setState])

  const votePoll = useCallback((messageId: string, optionIndex: number) => {
    setState(prev => ({
      ...prev,
      messagePolls: PollService.votePoll(prev.messagePolls, messageId, optionIndex)
    }))
    toast.success('Voto registrado', { icon: '✅' })
  }, [setState])

  const rateMessage = useCallback((messageId: string, rating: number) => {
    toast.success(`Calificación: ${rating}/5`, { icon: '⭐' })
    // La calificación se puede almacenar en un estado separado si es necesario
  }, [])

  return {
    createPoll,
    votePoll,
    rateMessage
  }
}

