/**
 * Servicio para encuestas y votaciones
 */

import type { MessagePoll } from '../types/message.types'

export class PollService {
  /**
   * Crea una encuesta
   */
  static createPoll(
    polls: Map<string, MessagePoll>,
    messageId: string,
    question: string,
    options: string[]
  ): Map<string, MessagePoll> {
    const newMap = new Map(polls)
    newMap.set(messageId, {
      question,
      options,
      votes: new Map()
    })
    return newMap
  }

  /**
   * Vota en una encuesta
   */
  static votePoll(
    polls: Map<string, MessagePoll>,
    messageId: string,
    optionIndex: number
  ): Map<string, MessagePoll> {
    const newMap = new Map(polls)
    const poll = newMap.get(messageId)
    
    if (!poll) return polls
    
    const newVotes = new Map(poll.votes)
    const currentVotes = newVotes.get(optionIndex.toString()) || 0
    newVotes.set(optionIndex.toString(), currentVotes + 1)
    
    newMap.set(messageId, {
      ...poll,
      votes: newVotes
    })
    
    return newMap
  }

  /**
   * Obtiene los resultados de una encuesta
   */
  static getPollResults(poll: MessagePoll): Map<number, number> {
    const results = new Map<number, number>()
    poll.votes.forEach((count, optionIndex) => {
      results.set(parseInt(optionIndex), count)
    })
    return results
  }

  /**
   * Obtiene el total de votos
   */
  static getTotalVotes(poll: MessagePoll): number {
    let total = 0
    poll.votes.forEach(count => {
      total += count
    })
    return total
  }
}



