import { useMutation } from '@tanstack/react-query';
import { chatbotApi } from '@/services/api';

export function useChatbotMessage() {
  return useMutation({
    mutationFn: chatbotApi.sendChatbotMessage,
  });
}

export function useStartChatbotSession() {
  return useMutation({
    mutationFn: chatbotApi.startChatbotSession,
  });
}

