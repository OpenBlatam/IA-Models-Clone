import { NextResponse } from 'next/server';
import { TIPTAP_AI_CONFIG } from '@/app/dashboard/mkt-ia/background-remover/config/ai';

export async function GET() {
  try {
    const response = await fetch(`${TIPTAP_AI_CONFIG.baseUrl}/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${TIPTAP_AI_CONFIG.secret}`,
      },
      body: JSON.stringify({
        appId: TIPTAP_AI_CONFIG.appId,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to generate token');
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error generating Tiptap AI token:', error);
    return NextResponse.json(
      { error: 'Failed to generate token' },
      { status: 500 }
    );
  }
} 