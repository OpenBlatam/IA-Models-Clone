import { NextResponse } from 'next/server';
import { auth } from '@/auth';

export async function POST() {
  try {
    const session = await auth();
    
    if (!session?.user) {
      return new NextResponse('Unauthorized', { status: 401 });
    }

    // Aquí podrías actualizar la preferencia del usuario en la base de datos
    // para indicar que ya no quiere usar Google Calendar

    return new NextResponse('Disconnected from Google Calendar', { status: 200 });
  } catch (error) {
    console.error('Error disconnecting from Google Calendar:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
} 