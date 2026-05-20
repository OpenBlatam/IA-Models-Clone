import { NextResponse } from 'next/server';
import { auth } from '@/auth';

export async function POST() {
  try {
    const session = await auth();
    
    if (!session?.user) {
      return new NextResponse('Unauthorized', { status: 401 });
    }

    // Aquí podrías guardar la preferencia del usuario en la base de datos
    // para recordar que quiere usar Google Calendar

    return new NextResponse('Connected to Google Calendar', { status: 200 });
  } catch (error) {
    console.error('Error connecting to Google Calendar:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
} 