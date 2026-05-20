import { AccessToken } from "livekit-server-sdk";

export function generateToken(roomName: string, participantName: string) {
  const at = new AccessToken(
    process.env.LIVEKIT_API_KEY!,
    process.env.LIVEKIT_API_SECRET!,
    {
      identity: participantName,
    }
  );
  at.addGrant({ roomJoin: true, room: roomName });

  return at.toJwt();
} 