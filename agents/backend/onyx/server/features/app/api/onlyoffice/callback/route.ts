import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/auth";
import { prisma } from "@/lib/prisma";

export async function POST(req: Request) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user?.id) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    const body = await req.json();
    const { status, url, key } = body;

    if (status === 2) { // Document is ready for saving
      // Update the document URL in the database
      await prisma.document.update({
        where: {
          id: key,
        },
        data: {
          content: url,
        },
      });
      
      return NextResponse.json({ error: 0 });
    }

    return NextResponse.json({ error: 0 });
  } catch (error) {
    console.error("[ONLYOFFICE_CALLBACK]", error);
    return new NextResponse("Internal Error", { status: 500 });
  }
}     