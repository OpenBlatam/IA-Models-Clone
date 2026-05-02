import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/auth";
import { prisma } from "@/lib/db";

export async function DELETE(req: NextRequest) {
  try {
    const session = await auth();
    
    if (!session?.user) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    const currentUser = session.user;
    if (!currentUser?.id) {
      return NextResponse.json({ error: "Invalid user" }, { status: 401 });
    }

    await prisma.user.delete({
      where: {
        id: currentUser.id,
      },
    });

    return NextResponse.json({ message: "User deleted successfully!" }, { status: 200 });
  } catch (error) {
    console.error("Delete user error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
