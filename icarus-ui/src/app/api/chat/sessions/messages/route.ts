import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_ENDPOINT!;

export async function GET(req: NextRequest) {
  try {
    const chatId = req.nextUrl.searchParams.get("chatId")!;
    const res = await fetch(
      `${API}/sessions/messages?chatId=${encodeURIComponent(chatId)}`,
      {
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(100000),
      }
    );
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json(
      { status: "FAILED", message: err.message },
      { status: 500 }
    );
  }
}
