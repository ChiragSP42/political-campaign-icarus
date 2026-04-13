import { NextResponse } from "next/server";

export async function POST() {
  const res = NextResponse.json({ success: true });
  res.cookies.delete("winflip_access_token");
  res.cookies.delete("winflip_id_token");
  res.cookies.delete("winflip_email");
  return res;
}
