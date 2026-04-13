import { NextResponse } from "next/server";

export async function POST() {
  const res = NextResponse.json({ success: true });
  res.cookies.delete("icarus_access_token");
  res.cookies.delete("icarus_id_token");
  res.cookies.delete("icarus_email");
  return res;
}
