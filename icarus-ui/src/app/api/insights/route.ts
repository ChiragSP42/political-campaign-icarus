import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  try {
    const email = req.nextUrl.searchParams.get("email")!;
    const username = email.split("@")[0];

    const { S3Client, GetObjectCommand } = await import("@aws-sdk/client-s3");
    const { STSClient, GetCallerIdentityCommand } = await import("@aws-sdk/client-sts");

    const creds = {
      accessKeyId: process.env.CUSTOM_ACCESS_KEY_ID!,
      secretAccessKey: process.env.CUSTOM_SECRET_ACCESS_KEY!,
    };
    const region = process.env.CUSTOM_REGION || "us-east-1";

    const sts = new STSClient({ region, credentials: creds });
    const identity = await sts.send(new GetCallerIdentityCommand({}));
    const accountId = identity.Account!;

    const s3 = new S3Client({ region, credentials: creds });
    const res = await s3.send(new GetObjectCommand({
      Bucket: `generated-insights-${accountId}`,
      Key: `${username}/${username}_insights.md`,
    }));

    const content = await res.Body!.transformToString();
    return NextResponse.json({ exists: true, content });
  } catch (err: any) {
    if (err.name === "NoSuchKey") {
      return NextResponse.json({ exists: false, content: null });
    }
    return NextResponse.json({ exists: false, content: null, error: err.message }, { status: 500 });
  }
}
