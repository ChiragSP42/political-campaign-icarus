import { NextRequest, NextResponse } from "next/server";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, GetCommand } from "@aws-sdk/lib-dynamodb";

const ddbClient = new DynamoDBClient({
  region: process.env.CUSTOM_REGION || "us-east-1",
  credentials: {
    accessKeyId: process.env.CUSTOM_ACCESS_KEY_ID!,
    secretAccessKey: process.env.CUSTOM_SECRET_ACCESS_KEY!,
  },
});
const docClient = DynamoDBDocumentClient.from(ddbClient);

const TABLE_NAME = process.env.MAIN_TABLE_NAME!;

export async function GET(req: NextRequest) {
  try {
    const email = req.nextUrl.searchParams.get("email");
    if (!email) {
      return NextResponse.json({ exists: false, content: null, error: "Email is required" }, { status: 400 });
    }

    const result = await docClient.send(new GetCommand({
      TableName: TABLE_NAME,
      Key: {
        userId: `USER#${email}`,
        SK: "INSIGHTS",
      },
    }));

    if (result.Item && result.Item.insights) {
      return NextResponse.json({ exists: true, content: result.Item.insights });
    }

    return NextResponse.json({ exists: false, content: null });
  } catch (err: any) {
    console.error("Error fetching insights:", err.name, err.message);
    return NextResponse.json({ exists: false, content: null, error: err.message }, { status: 500 });
  }
}
