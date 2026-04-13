import { NextRequest, NextResponse } from "next/server";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand } from "@aws-sdk/lib-dynamodb";

const ddbClient = new DynamoDBClient({
  region: process.env.CUSTOM_REGION || "us-east-1",
  credentials: {
    accessKeyId: process.env.CUSTOM_ACCESS_KEY_ID!,
    secretAccessKey: process.env.CUSTOM_SECRET_ACCESS_KEY!,
  },
});
const docClient = DynamoDBDocumentClient.from(ddbClient);

const TABLE_NAME = process.env.MAIN_TABLE_NAME!;

export async function POST(req: NextRequest) {
  try {
    const { email, code } = await req.json();
    const { CognitoIdentityProviderClient, ConfirmSignUpCommand } = await import("@aws-sdk/client-cognito-identity-provider");

    const client = new CognitoIdentityProviderClient({
      region: process.env.COGNITO_REGION || "us-east-1",
      credentials: {
        accessKeyId: process.env.CUSTOM_ACCESS_KEY_ID!,
        secretAccessKey: process.env.CUSTOM_SECRET_ACCESS_KEY!,
      },
    });

    await client.send(new ConfirmSignUpCommand({
      ClientId: process.env.COGNITO_CLIENT_ID!,
      Username: email,
      ConfirmationCode: code,
    }));

    // Create user row in the main DynamoDB table
    try {
      await docClient.send(
        new PutCommand({
          TableName: TABLE_NAME,
          Item: {
            userId: `USER#${email}`,
            SK: "META#PROFILE",
            email,
            createdAt: new Date().toISOString(),
            status: "active",
          },
          ConditionExpression: "attribute_not_exists(userId)",
        })
      );
    } catch (ddbErr: any) {
      // ConditionalCheckFailedException is fine — user row already exists
      if (ddbErr.name !== "ConditionalCheckFailedException") {
        console.error("Failed to create user in DynamoDB:", ddbErr);
      }
      else {
        console.error("Failed to create user in DynamoDB:", ddbErr)
      }
    }

    return NextResponse.json({ success: true, message: "Email confirmed! You can now sign in." });
  } catch (err: any) {
    const code = err?.name || "";
    const messages: Record<string, string> = {
      CodeMismatchException: "Invalid verification code.",
      UserNotFoundException: "User not found.",
    };
    return NextResponse.json({ success: false, message: messages[code] || err.message }, { status: 400 });
  }
}
