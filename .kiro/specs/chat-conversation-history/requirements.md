# Requirements Document

## Introduction

The WinFlip / Project Icarus campaign chatbot currently has no server-side conversation history. The frontend holds messages in React state and passes the full `conversation_history` array to the backend on every request. When the page reloads, all history is lost. This feature adds a DynamoDB-backed conversation history store with full session management — creating, listing, selecting, continuing, and deleting chat sessions — so users can maintain persistent, multi-session conversations with the campaign advisor.

## Glossary

- **Chat_Session**: A single conversation thread between a user and the campaign advisor chatbot, identified by a unique `chatId`.
- **Conversation_History_Table**: The DynamoDB table that stores all chat messages, partitioned by `chatId` with a sort key of `timestamp`.
- **Chat_Sessions_Table**: The DynamoDB table (or GSI) that indexes chat sessions by `userId` for listing and lookup.
- **Session_Manager_Lambda**: The AWS Lambda function responsible for CRUD operations on chat sessions (create, list, delete, get).
- **Chatbot_Lambda**: The existing Lambda function (`chatbot_lambda.py`) that processes user queries via Bedrock and writes responses to S3.
- **Trigger_Lambda**: The existing Lambda function (`trigger_chatbot.py`) that receives API Gateway requests and invokes Chatbot_Lambda asynchronously.
- **Check_Response_Lambda**: The existing Lambda function (`check_LLM_response_lambda.py`) that polls S3 for the chatbot response.
- **Dashboard_UI**: The Next.js React page at `(authenticated)/dashboard/page.tsx` that renders the chat interface.
- **Chat_Sidebar**: The UI component within Dashboard_UI that displays the list of previous chat sessions for selection.
- **API_Gateway**: The existing AWS API Gateway (`icarus-api`) that routes HTTP requests to Lambda functions.
- **userId**: The user's email address, sourced from the auth context (`auth.email`).
- **chatId**: A unique identifier (UUID) for each Chat_Session.

## Requirements

### Requirement 1: DynamoDB Table Provisioning

**User Story:** As a platform operator, I want a DynamoDB table provisioned via CDK, so that conversation history is stored durably and can be queried by chat session or user.

#### Acceptance Criteria

1. THE infra-stack SHALL provision a DynamoDB table named `icarus-chat-history-{account_id}` with `chatId` (String) as the partition key and `timestamp` (String, ISO 8601) as the sort key.
2. THE infra-stack SHALL provision a Global Secondary Index named `userId-index` on the Conversation_History_Table with `userId` (String) as the partition key and `createdAt` (String, ISO 8601) as the sort key.
3. THE infra-stack SHALL set the Conversation_History_Table billing mode to PAY_PER_REQUEST.
4. THE infra-stack SHALL set the Conversation_History_Table removal policy to DESTROY for development environments.
5. THE infra-stack SHALL grant the lambda_role read and write permissions to the Conversation_History_Table.
6. THE infra-stack SHALL pass the Conversation_History_Table name as an environment variable `CHAT_HISTORY_TABLE` to Chatbot_Lambda, Trigger_Lambda, Check_Response_Lambda, and Session_Manager_Lambda.

### Requirement 2: Persist Messages to DynamoDB

**User Story:** As a user, I want my chat messages and assistant responses saved automatically, so that I can close the browser and return to my conversation later.

#### Acceptance Criteria

1. WHEN Trigger_Lambda receives a chat request with a `chatId`, THE Trigger_Lambda SHALL write the user message to the Conversation_History_Table with the `chatId`, `userId`, `timestamp`, `role` set to "user", and `content` containing the message text.
2. WHEN Trigger_Lambda receives a chat request without a `chatId`, THE Trigger_Lambda SHALL generate a new UUID as the `chatId`, write the user message to the Conversation_History_Table, and return the generated `chatId` in the response.
3. WHEN Chatbot_Lambda finishes generating a response, THE Chatbot_Lambda SHALL write the assistant message to the Conversation_History_Table with the same `chatId`, `userId`, `timestamp`, `role` set to "assistant", and `content` containing the response text.
4. WHEN Chatbot_Lambda loads conversation history for a Bedrock call, THE Chatbot_Lambda SHALL query the Conversation_History_Table for all messages with the given `chatId`, sorted by `timestamp` in ascending order, instead of relying on the `conversation_history` field from the request payload.
5. THE Trigger_Lambda SHALL continue to pass the `chatId` to Chatbot_Lambda in the async invocation payload.

### Requirement 3: Session Metadata Storage

**User Story:** As a user, I want each chat session to have a title and creation timestamp, so that I can identify and distinguish between my conversations.

#### Acceptance Criteria

1. WHEN a new Chat_Session is created (first message with no existing `chatId`), THE Trigger_Lambda SHALL write a metadata record to the Conversation_History_Table with `chatId` as the partition key, `timestamp` set to "META", `userId`, `createdAt` (ISO 8601), and `title` derived from the first 50 characters of the first user message.
2. THE Session_Manager_Lambda SHALL use the `userId-index` GSI to query metadata records (where sort key equals "META") for a given `userId`.
3. WHEN a metadata record is queried, THE Session_Manager_Lambda SHALL return `chatId`, `title`, and `createdAt` for each Chat_Session.

### Requirement 4: Session Management Lambda

**User Story:** As a user, I want to create new chat sessions, list my previous sessions, and delete sessions I no longer need, so that I can organize my campaign advisor conversations.

#### Acceptance Criteria

1. THE infra-stack SHALL provision a Session_Manager_Lambda with handler `session_manager_lambda.lambda_handler`, Python 3.13 runtime, and access to the Conversation_History_Table.
2. WHEN Session_Manager_Lambda receives a GET request with `userId` query parameter on the `/sessions` route, THE Session_Manager_Lambda SHALL return a JSON array of all Chat_Sessions for that user, sorted by `createdAt` descending, each containing `chatId`, `title`, and `createdAt`.
3. WHEN Session_Manager_Lambda receives a DELETE request with `chatId` and `userId` query parameters on the `/sessions` route, THE Session_Manager_Lambda SHALL delete all records in the Conversation_History_Table with the specified `chatId` (metadata and messages) and return a success confirmation.
4. IF Session_Manager_Lambda receives a DELETE request for a `chatId` that does not exist, THEN THE Session_Manager_Lambda SHALL return a 404 status with an error message indicating the session was not found.
5. THE infra-stack SHALL add a `/sessions` resource to API_Gateway with GET and DELETE methods integrated with Session_Manager_Lambda.

### Requirement 5: Load Conversation on Session Selection

**User Story:** As a user, I want to select a previous chat session and see its full message history, so that I can continue the conversation where I left off.

#### Acceptance Criteria

1. WHEN Session_Manager_Lambda receives a GET request with `chatId` query parameter on the `/sessions/messages` route, THE Session_Manager_Lambda SHALL return all messages for that `chatId` sorted by `timestamp` ascending, excluding the metadata record (sort key "META").
2. THE Session_Manager_Lambda SHALL return each message with `role`, `content`, and `timestamp` fields.
3. THE infra-stack SHALL add a `/sessions/messages` resource to API_Gateway with a GET method integrated with Session_Manager_Lambda.
4. IF Session_Manager_Lambda receives a GET request for a `chatId` with no messages, THEN THE Session_Manager_Lambda SHALL return an empty array with a 200 status.

### Requirement 6: API Route Layer for Session Management

**User Story:** As a frontend developer, I want Next.js API routes that proxy session management requests to API Gateway, so that the frontend can manage sessions through the same pattern used for chat.

#### Acceptance Criteria

1. THE Next.js application SHALL expose a GET route at `/api/chat/sessions` that forwards the `email` query parameter as `userId` to the API_Gateway `/sessions` endpoint and returns the session list.
2. THE Next.js application SHALL expose a DELETE route at `/api/chat/sessions` that forwards `chatId` and `email` query parameters to the API_Gateway `/sessions` endpoint and returns the deletion result.
3. THE Next.js application SHALL expose a GET route at `/api/chat/sessions/messages` that forwards the `chatId` query parameter to the API_Gateway `/sessions/messages` endpoint and returns the message list.
4. IF any API_Gateway call fails, THEN THE Next.js API route SHALL return a JSON response with `status` set to "FAILED" and the error `message`, with an appropriate HTTP status code.

### Requirement 7: Update Send Route to Include chatId

**User Story:** As a frontend developer, I want the chat send flow to support a `chatId` parameter, so that messages are associated with the correct session.

#### Acceptance Criteria

1. THE Next.js `/api/chat/send` route SHALL accept an optional `chatId` field in the request body and forward it to the API_Gateway `/chat` endpoint.
2. THE Next.js `/api/chat/send` route SHALL return the `chatId` from the API_Gateway response (either the one sent or the newly generated one) in its response body.
3. THE Dashboard_UI SHALL include the current `chatId` in every send request after the first message of a session.
4. WHEN Dashboard_UI receives a response with a new `chatId`, THE Dashboard_UI SHALL store that `chatId` in component state for subsequent messages in the same session.

### Requirement 8: Chat Sidebar UI

**User Story:** As a user, I want a sidebar in the chat panel that shows my previous conversations, so that I can quickly switch between sessions.

#### Acceptance Criteria

1. WHEN Dashboard_UI loads, THE Chat_Sidebar SHALL fetch the list of Chat_Sessions for the current user from `/api/chat/sessions`.
2. THE Chat_Sidebar SHALL display each Chat_Session as a clickable item showing the `title` and a relative timestamp (e.g., "2 hours ago").
3. WHEN a user clicks a Chat_Session item in Chat_Sidebar, THE Dashboard_UI SHALL fetch the full message history from `/api/chat/sessions/messages` with the selected `chatId` and display the messages in the chat panel.
4. WHEN a user clicks a Chat_Session item, THE Dashboard_UI SHALL set the active `chatId` to the selected session so subsequent messages are appended to that session.
5. THE Chat_Sidebar SHALL provide a delete button on each Chat_Session item that calls `/api/chat/sessions` with DELETE method and removes the session from the list upon success.
6. THE Chat_Sidebar SHALL provide a "New Chat" button that clears the current messages, resets the active `chatId` to null, and starts a fresh session.

### Requirement 9: Clear Chat Session

**User Story:** As a user, I want to clear the current chat display without deleting the session from history, so that I can start fresh visually while keeping the record.

#### Acceptance Criteria

1. WHEN the user clicks the existing "Clear" button in Dashboard_UI, THE Dashboard_UI SHALL clear the displayed messages from the chat panel and reset the active `chatId` to null.
2. THE Dashboard_UI SHALL NOT delete the Chat_Session from the Conversation_History_Table when the "Clear" button is clicked.
3. AFTER clearing, WHEN the user sends a new message, THE Dashboard_UI SHALL treat the message as the start of a new Chat_Session.

### Requirement 10: Update Check Response Flow with chatId

**User Story:** As a developer, I want the response polling flow to use `chatId` for response file keying, so that concurrent sessions for the same user do not overwrite each other's responses.

#### Acceptance Criteria

1. WHEN Chatbot_Lambda writes the response to S3, THE Chatbot_Lambda SHALL use the S3 key `{username}/{chatId}_response.md` instead of `{username}/{username}_response.md`.
2. WHEN Check_Response_Lambda polls for a response, THE Check_Response_Lambda SHALL accept a `chatId` query parameter and look for the S3 key `{username}/{chatId}_response.md`.
3. THE Next.js `/api/chat/check` route SHALL accept and forward a `chatId` query parameter to the API_Gateway `/check-response` endpoint.
4. THE Dashboard_UI SHALL include the active `chatId` in every poll request to `/api/chat/check`.
5. WHEN Check_Response_Lambda retrieves and deletes the response file, THE Trigger_Lambda SHALL also write the assistant response to the Conversation_History_Table (or delegate this to Chatbot_Lambda as defined in Requirement 2, Criterion 3).
