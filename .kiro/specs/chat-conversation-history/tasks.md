# Implementation Plan: Chat Conversation History

## Overview

Add DynamoDB-backed conversation history and session management to the Project Icarus campaign chatbot. Implementation proceeds infrastructure-first, then backend lambdas, then Next.js API routes, then frontend UI. Each task builds incrementally on the previous, ensuring no orphaned code.

## Tasks

- [x] 1. Provision DynamoDB table and update CDK infrastructure
  - [x] 1.1 Add DynamoDB table and GSI to `infra-stack.ts`
    - Create `icarus-chat-history-{account_id}` table with `chatId` (String) PK and `timestamp` (String) SK
    - Add GSI `userId-index` with `userId` (String) PK and `createdAt` (String) SK
    - Set billing to PAY_PER_REQUEST and removal policy to DESTROY
    - Grant `lambda_role` read/write permissions on the table
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.2 Add session_manager_lambda and API Gateway routes to `infra-stack.ts`
    - Create `session_manager_lambda` with Python 3.13 runtime, handler `session_manager_lambda.lambda_handler`, and `lambda_role`
    - Add `/sessions` resource with GET and DELETE methods integrated with session_manager_lambda
    - Add `/sessions/messages` resource with GET method integrated with session_manager_lambda
    - _Requirements: 4.1, 4.5, 5.3_

  - [x] 1.3 Pass `CHAT_HISTORY_TABLE` environment variable to all relevant lambdas
    - Add `CHAT_HISTORY_TABLE` env var to trigger_chatbot_lambda, chatbot_lambda, check_chatbot_response_lambda, and session_manager_lambda
    - _Requirements: 1.6_

- [x] 2. Checkpoint - Verify CDK infrastructure compiles
  - Ensure CDK TypeScript compiles without errors, ask the user if questions arise.

- [x] 3. Modify trigger_chatbot_lambda to persist user messages
  - [x] 3.1 Update `trigger_chatbot.py` to handle chatId and write to DynamoDB
    - Initialize DynamoDB resource and read `CHAT_HISTORY_TABLE` env var
    - Accept optional `chatId` from request body; generate UUID if absent
    - When chatId is new, write a META record with `timestamp="META"`, `userId`, `createdAt` (ISO 8601), and `title` (first 50 chars of query)
    - Write user message record with `chatId`, `timestamp` (ISO 8601), `userId`, `role="user"`, `content`
    - Pass `chatId` (not `conversation_history`) to chatbot_lambda in the async invoke payload
    - Return `chatId` in the response body
    - _Requirements: 2.1, 2.2, 2.5, 3.1_

  - [ ]* 3.2 Write property test: Trigger lambda persists user messages correctly (Property 1)
    - **Property 1: Trigger lambda persists user messages correctly**
    - Use hypothesis to generate random emails, queries, optional chatIds
    - Verify DDB write structure: valid UUID chatId, matching userId, role="user", content matches query, valid ISO 8601 timestamp, chatId in response
    - **Validates: Requirements 2.1, 2.2**

  - [ ]* 3.3 Write property test: Session metadata record structure (Property 4)
    - **Property 4: Session metadata record has correct structure and truncated title**
    - Use hypothesis to generate random first messages of varying length
    - Verify META record: timestamp="META", userId matches, valid ISO 8601 createdAt, title equals first min(50, len(message)) chars
    - **Validates: Requirements 3.1**

- [x] 4. Modify chatbot_lambda to use DynamoDB history and persist responses
  - [x] 4.1 Update `chatbot_lambda.py` to load history from DynamoDB and write assistant messages
    - Initialize DynamoDB resource and read `CHAT_HISTORY_TABLE` env var
    - Receive `chatId` from the trigger_lambda payload
    - Query DynamoDB for all messages with the given `chatId`, sorted by `timestamp` ascending, excluding META records
    - Use DynamoDB history for the Bedrock `messages` array instead of `conversation_history` from payload
    - After Bedrock response, write assistant message to DynamoDB with `chatId`, `timestamp`, `userId`, `role="assistant"`, `content`
    - Change S3 response key from `{username}/{username}_response.md` to `{username}/{chatId}_response.md`
    - _Requirements: 2.3, 2.4, 10.1_

  - [ ]* 4.2 Write property test: Chatbot lambda persists assistant messages (Property 2)
    - **Property 2: Chatbot lambda persists assistant messages correctly**
    - Use hypothesis to generate random chatIds, emails, mock Bedrock responses
    - Verify DDB write: chatId matches, userId matches, role="assistant", content matches response, valid ISO 8601 timestamp
    - **Validates: Requirements 2.3**

  - [ ]* 4.3 Write property test: History loaded in ascending order (Property 3)
    - **Property 3: Chatbot lambda loads history from DynamoDB in ascending order**
    - Use hypothesis to generate random message sets in mock DDB
    - Verify Bedrock messages array is sorted by timestamp ascending, META excluded
    - **Validates: Requirements 2.4**

  - [ ]* 4.4 Write property test: S3 response key uses chatId (Property 8)
    - **Property 8: S3 response key uses chatId for both write and read**
    - Use hypothesis to generate random usernames and chatIds
    - Verify S3 key equals `{username}/{chatId}_response.md` for both write and read paths
    - **Validates: Requirements 10.1, 10.2**

- [x] 5. Modify check_LLM_response_lambda to use chatId-based S3 keys
  - [x] 5.1 Update `check_LLM_response_lambda.py` to accept chatId and use new S3 key
    - Accept `chatId` query parameter from the request
    - Change S3 key lookup from `{username}/{username}_response.md` to `{username}/{chatId}_response.md`
    - Delete the response file after reading (existing behavior, new key)
    - _Requirements: 10.2, 10.5_

- [x] 6. Implement session_manager_lambda
  - [x] 6.1 Create `session_manager_lambda.py` with list, delete, and get-messages operations
    - Initialize DynamoDB resource and read `CHAT_HISTORY_TABLE` env var
    - GET `/sessions` with `userId` param: query `userId-index` GSI for META records, return array sorted by `createdAt` descending with `chatId`, `title`, `createdAt`
    - DELETE `/sessions` with `chatId` and `userId` params: query all records for `chatId`, batch-delete all, return success; return 404 if chatId not found
    - GET `/sessions/messages` with `chatId` param: query by `chatId`, exclude META, return messages sorted by `timestamp` ascending with `role`, `content`, `timestamp`; return empty array if no messages
    - _Requirements: 3.2, 3.3, 4.2, 4.3, 4.4, 5.1, 5.2, 5.4_

  - [ ]* 6.2 Write property test: List sessions sorted descending (Property 5)
    - **Property 5: List sessions returns correct fields sorted by creation time descending**
    - Use hypothesis to generate random session metadata sets
    - Verify returned array has chatId, title, createdAt per item, sorted by createdAt descending
    - **Validates: Requirements 3.3, 4.2**

  - [ ]* 6.3 Write property test: Delete session removes all records (Property 6)
    - **Property 6: Delete session removes all records for the chatId**
    - Use hypothesis to generate random sessions with varying message counts
    - Verify zero records remain for the chatId after delete
    - **Validates: Requirements 4.3**

  - [ ]* 6.4 Write property test: Get messages excludes META and sorts ascending (Property 7)
    - **Property 7: Get messages returns correctly shaped records sorted ascending, excluding META**
    - Use hypothesis to generate random message sets with META
    - Verify META excluded, each record has role/content/timestamp, sorted ascending
    - **Validates: Requirements 5.1, 5.2**

- [x] 7. Checkpoint - Verify all backend lambdas
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Create and update Next.js API routes for session management
  - [x] 8.1 Create `/api/chat/sessions/route.ts` with GET and DELETE handlers
    - GET: forward `email` query param as `userId` to API Gateway `/sessions`, return session list
    - DELETE: forward `chatId` and `email` query params to API Gateway `/sessions`, return deletion result
    - Handle errors: return `{ status: "FAILED", message }` with appropriate HTTP status
    - _Requirements: 6.1, 6.2, 6.4_

  - [x] 8.2 Create `/api/chat/sessions/messages/route.ts` with GET handler
    - Forward `chatId` query param to API Gateway `/sessions/messages`, return message list
    - Handle errors: return `{ status: "FAILED", message }` with appropriate HTTP status
    - _Requirements: 6.3, 6.4_

  - [x] 8.3 Update `/api/chat/send/route.ts` to include chatId
    - Accept optional `chatId` field in request body, forward to API Gateway `/chat`
    - Return `chatId` from API Gateway response in the response body
    - _Requirements: 7.1, 7.2_

  - [x] 8.4 Update `/api/chat/check/route.ts` to include chatId
    - Accept `chatId` query parameter, forward to API Gateway `/check-response`
    - _Requirements: 10.3_

- [x] 9. Implement Chat Sidebar and update Dashboard UI
  - [x] 9.1 Create ChatSidebar component with session list, delete, and new chat
    - Fetch sessions on mount via `/api/chat/sessions` using current user email
    - Render each session as a clickable item with `title` and relative timestamp (e.g., "2 hours ago")
    - Add delete button per session that calls DELETE `/api/chat/sessions` and removes from list on success
    - Add "New Chat" button that clears messages and resets `chatId` to null
    - _Requirements: 8.1, 8.2, 8.5, 8.6_

  - [x] 9.2 Update Dashboard page to integrate ChatSidebar and chatId state management
    - Add `chatId` state (string | null) and `sessions` state to Dashboard
    - On session click: fetch messages from `/api/chat/sessions/messages`, display in chat panel, set active `chatId`
    - Include `chatId` in every send request body after first message; store returned `chatId` in state
    - Include `chatId` in every poll request to `/api/chat/check`
    - Update "Clear" button to reset messages and `chatId` to null without deleting the session from DynamoDB
    - After clearing, treat next message as start of a new session
    - _Requirements: 7.3, 7.4, 8.3, 8.4, 9.1, 9.2, 9.3, 10.4_

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests use `hypothesis` (Python) and validate universal correctness properties from the design
- Infrastructure (CDK) is implemented first so environment variables and resources are available for lambda development
- Backend lambdas are implemented before API routes and frontend to ensure the full chain works bottom-up
