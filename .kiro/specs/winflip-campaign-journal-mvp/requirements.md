# Requirements Document

## Introduction

WinFlip Phase 2 MVP is a full-stack campaign intelligence platform for down-ballot Virginia candidates. It evolves the Phase 1 "Project Icarus" proof-of-concept (Streamlit frontend, Lambda/Bedrock backend, S3 storage, Cognito auth) into a production-grade system with a WinFlip-branded React/Next.js frontend on AWS Amplify, DynamoDB-backed structured data, an enhanced onboarding questionnaire covering all Virginia offices, an improved insights engine with token-limit handling, an AI-assisted campaign journal, archetype-driven tone controls, Stripe-based free/paid tiers, an enhanced chatbot with conversation threading, CI/CD pipelines, and CloudWatch-based logging and analytics. All code resides within the "WinFlip Phase 2 - MVP" folder. The project uses `uv` as the Python package installer. Comet integration is explicitly out of scope.

## Glossary

- **WinFlip_Platform**: The overall Phase 2 MVP system encompassing frontend, backend, infrastructure, and AI subsystems.
- **Candidate**: An authenticated WinFlip user (via Amazon Cognito) running for a Virginia legislative or local office.
- **Frontend**: The WinFlip-branded responsive web application built with React/Next.js and deployed via AWS Amplify, replacing the Phase 1 Streamlit interface.
- **Backend**: The set of AWS Lambda functions, API Gateway endpoints, DynamoDB tables, S3 buckets, and Amazon Bedrock integrations that power the platform.
- **Onboarding_Questionnaire**: The multi-step intake form that captures candidate basic info, demographics, address, race selection, background, and communication style to personalize the platform experience.
- **Insights_Engine**: The AI subsystem (powered by Amazon Bedrock) that loads historical election data, processes questionnaire answers, and generates strategic campaign insights for a Candidate.
- **Campaign_Journal**: The core module that stores, organizes, and surfaces journal entries, insights, actions, and generated messaging for an authenticated Candidate.
- **Journal_Entry**: A timestamped record created from a manual note, event note, or chatbot conversation, automatically tagged with location context, topic, and event type.
- **Entry_Tagger**: The AI subsystem (powered by Amazon Bedrock) that automatically assigns Location_Tag, Topic_Tag, and Event_Type_Tag values to each Journal_Entry upon creation.
- **Location_Tag**: A contextual label (not geo-referenced) describing the setting of a Journal_Entry (e.g., "Town Hall", "Neighborhood Canvass").
- **Topic_Tag**: A subject-matter label assigned to a Journal_Entry (e.g., "education", "housing", "taxes", "affordability").
- **Event_Type_Tag**: A campaign-activity label assigned to a Journal_Entry (e.g., "meeting", "fundraising", "canvassing", "town hall").
- **Insight**: An AI-generated observation derived from one or more Journal_Entry records that surfaces a pattern, trend, or strategic finding for the Candidate.
- **Validation_Status**: The state of an Insight as set by the Candidate: one of "pending", "confirmed", "edited", or "dismissed".
- **Action_Generator**: The AI subsystem that produces suggested Action items from confirmed or edited Insight records.
- **Action**: A concrete next-step recommendation tied to a confirmed Insight (e.g., "Adjust messaging on affordability", "Attend local housing forum").
- **Action_Status**: The completion state of an Action: one of "pending", "in_progress", or "completed".
- **Messaging_Generator**: The AI subsystem that produces campaign messaging artifacts (talking points, social media content, messaging angles) from confirmed Insights and completed or in-progress Actions.
- **Chatbot**: The interactive AI campaign advisor that maintains conversation threading, loads candidate context (questionnaire, insights, journal), and provides strategic guidance.
- **Chatbot_Session**: An interactive conversation between the Candidate and the Chatbot, from which Journal_Entry records can be automatically created.
- **Archetype_Tone**: A preset communication intensity scale ("Cautious", "Calm", "Balanced", "Engaged", "Fired-up") that controls the tone of AI-generated messaging and chatbot responses.
- **Stripe_Payment**: The Stripe Checkout integration that manages free vs. paid subscription tiers for Candidates.
- **Subscription_Tier**: The Candidate's current plan level: "free" or "paid".
- **CI_CD_Pipeline**: The automated build, test, and deployment pipeline for frontend and backend code with environment promotion (dev → prod).
- **DynamoDB_Store**: The Amazon DynamoDB tables used for structured data storage (questionnaires, journal entries, insights, actions, sessions, subscriptions), replacing Phase 1 S3-only storage for structured data.
- **Election_Data**: Historical precinct-level election results for Virginia offices stored in S3, covering Senate, House of Delegates, town/city councils, boards of supervisors, school boards, and other local offices.

## Requirements

---

### Requirement 1: Frontend Redesign — WinFlip-Branded Web Application

**User Story:** As a Candidate, I want a modern, responsive, WinFlip-branded web interface so that I can access all platform features from any device with a professional experience that replaces the Phase 1 Streamlit prototype.

#### Acceptance Criteria

1. THE Frontend SHALL be built with React/Next.js and deployed via AWS Amplify.
2. THE Frontend SHALL provide responsive layouts that render correctly on desktop, tablet, and mobile screen widths.
3. THE Frontend SHALL include dedicated views for authentication, onboarding questionnaire, campaign insights, campaign journal, chatbot, account/subscription management, and archetype tone controls.
4. THE Frontend SHALL apply WinFlip branding (logo, color palette, typography) consistently across all views.
5. WHEN a Candidate navigates between views, THE Frontend SHALL maintain session state so that in-progress work is not lost.
6. THE Frontend SHALL communicate with the Backend exclusively through the API Gateway REST endpoints.
7. WHEN the Frontend receives an HTTP error response from the Backend, THE Frontend SHALL display a user-friendly error message describing the issue without exposing internal details.

---

### Requirement 2: Backend Refactoring — DynamoDB and Lambda Improvements

**User Story:** As a developer, I want the backend to use DynamoDB for structured data and have refactored Lambda functions so that the system supports new races, demographics, and scales beyond the Phase 1 S3-only storage pattern.

#### Acceptance Criteria

1. THE Backend SHALL store questionnaire responses, journal entries, insights, actions, chatbot sessions, and subscription records in DynamoDB_Store tables with the Candidate identifier as the partition key.
2. THE Backend SHALL continue to use S3 for storing election data files, Bedrock prompts, and generated insight markdown documents.
3. WHEN a Lambda function writes structured data, THE Backend SHALL write to DynamoDB_Store instead of S3 JSON files.
4. THE Backend SHALL use `uv` as the Python package installer for all Lambda function dependency management.
5. THE Backend SHALL organize all Phase 2 code within the "WinFlip Phase 2 - MVP" folder.
6. WHEN the Backend receives an API request, THE Backend SHALL validate the request payload against the expected schema before processing.
7. IF a request payload fails schema validation, THEN THE Backend SHALL return an HTTP 400 response with a descriptive error message.

---

### Requirement 3: Onboarding Questionnaire Enhancement

**User Story:** As a Candidate, I want an extended onboarding questionnaire that captures my demographics, address, and lets me select from all Virginia office types so that the platform can generate insights tailored to my specific race and district.

#### Acceptance Criteria

1. THE Onboarding_Questionnaire SHALL include steps for: basic information (name, email), demographics capture (age range, gender, ethnicity — all optional), address capture (Virginia locality), office and race selection, background and profile questions, and communication style and archetype questions.
2. THE Onboarding_Questionnaire SHALL support race selection across all Virginia office types: Senate of Virginia, House of Delegates, town councils, city councils, boards of supervisors, school boards, mayor, sheriff, commonwealth's attorney, commissioner of the revenue, clerk of court, treasurer, and soil and water conservation director.
3. WHEN a Candidate selects a statewide office (Governor, Lieutenant Governor, Attorney General, U.S. Senate), THE Onboarding_Questionnaire SHALL set the district to "Statewide" automatically.
4. WHEN a Candidate selects a district-based office, THE Onboarding_Questionnaire SHALL present a district selection input appropriate for that office type.
5. THE Onboarding_Questionnaire SHALL persist completed questionnaire data to DynamoDB_Store associated with the Candidate identifier.
6. WHEN a Candidate has previously completed the questionnaire, THE Onboarding_Questionnaire SHALL pre-populate fields with saved answers and allow the Candidate to update responses.
7. THE Onboarding_Questionnaire SHALL retain the existing Phase 1 background questions (military, public safety, union, business owner, public service, faith community, first-time candidate) and communication style archetype questions.

---

### Requirement 4: Insights Engine Enhancement — Token Limits and Extended Race Coverage

**User Story:** As a Candidate, I want the insights engine to handle large districts with many precincts and support all Virginia office types so that I receive comprehensive strategic analysis regardless of my race or district size.

#### Acceptance Criteria

1. THE Insights_Engine SHALL support generating insights for all Virginia office types defined in the Election_Data (Senate of Virginia, House of Delegates, town councils, city councils, boards of supervisors, school boards, mayor, sheriff, commonwealth's attorney, and all other offices in election_cycles.json).
2. WHEN the combined election data and prompt exceed the Bedrock model token limit, THE Insights_Engine SHALL split the data into chunks and process them in parallel or sequentially, then summarize the partial results into a unified insight document.
3. THE Insights_Engine SHALL load up to 5 years of historical election data for the Candidate's office and cross-reference data from other Virginia offices in the same district precincts, consistent with Phase 1 behavior.
4. WHEN the Insights_Engine completes processing, THE Insights_Engine SHALL store the generated insights document in S3 and record a reference in DynamoDB_Store linked to the Candidate identifier.
5. IF the Insights_Engine encounters missing election data for a requested office/year combination, THEN THE Insights_Engine SHALL skip that data source, log a warning, and continue processing with available data.
6. THE Insights_Engine SHALL include in each generated insight document: precinct-level forensics, demographic intelligence, turnout scenarios, competitive positioning, and tactical recommendations, consistent with Phase 1 output structure.
7. WHEN a Candidate updates their questionnaire, THE Insights_Engine SHALL regenerate insights using the updated questionnaire data.

---

### Requirement 5: Manual Journal Entry Creation

**User Story:** As a Candidate, I want to quickly type a note about an event, meeting, or thought so that the Campaign_Journal captures it as a structured Journal_Entry without extra effort.

#### Acceptance Criteria

1. WHEN a Candidate submits a text note through the journal input interface, THE Campaign_Journal SHALL create a new Journal_Entry containing the note text, the Candidate identifier, and a UTC timestamp.
2. WHEN a Journal_Entry is created from a manual note, THE Entry_Tagger SHALL assign at least one Location_Tag, one Topic_Tag, and one Event_Type_Tag to the Journal_Entry within 5 seconds of creation.
3. IF the Entry_Tagger cannot determine a tag value with sufficient confidence, THEN THE Entry_Tagger SHALL assign the tag value "unclassified" for that tag category.
4. WHEN a Candidate views a newly created Journal_Entry, THE Campaign_Journal SHALL display the assigned Location_Tag, Topic_Tag, and Event_Type_Tag values alongside the entry text.
5. THE Campaign_Journal SHALL persist each Journal_Entry to DynamoDB_Store so that the entry is retrievable across sessions.

---

### Requirement 6: Automatic Entry Creation from Chatbot Conversations

**User Story:** As a Candidate, I want my chatbot conversations to automatically become journal entries so that strategy discussions and AI advice are captured without manual re-entry.

#### Acceptance Criteria

1. WHEN a Chatbot_Session ends or the Candidate explicitly saves the conversation, THE Campaign_Journal SHALL create a new Journal_Entry containing a summary of the Chatbot_Session content.
2. WHEN a Journal_Entry is created from a Chatbot_Session, THE Entry_Tagger SHALL assign Location_Tag, Topic_Tag, and Event_Type_Tag values to the Journal_Entry using the same tagging logic as manual entries.
3. THE Campaign_Journal SHALL store a reference linking the Journal_Entry back to the originating Chatbot_Session so that the Candidate can navigate from the entry to the full conversation.
4. IF the Chatbot_Session content is empty or contains only system messages, THEN THE Campaign_Journal SHALL skip Journal_Entry creation for that session.

---

### Requirement 7: Journal Entry Search and Filtering

**User Story:** As a Candidate, I want to search and filter my journal entries by tag, date, or keyword so that I can quickly find relevant past notes and conversations.

#### Acceptance Criteria

1. WHEN a Candidate enters a keyword in the journal search field, THE Campaign_Journal SHALL return all Journal_Entry records whose text content contains the keyword, ordered by most recent first.
2. WHEN a Candidate selects one or more Topic_Tag, Location_Tag, or Event_Type_Tag filters, THE Campaign_Journal SHALL return only Journal_Entry records matching all selected tag filters.
3. WHEN a Candidate specifies a date range filter, THE Campaign_Journal SHALL return only Journal_Entry records with timestamps within the specified range (inclusive).
4. THE Campaign_Journal SHALL support combining keyword search with tag filters and date range filters in a single query.
5. WHEN no Journal_Entry records match the search or filter criteria, THE Campaign_Journal SHALL display a message indicating zero results were found.

---

### Requirement 8: AI-Driven Insight Generation from Journal

**User Story:** As a Candidate, I want the system to automatically analyze my journal entries and surface patterns and strategic insights so that I can make informed campaign decisions without manually reviewing every note.

#### Acceptance Criteria

1. WHEN the Campaign_Journal Insight engine processes a collection of Journal_Entry records, THE Campaign_Journal SHALL identify recurring themes, sentiment shifts, or strategic patterns across entries and generate one or more Insight objects.
2. THE Campaign_Journal SHALL include in each Insight a human-readable summary, a list of references to the source Journal_Entry records, and a "pending" Validation_Status.
3. WHEN a new Insight is generated, THE Campaign_Journal SHALL present the Insight to the Candidate in a dedicated insights review area.
4. THE Campaign_Journal SHALL process new Journal_Entry records and generate updated Insights at least once per Candidate login session.
5. IF the Campaign_Journal receives fewer than 2 Journal_Entry records for analysis, THEN THE Campaign_Journal SHALL skip insight generation and inform the Candidate that more entries are needed.

---

### Requirement 9: Insight Validation (Human in the Loop)

**User Story:** As a Candidate, I want to confirm, edit, or dismiss AI-generated insights so that the system stays grounded in my real campaign experience and improves over time.

#### Acceptance Criteria

1. WHEN a Candidate views a pending Insight, THE Campaign_Journal SHALL provide options to confirm, edit, or dismiss the Insight.
2. WHEN a Candidate confirms an Insight, THE Campaign_Journal SHALL update the Insight Validation_Status to "confirmed" and make the Insight available for action generation.
3. WHEN a Candidate edits an Insight, THE Campaign_Journal SHALL save the edited text, update the Validation_Status to "edited", and make the Insight available for action generation.
4. WHEN a Candidate dismisses an Insight, THE Campaign_Journal SHALL update the Validation_Status to "dismissed" and exclude the Insight from action generation and messaging generation.
5. THE Campaign_Journal SHALL persist all Validation_Status changes to DynamoDB_Store so that the Candidate can review past validation decisions.

---

### Requirement 10: Action Generation from Confirmed Insights

**User Story:** As a Candidate, I want the system to suggest concrete next steps from confirmed insights so that I can turn strategic observations into real campaign activity.

#### Acceptance Criteria

1. WHEN an Insight Validation_Status is set to "confirmed" or "edited", THE Action_Generator SHALL generate one or more Action items tied to that Insight.
2. THE Action_Generator SHALL include in each Action a description of the suggested next step and a reference to the source Insight.
3. WHEN a Candidate views generated Actions, THE Campaign_Journal SHALL display each Action with its description, source Insight reference, and current Action_Status ("pending" by default).
4. WHEN a Candidate marks an Action as "in_progress" or "completed", THE Campaign_Journal SHALL update the Action_Status accordingly and persist the change to DynamoDB_Store.
5. IF the Action_Generator cannot produce a meaningful Action from an Insight, THEN THE Action_Generator SHALL skip action creation for that Insight and log the reason internally.

---

### Requirement 11: Messaging and Content Generation

**User Story:** As a Candidate, I want the system to generate talking points, social media content, and messaging angles from my confirmed insights and actions so that I can communicate effectively without hiring a full-time communications staff.

#### Acceptance Criteria

1. WHEN a Candidate requests messaging generation for a confirmed or edited Insight, THE Messaging_Generator SHALL produce at least one talking point, one social media post draft, and one messaging angle based on the Insight and any associated Actions.
2. THE Messaging_Generator SHALL apply the Candidate's current Archetype_Tone setting when generating all messaging content.
3. THE Messaging_Generator SHALL return generated messaging content within 10 seconds of the Candidate request.
4. WHEN the Messaging_Generator produces content, THE Campaign_Journal SHALL display the generated talking points, social media drafts, and messaging angles in a structured view.
5. THE Campaign_Journal SHALL allow the Candidate to copy, edit, or save generated messaging content for later use.
6. IF the source Insight lacks sufficient context for meaningful messaging, THEN THE Messaging_Generator SHALL return a message indicating that more context is needed and suggest the Candidate add related journal entries.

---

### Requirement 12: Archetype and Tone Controls

**User Story:** As a Candidate, I want to adjust the tone of AI-generated messaging and chatbot responses using a preset scale so that the platform's communication style matches my campaign personality.

#### Acceptance Criteria

1. THE WinFlip_Platform SHALL provide a tone control with five preset levels: "Cautious", "Calm", "Balanced", "Engaged", and "Fired-up".
2. THE WinFlip_Platform SHALL default the Archetype_Tone to "Balanced" for new Candidates.
3. WHEN a Candidate selects a different Archetype_Tone level, THE WinFlip_Platform SHALL persist the selection to DynamoDB_Store and apply the new tone to all subsequent AI-generated content.
4. WHEN the Messaging_Generator generates talking points, social media content, or messaging angles, THE Messaging_Generator SHALL adjust vocabulary, sentence structure, and emotional intensity to match the Candidate's current Archetype_Tone setting.
5. WHEN the Chatbot generates a response, THE Chatbot SHALL adjust its advisory tone to reflect the Candidate's current Archetype_Tone setting.
6. THE WinFlip_Platform SHALL derive the initial Archetype_Tone recommendation from the Candidate's communication style answers in the Onboarding_Questionnaire.

---

### Requirement 13: Chatbot Enhancement — Conversation Threading and Performance

**User Story:** As a Candidate, I want the chatbot to support threaded conversations with faster response times and awareness of my journal and insights so that I get contextual strategic advice without long waits.

#### Acceptance Criteria

1. THE Chatbot SHALL maintain conversation threads so that a Candidate can have multiple distinct conversation topics and return to previous threads.
2. THE Chatbot SHALL store each Chatbot_Session (thread) in DynamoDB_Store with the full message history, Candidate identifier, creation timestamp, and last-updated timestamp.
3. WHEN a Candidate sends a message, THE Chatbot SHALL return a response within 30 seconds, replacing the Phase 1 async S3 polling pattern with a synchronous or streaming response mechanism.
4. THE Chatbot SHALL load the Candidate's generated insights, questionnaire context, and recent Journal_Entry summaries as context for each response.
5. WHEN a Candidate starts a new conversation thread, THE Chatbot SHALL create a new Chatbot_Session record and present an empty conversation view.
6. THE Chatbot SHALL maintain a chronological list of all Chatbot_Session records for each Candidate, displaying the date, a brief summary, and associated tags for each session.
7. THE Chatbot SHALL support keyword search across Chatbot_Session content, returning sessions that contain the search term.

---

### Requirement 14: Stripe Payment Integration — Free vs. Paid Tiers

**User Story:** As a Candidate, I want to upgrade from a free tier to a paid tier via Stripe so that I can unlock premium features, and as the WinFlip team, we want a basic payment infrastructure for monetization.

#### Acceptance Criteria

1. THE WinFlip_Platform SHALL support two Subscription_Tier levels: "free" and "paid".
2. THE WinFlip_Platform SHALL default new Candidates to the "free" Subscription_Tier upon account creation.
3. WHEN a Candidate initiates an upgrade to the "paid" tier, THE WinFlip_Platform SHALL redirect the Candidate to a Stripe Checkout session configured in sandbox mode.
4. WHEN Stripe sends a successful payment webhook, THE Backend SHALL update the Candidate's Subscription_Tier to "paid" in DynamoDB_Store.
5. WHEN a Candidate's Subscription_Tier is "free", THE WinFlip_Platform SHALL restrict access to paid-tier features as defined by the product configuration.
6. THE Backend SHALL verify Stripe webhook signatures to prevent unauthorized subscription changes.
7. THE WinFlip_Platform SHALL display the Candidate's current Subscription_Tier and provide an option to manage the subscription in the account settings view.
8. IF a Stripe Checkout session is abandoned or payment fails, THEN THE WinFlip_Platform SHALL keep the Candidate on the "free" tier and display a message indicating the payment was not completed.

---

### Requirement 15: Authentication and Data Isolation

**User Story:** As a Candidate, I want my data to be private and accessible only to me so that my campaign strategy, journal, and personal information remain confidential.

#### Acceptance Criteria

1. THE WinFlip_Platform SHALL require Amazon Cognito authentication before granting access to any platform data or functionality.
2. THE WinFlip_Platform SHALL associate all Journal_Entry, Insight, Action, Chatbot_Session, questionnaire, and subscription records with the authenticated Candidate identifier.
3. WHEN a Candidate requests data, THE WinFlip_Platform SHALL return only records associated with that Candidate identifier.
4. IF an unauthenticated request is made to any Backend API endpoint, THEN THE Backend SHALL return an HTTP 401 response and provide no data.
5. THE WinFlip_Platform SHALL enforce HTTPS for all communication between the Frontend and Backend.
6. THE Backend SHALL validate the Cognito JWT token on every API request before processing.

---

### Requirement 16: Journal Entry Tag Editing

**User Story:** As a Candidate, I want to edit the AI-assigned tags on my journal entries so that I can correct misclassifications and keep my journal organized accurately.

#### Acceptance Criteria

1. WHEN a Candidate views a Journal_Entry, THE Campaign_Journal SHALL provide an option to edit the assigned Location_Tag, Topic_Tag, and Event_Type_Tag values.
2. WHEN a Candidate submits updated tag values for a Journal_Entry, THE Campaign_Journal SHALL persist the new tag values to DynamoDB_Store and replace the previous values.
3. THE Campaign_Journal SHALL reflect updated tag values in all subsequent search and filter results immediately after the edit is saved.

---

### Requirement 17: Daily Loop and Entry Prompt

**User Story:** As a Candidate, I want the system to encourage daily journaling so that I build a consistent habit of capturing campaign activity and the AI has enough data to generate useful insights.

#### Acceptance Criteria

1. WHEN a Candidate logs in and has not created a Journal_Entry in the current calendar day, THE Campaign_Journal SHALL display a prompt encouraging the Candidate to log a note or start a chat.
2. WHEN a Candidate has created at least one Journal_Entry in the current calendar day, THE Campaign_Journal SHALL display a summary of the day's entries and any new pending Insights.
3. THE Campaign_Journal SHALL display a count of total Journal_Entry records and the date of the most recent entry on the journal dashboard.

---

### Requirement 18: Journal Entry and Insight Data Serialization

**User Story:** As a developer, I want journal entries and insights to be reliably serialized to and deserialized from JSON so that data integrity is maintained across API boundaries and storage layers.

#### Acceptance Criteria

1. THE Campaign_Journal SHALL serialize Journal_Entry objects to JSON format for API responses and data storage.
2. THE Campaign_Journal SHALL deserialize JSON payloads into Journal_Entry objects for API request processing.
3. THE Campaign_Journal SHALL serialize Insight objects to JSON format for API responses and data storage.
4. THE Campaign_Journal SHALL deserialize JSON payloads into Insight objects for API request processing.
5. FOR ALL valid Journal_Entry objects, serializing to JSON then deserializing back SHALL produce an equivalent Journal_Entry object (round-trip property).
6. FOR ALL valid Insight objects, serializing to JSON then deserializing back SHALL produce an equivalent Insight object (round-trip property).

---

### Requirement 19: Questionnaire Data Serialization

**User Story:** As a developer, I want questionnaire data to be reliably serialized to and deserialized from JSON so that data integrity is maintained when migrating from S3 to DynamoDB and across API boundaries.

#### Acceptance Criteria

1. THE Backend SHALL serialize Onboarding_Questionnaire response objects to JSON format for API responses and DynamoDB_Store persistence.
2. THE Backend SHALL deserialize JSON payloads into Onboarding_Questionnaire response objects for API request processing.
3. FOR ALL valid Onboarding_Questionnaire response objects, serializing to JSON then deserializing back SHALL produce an equivalent object (round-trip property).

---

### Requirement 20: Error Handling and Resilience

**User Story:** As a Candidate, I want the system to handle errors gracefully so that a failure in AI processing does not lose my data or block my workflow.

#### Acceptance Criteria

1. IF the Entry_Tagger encounters an error during tag assignment, THEN THE Campaign_Journal SHALL save the Journal_Entry with "unclassified" tags and notify the Candidate that tagging will be retried.
2. IF the Insights_Engine encounters an error during insight generation, THEN THE WinFlip_Platform SHALL log the error, preserve all existing data, and notify the Candidate that insight generation is temporarily unavailable.
3. IF the Action_Generator encounters an error during action creation, THEN THE Campaign_Journal SHALL log the error and notify the Candidate that action suggestions are temporarily unavailable.
4. IF the Messaging_Generator encounters an error during content generation, THEN THE Campaign_Journal SHALL log the error and notify the Candidate that messaging generation is temporarily unavailable.
5. IF a DynamoDB_Store write operation fails, THEN THE Backend SHALL retry the operation once and, upon second failure, notify the Candidate that the save operation failed and the data should be re-submitted.
6. IF the Chatbot encounters an error during response generation, THEN THE Chatbot SHALL display an error message to the Candidate and preserve the conversation history up to the failed message.
7. IF the Stripe webhook processing fails, THEN THE Backend SHALL log the error with the webhook payload and retry processing on the next webhook delivery.

---

### Requirement 21: CI/CD and Deployment Pipelines

**User Story:** As the WinFlip team, I want automated CI/CD pipelines for frontend and backend so that code changes are tested, validated, and promoted through environments reliably.

#### Acceptance Criteria

1. THE CI_CD_Pipeline SHALL provide separate pipelines for the Frontend (React/Next.js on Amplify) and the Backend (Lambda functions and CDK infrastructure).
2. THE CI_CD_Pipeline SHALL run automated linting, unit tests, and build checks on every code push before allowing deployment.
3. THE CI_CD_Pipeline SHALL support at least two environments: "dev" and "prod".
4. WHEN code passes all automated checks in the dev environment, THE CI_CD_Pipeline SHALL allow promotion to the prod environment through a manual approval step.
5. THE CI_CD_Pipeline SHALL deploy Backend infrastructure changes using AWS CDK with the appropriate environment configuration.
6. THE CI_CD_Pipeline SHALL deploy Frontend changes through AWS Amplify's built-in build and deploy process.
7. IF a pipeline stage fails, THEN THE CI_CD_Pipeline SHALL halt the deployment, log the failure reason, and notify the team.

---

### Requirement 22: Logging, Metrics, and Analytics

**User Story:** As the WinFlip team, we want comprehensive logging, Bedrock token usage tracking, and internal analytics so that we can monitor system health, track costs, and demonstrate engagement metrics for investor demos.

#### Acceptance Criteria

1. THE Backend SHALL log all API requests to Amazon CloudWatch including the endpoint, Candidate identifier (hashed), timestamp, response status code, and response latency.
2. THE Backend SHALL log Amazon Bedrock invocation counts and token usage (input tokens, output tokens) for each AI operation (insights generation, entry tagging, journal insight generation, action generation, messaging generation, chatbot response).
3. THE Backend SHALL log error events with sufficient detail to support debugging, including error type, stack trace summary, and the operation that failed.
4. WHEN the WinFlip team queries CloudWatch logs, THE Backend logs SHALL support filtering by Candidate identifier, operation type, and time range.
5. THE Backend SHALL publish custom CloudWatch metrics for: total API requests per endpoint, Bedrock token consumption per operation type, error rates per Lambda function, and average response latency per endpoint.
6. THE WinFlip_Platform SHALL provide an internal analytics dashboard (accessible to the WinFlip team only) displaying key engagement metrics: daily active users, questionnaire completion rate, journal entries per user, chatbot sessions per user, and insights generated.
7. THE Backend SHALL track and log estimated Bedrock cost per Candidate per operation to support cost monitoring and budget alerts.
