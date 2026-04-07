export const OFFICE_OPTIONS = [
  "House_of_Delegates", "U.S._House", "Congress", "Governor", "Local_Office",
  "President", "U.S._Senate", "Lieutenant_Governor", "Attorney_General",
  "Senate_of_Virginia", "Commonwealth's_Attorney", "Commissioner_of_the_Revenue",
  "County_Board_Member", "Sheriff", "Treasurer", "Clerk_of_Court", "School_Board",
  "Soil_and_Water_Conservation_Director", "Mayor", "City_Council", "Town_Council",
  "Board_of_Supervisors",
] as const;

export const STATEWIDE_OFFICES = new Set([
  "President", "U.S._Senate", "Governor", "Lieutenant_Governor", "Attorney_General",
]);

export const BACKGROUND_QUESTIONS: Record<string, string> = {
  militaryBackground: "Do you have a military background (veteran, active duty, or immediate family)?",
  publicSafety: "Have you served in law enforcement, firefighting, or another public safety role?",
  unionBackground: "Do you come from a union household or have direct ties to labor/organizing?",
  businessOwner: "Are you a small business owner or entrepreneur?",
  publicService: "Have you held any public service roles (school board, council, community board)?",
  faithCommunity: "Do you identify strongly with a faith community or civic organization?",
  firstTime: "Is this your first campaign, or have you run for office before?",
};

export const ARCHETYPE_QUESTIONS: Record<string, { question: string; options: string[] }> = {
  themeSong: {
    question: "If your campaign had a theme song, what would it sound like?",
    options: [
      "🎶 Upbeat pop anthem – optimistic, inclusive, energetic",
      "🎸 Rock/hip-hop banger – fiery, bold, disruptive",
      "🎻 Folk/acoustic ballad – personal, grounded, community-focused",
      "🎺 Patriotic or orchestral march – traditional, serious, trustworthy",
      "🎷 Jazz / soulful groove – creative, improvisational, approachable",
      "🎤 Country anthem – rooted, local, values-driven",
      "🎧 Electronic / techno beat – modern, youthful, future-focused",
      "🎹 Piano/classical piece – thoughtful, steady, intellectual",
    ],
  },
  debateReaction: {
    question: "At your first debate, you get a tough, unexpected question. What do you do?",
    options: [
      "Answer honestly, even if imperfect",
      "Pivot to a key policy plan",
      "Use humor to break tension",
      "Respond passionately about values",
    ],
  },
  coffeeShopIntro: {
    question: "When you meet voters in a coffee shop, how do you introduce yourself?",
    options: [
      "Ask them what issues matter most",
      "Introduce yourself formally with experience",
      "Make a friendly, neighborly comment first",
      "Dive straight into an issue",
    ],
  },
  leadershipStyle: {
    question: "Which leader's style feels closest to yours?",
    options: [
      "Barack Obama (inspirational coalition-builder)",
      "Elizabeth Warren (policy fighter)",
      "Alexandria Ocasio-Cortez (energetic change agent)",
      "Joe Biden (empathetic unifier)",
    ],
  },
  eventExcitement: {
    question: "Which event excites you most?",
    options: [
      "Town hall with live Q&A",
      "Big rally with cheering supporters",
      "Policy roundtable with experts",
      "Block party with neighbors",
    ],
  },
  quickDecisions: {
    question: "When making a quick decision on policy, what guides you most?",
    options: ["Consensus with advisors", "Gut and values", "Data and evidence", "Constituents' views"],
  },
  tagLine: {
    question: "Which tagline would you pick for your campaign?",
    options: [
      "\u201cFor a Stronger, Fairer Community\u201d",
      "\u201cNew Leadership, New Ideas\u201d",
      "\u201cProven Experience. Trusted Results.\u201d",
      "\u201cStanding Up for Working Families\u201d",
    ],
  },
  socialMedia: {
    question: "On social media, how would you announce a new policy?",
    options: [
      "With facts and clarity (informative)",
      "Through a short story about someone affected",
      "With bold, catchy language (edgy)",
      "As an optimistic call to action",
    ],
  },
  negativeComments: {
    question: "Your opponent airs a negative ad against you. How do you respond?",
    options: [
      "Calmly fact-check it",
      "Respond with a clever or story-driven video",
      "Hit back hard in your next speech",
      "Stay positive and continue with your message",
    ],
  },
  success: {
    question: "After the election, beyond winning, what would make you feel most successful?",
    options: [
      "Inspiring new voters and building a movement",
      "Shifting the community conversation",
      "Running with integrity and positivity",
      "Building lasting coalitions",
    ],
  },
  symbolism: {
    question: "If your campaign were represented by a symbol, which feels most like you?",
    options: [
      "A star ⭐️ - standing for integrity, recognition, and achievement.",
      "A bridge 🌉 - connecting people and ideas across divides.",
      "A flame 🔥 - energy, urgency, and fighting spirit.",
      "A toolbox 🧰 - practical, focused on solving problems.",
      "A sunrise 🌄 - hope, inspiration, and new beginnings.",
    ],
  },
  headlines: {
    question: "Imagine the local newspaper writes a headline about your campaign. Which one would you prefer?",
    options: [
      "\u201cCandidate Brings Community Together Across Divides\u201d",
      "\u201cCandidate Pushes Bold New Ideas to Shake Up the System\u201d",
      "\u201cCandidate Offers Detailed Plan to Fix Local Problems\u201d",
      "\u201cCandidate Inspires Hope for a Brighter Future\u201d",
    ],
  },
};
