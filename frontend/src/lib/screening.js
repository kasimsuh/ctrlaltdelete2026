export const modelName = "gemini-2.5-flash-native-audio-preview-12-2025";

export const systemInstruction = `ROLE: You are a professional Medical Screening Assistant.
OBJECTIVE: Conduct a brief well-being check by asking specific questions.
RULES:
1. First ask the user to look at their camera for 10 seconds and wait for a "camera done" signal.
2. Then  tell them you are about to ask them some questions on their general wellbeing.First ask: "How are you feeling today?"
3. Follow up with: "Are you experiencing any dizziness, chest pain, or trouble breathing?"
4. Finally ask: "Did you take your morning medications?"
5. STRICT: Only ask these questions. If the user tries to change the subject, politely redirect them back to the screening.
6. TERMINATION: Once all questions are answered, say exactly: "Thank you for your responses. The screening is now complete. Goodbye."`;

export const cameraDurationMs = 10000;

export const completionPhrase =
  "Thank you for your responses. The screening is now complete. Goodbye.";

export const statusColor = {
  Green: "bg-emerald-100 text-moss",
  Yellow: "bg-amber-100 text-gold",
  Red: "bg-rose-100 text-rose",
  Error: "bg-rose-100 text-rose",
  neutral: "bg-amber-50 text-stone-600",
};

export const INITIAL_RESPONSES = [
  { q: "How are you feeling today?", answer: null, transcript: null },
  {
    q: "Are you experiencing any dizziness, chest pain, or trouble breathing?",
    answer: null,
    transcript: null,
  },
  {
    q: "Did you take your morning medications?",
    answer: null,
    transcript: null,
  },
];

export const normalizeAnswer = (questionIndex, text) => {
  if (!text) return null;
  const textLower = text.toLowerCase();
  const yesTerms = ["yes", "yeah", "yep", "affirmative", "true"];
  const noTerms = ["no", "nope", "negative", "false"];

  if (yesTerms.some((term) => textLower.includes(term))) return true;
  if (noTerms.some((term) => textLower.includes(term))) return false;

  if (questionIndex === 0) {
    const positiveTerms = [
      "good",
      "fine",
      "okay",
      "ok",
      "well",
      "great",
      "better",
    ];
    const negativeTerms = [
      "bad",
      "not good",
      "sick",
      "unwell",
      "awful",
      "worse",
    ];
    if (positiveTerms.some((term) => textLower.includes(term))) return true;
    if (negativeTerms.some((term) => textLower.includes(term))) return false;
  }

  return null;
};
