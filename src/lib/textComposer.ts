export function appendLetter(text: string, letter: string): string {
  return text + letter;
}

export function appendRecognizedLetter(
  text: string,
  letter: string | null,
  confidence: number,
  confidenceThreshold: number,
  isStable: boolean,
): string {
  if (!letter || !isStable || !(confidence >= confidenceThreshold)) return text;
  return appendLetter(text, letter);
}

export function appendSpace(text: string): string {
  return text.endsWith(' ') ? text : text + ' ';
}

export function removeLastCharacter(text: string): string {
  return text.slice(0, -1);
}

export function countWords(text: string): number {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}
