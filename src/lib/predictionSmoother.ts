import { PredictionResult } from '../types/prediction';

interface PredictionSample {
  letter: string | null;
  confidence: number;
  timestamp: number;
}

export interface SmoothedPrediction {
  prediction: PredictionResult;
  isStable: boolean;
  stableSince: number | null;
}

export class PredictionSmoother {
  private readonly samples: PredictionSample[] = [];
  private candidateLetter: string | null = null;
  private candidateSince: number | null = null;

  constructor(
    private readonly windowSize = 5,
    private readonly votesRequired = 4,
  ) {}

  update(
    result: PredictionResult,
    timestamp: number,
    confidenceThreshold: number,
  ): SmoothedPrediction {
    const acceptedLetter =
      result.letter && result.confidence >= confidenceThreshold
        ? result.letter
        : null;

    this.samples.push({
      letter: acceptedLetter,
      confidence: result.confidence,
      timestamp,
    });
    if (this.samples.length > this.windowSize) this.samples.shift();

    if (acceptedLetter !== this.candidateLetter) {
      this.candidateLetter = acceptedLetter;
      this.candidateSince = acceptedLetter ? timestamp : null;
    }

    const matchingSamples = acceptedLetter
      ? this.samples.filter((sample) => sample.letter === acceptedLetter)
      : [];
    const isStable = matchingSamples.length >= this.votesRequired;
    const confidence = isStable
      ? matchingSamples.reduce((sum, sample) => sum + sample.confidence, 0) /
        matchingSamples.length
      : result.confidence;

    return {
      prediction: {
        ...result,
        letter: isStable ? acceptedLetter : null,
        confidence,
      },
      isStable,
      stableSince: isStable ? this.candidateSince : null,
    };
  }

  reset(): void {
    this.samples.length = 0;
    this.candidateLetter = null;
    this.candidateSince = null;
  }
}

export class AutoCommitGate {
  private lastCommittedLetter: string | null = null;

  shouldCommit(
    letter: string,
    stableSince: number,
    now: number,
    enabled: boolean,
  ): boolean {
    if (
      !enabled ||
      now - stableSince < 1000 ||
      this.lastCommittedLetter === letter
    ) {
      return false;
    }

    this.lastCommittedLetter = letter;
    return true;
  }

  markCommitted(letter: string): void {
    this.lastCommittedLetter = letter;
  }

  reset(): void {
    this.lastCommittedLetter = null;
  }
}
