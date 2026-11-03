import assert from 'node:assert/strict';
import test from 'node:test';
import { PredictionResult } from '../types/prediction';
import { AutoCommitGate, PredictionSmoother } from './predictionSmoother';
import {
  appendRecognizedLetter,
  appendLetter,
  appendSpace,
  countWords,
  removeLastCharacter,
} from './textComposer';

function prediction(
  letter: string | null,
  confidence: number,
): PredictionResult {
  return {
    ready: true,
    letter,
    confidence,
    model_version: 'test',
    inference_time_ms: 1,
  };
}

test('requires a 4-of-5 vote before exposing a letter', () => {
  const smoother = new PredictionSmoother();
  const first = smoother.update(prediction('A', 0.9), 100, 0.6);
  const second = smoother.update(prediction('G', 0.9), 200, 0.6);
  const third = smoother.update(prediction('A', 0.9), 300, 0.6);
  const fourth = smoother.update(prediction('A', 0.9), 400, 0.6);
  const fifth = smoother.update(prediction('A', 0.9), 500, 0.6);

  assert.equal(first.prediction.letter, null);
  assert.equal(second.prediction.letter, null);
  assert.equal(third.prediction.letter, null);
  assert.equal(fourth.prediction.letter, null);
  assert.equal(fifth.prediction.letter, 'A');
  assert.equal(fifth.isStable, true);
  assert.equal(fifth.prediction.confidence, 0.9);
});

test('rejects low-confidence outputs and resets the hold timer', () => {
  const smoother = new PredictionSmoother();
  for (let i = 0; i < 4; i++) {
    smoother.update(prediction('A', 0.9), i * 100, 0.6);
  }

  const uncertain = smoother.update(prediction('A', 0.59), 500, 0.6);
  const restarting = smoother.update(prediction('A', 0.9), 600, 0.6);

  assert.equal(uncertain.prediction.letter, null);
  assert.equal(uncertain.stableSince, null);
  assert.equal(restarting.isStable, true);
  assert.equal(restarting.stableSince, 600);
});

test('auto-commit waits one second and rearms only after another letter', () => {
  const gate = new AutoCommitGate();

  assert.equal(gate.shouldCommit('A', 100, 1099, true), false);
  assert.equal(gate.shouldCommit('A', 100, 1100, true), true);
  assert.equal(gate.shouldCommit('A', 100, 3000, true), false);
  assert.equal(gate.shouldCommit('B', 2000, 2999, true), false);
  assert.equal(gate.shouldCommit('B', 2000, 3000, true), true);
  assert.equal(gate.shouldCommit('A', 4000, 5000, true), true);

  gate.reset();
  assert.equal(gate.shouldCommit('A', 6000, 7000, true), true);
});

test('text composer operations update the same string used for counters', () => {
  let text = appendLetter('', 'A');
  text = appendLetter(text, 'B');
  text = appendSpace(text);
  text = appendSpace(text);
  text = appendLetter(text, 'C');

  assert.equal(text, 'AB C');
  assert.equal(text.length, 4);
  assert.equal(countWords(text), 2);
  assert.equal(removeLastCharacter(text), 'AB ');
  assert.equal(removeLastCharacter(''), '');
});

test('Add Letter appends only a stable prediction above the confidence threshold', () => {
  assert.equal(appendRecognizedLetter('AB', 'C', 0.6, 0.6, true), 'ABC');
  assert.equal(appendRecognizedLetter('AB', 'C', 0.59, 0.6, true), 'AB');
  assert.equal(appendRecognizedLetter('AB', 'C', 0.99, 0.6, false), 'AB');
  assert.equal(appendRecognizedLetter('AB', null, 0.99, 0.6, true), 'AB');
});
