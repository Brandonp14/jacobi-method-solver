import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { solve } from '../src/index.js';

function approxEqual(actual, expected, tol = 1e-6) {
  assert.ok(
    Math.abs(actual - expected) <= tol,
    `expected ${expected}, got ${actual}`,
  );
}

describe('solve — input validation', () => {
  it('rejects a non-array A', () => {
    assert.throws(() => solve('no', [1, 2]), /2D array/);
  });

  it('rejects a non-array b', () => {
    assert.throws(() => solve([[1]], 'no'), /1D array/);
  });

  it('rejects an empty system', () => {
    assert.throws(() => solve([], []), /positive/);
  });

  it('rejects a row-count mismatch', () => {
    assert.throws(() => solve([[1, 2]], [1, 2, 3]), /has 1 rows but b has length 3/);
  });

  it('rejects a non-square matrix', () => {
    assert.throws(() => solve([[1, 2, 3], [4, 5, 6]], [1, 2]), /length-2 array/);
  });

  it('rejects a non-finite entry in A', () => {
    assert.throws(() => solve([[1, NaN], [0, 1]], [1, 2]), /not finite/);
  });

  it('rejects a non-finite entry in b', () => {
    assert.throws(
      () => solve([[2, 0], [0, 2]], [Infinity, 1]),
      /not finite/,
    );
  });

  it('rejects a non-positive tol', () => {
    assert.throws(
      () => solve([[2, 0], [0, 2]], [2, 2], { tol: 0 }),
      /positive finite/,
    );
  });

  it('rejects a non-positive maxIterations', () => {
    assert.throws(
      () => solve([[2, 0], [0, 2]], [2, 2], { maxIterations: 0 }),
    );
  });

  it('rejects an initialGuess of the wrong length', () => {
    assert.throws(
      () => solve([[2, 0], [0, 2]], [2, 2], { initialGuess: [1] }),
      /wrong length/,
    );
  });

  it('does not iterate when the initial guess already satisfies tol', () => {
    const A = [[2, 0], [0, 2]];
    const b = [2, 4];
    const result = solve(A, b, { initialGuess: [1, 2] });
    assert.equal(result.iterations, 0);
    assert.equal(result.converged, true);
    approxEqual(result.residual, 0);
  });

  it('reports non-convergence when maxIterations is exhausted', () => {
    const A = [[4, 1], [1, 3]];
    const b = [1, 2];
    const result = solve(A, b, { maxIterations: 1, tol: 1e-20 });
    assert.equal(result.converged, false);
    assert.equal(result.iterations, 1);
  });

  it('does not mutate the caller\'s A or b', () => {
    const A = [[4, 1], [1, 3]];
    const b = [1, 2];
    const Acopy = A.map((r) => r.slice());
    const bcopy = b.slice();
    solve(A, b, { initialGuess: [0, 0] });
    assert.deepEqual(A, Acopy);
    assert.deepEqual(b, bcopy);
  });

  it('does not mutate the caller\'s initial guess', () => {
    const A = [[4, 1], [1, 3]];
    const b = [1, 2];
    const guess = [0, 0];
    const guessCopy = guess.slice();
    solve(A, b, { initialGuess: guess });
    assert.deepEqual(guess, guessCopy);
  });
});
