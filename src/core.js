/**
 * Jacobi-method-solver — core implementation.
 *
 * The Jacobi method solves A·x = b by holding x fixed during each sweep and
 * computing every new component purely from the previous sweep's values.
 * This is what makes it "Jacobi" rather than Gauss–Seidel (which uses
 * already-updated components within the same sweep).
 *
 * Convergence is guaranteed only for strictly diagonally dominant matrices.
 * We check that precondition up front and refuse to iterate otherwise, because
 * running Jacobi on a non-convergent system produces a confidently wrong
 * answer — far worse than an explicit error.
 */

/**
 * Result of a solve attempt.
 *
 * @typedef {Object} SolveResult
 * @property {boolean} converged      True when the 2-norm of the residual fell below tol.
 * @property {number[]} x             Last iterate (length n).
 * @property {number} residual        2-norm of A·x − b at the final iterate.
 * @property {number} iterations      Number of sweeps performed (capped at maxIterations).
 */

/**
 * Solve A·x = b with the Jacobi method.
 *
 * @param {number[][]} A            Square coefficient matrix (n×n), row-major.
 * @param {number[]} b             Right-hand side (length n).
 * @param {Object} [opts]
 * @param {number} [opts.tol=1e-10]         Stop when the residual 2-norm is below this.
 * @param {number} [opts.maxIterations=1000] Hard cap on sweeps.
 * @param {number[]} [opts.initialGuess]   Starting vector; defaults to zeros.
 * @returns {SolveResult}
 * @throws {TypeError|Error} On shape mismatch, non-finite entries, or a matrix
 *   that is not strictly diagonally dominant.
 */
export function solve(A, b, opts = {}) {
  if (!Array.isArray(A) || !Array.isArray(b)) {
    throw new TypeError('A must be a 2D array and b a 1D array');
  }
  const n = b.length;
  if (n === 0) {
    throw new Error('System size must be positive');
  }
  if (A.length !== n) {
    throw new Error(`A has ${A.length} rows but b has length ${n}`);
  }
  for (let i = 0; i < n; i++) {
    if (!Array.isArray(A[i]) || A[i].length !== n) {
      throw new Error(`Row ${i} of A is not a length-${n} array`);
    }
  }

  const tol = opts.tol ?? 1e-10;
  const maxIterations = opts.maxIterations ?? 1000;
  if (!(tol > 0) || !Number.isFinite(tol)) {
    throw new Error('tol must be a positive finite number');
  }
  if (!Number.isInteger(maxIterations) || maxIterations < 1) {
    throw new Error('maxIterations must be a positive integer');
  }

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (!Number.isFinite(A[i][j])) {
        throw new Error(`A[${i}][${j}] is not finite`);
      }
    }
    if (!Number.isFinite(b[i])) {
      throw new Error(`b[${i}] is not finite`);
    }
  }

  if (!isStrictlyDiagonallyDominant(A)) {
    throw new Error(
      'Matrix is not strictly diagonally dominant; Jacobi is not guaranteed to converge',
    );
  }

  let x = opts.initialGuess
    ? opts.initialGuess.slice()
    : new Array(n).fill(0);
  if (x.length !== n) {
    throw new Error('initialGuess has the wrong length');
  }
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(x[i])) {
      throw new Error(`initialGuess[${i}] is not finite`);
    }
  }

  let residual = residualNorm(A, b, x);
  let iter = 0;

  while (residual > tol && iter < maxIterations) {
    const next = new Array(n);
    for (let i = 0; i < n; i++) {
      let sum = b[i];
      for (let j = 0; j < n; j++) {
        if (j !== i) {
          sum -= A[i][j] * x[j];
        }
      }
      next[i] = sum / A[i][i];
    }
    x = next;
    residual = residualNorm(A, b, x);
    iter++;
  }

  return {
    converged: residual <= tol,
    x,
    residual,
    iterations: iter,
  };
}

/**
 * A matrix is strictly diagonally dominant when, for every row i,
 * |A[i][i]| > Σ_{j≠i} |A[i][j]|.
 *
 * We require strict dominance because weak dominance (≥) does not guarantee
 * convergence in the general case.
 */
function isStrictlyDiagonallyDominant(A) {
  const n = A.length;
  for (let i = 0; i < n; i++) {
    let diag = Math.abs(A[i][i]);
    let off = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i) off += Math.abs(A[i][j]);
    }
    if (!(diag > off)) return false;
  }
  return true;
}

/**
 * Euclidean norm of the residual r = A·x − b.
 *
 * Using the 2-norm (rather than the max-norm) so the tolerance has a familiar
 * geometric meaning: the distance from x to the exact solution set.
 */
function residualNorm(A, b, x) {
  const n = b.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    let r = -b[i];
    for (let j = 0; j < n; j++) {
      r += A[i][j] * x[j];
    }
    sum += r * r;
  }
  return Math.sqrt(sum);
}
