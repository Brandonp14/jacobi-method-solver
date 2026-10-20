# jacobi-method-solver

Solves a strictly diagonally dominant system of linear equations `A·x = b` using the Jacobi iterative method, returning the solution vector along with the final residual and iteration count.

## Usage

```js
import { solve } from 'jacobi-method-solver';

const A = [
  [4, 1],
  [1, 3],
];
const b = [1, 2];

const result = solve(A, b, { tol: 1e-10, maxIterations: 1000 });
// result.converged  -> true
// result.x          -> [0.2, 0.4]  (approximately)
// result.residual   -> near 0
// result.iterations -> number of sweeps performed
```

`solve(A, b, opts?)` returns `{ converged, x, residual, iterations }`.

## Why this exists

Direct solvers (LU, Cholesky) are exact but allocate and factorise. For large, sparse, diagonally dominant systems — the kind that show up in finite-difference discretisations — a few Jacobi sweeps are often all you need to reach a usable approximation, and each sweep is trivially parallelisable because it reads only the previous iterate. The trade-off: Jacobi converges only for strictly diagonally dominant (or symmetric positive definite) matrices, and convergence is linear, not quadratic. This library leans into the narrow guaranteed-convergence case rather than trying to be a general-purpose solver.

## The awkward edge

The solver checks strict diagonal dominance up front and throws if it does not hold. If your matrix is merely *weakly* dominant (`|a_ii| ≥ Σ|a_ij|`), Jacobi may still converge in practice — but this library will refuse to run rather than guess. Transform your system (row scaling, reordering) or use a different method if you need that case.

Default tolerance is `1e-10` on the residual 2-norm, capped at `1000` iterations. Both are overridable via the options object.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

