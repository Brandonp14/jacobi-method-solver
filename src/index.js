/**
 * Public entry point for the jacobi-method-solver package.
 *
 * Re-exports the single solve function so callers depend on the package
 * boundary, not on internal layout.
 */
export { solve } from './core.js';
