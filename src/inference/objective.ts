/**
 * Objective/fitness function for parameter inference
 *
 * Computes how well a simulated cell state matches a target state.
 */

import type { CellState } from './inference-engine';

export interface TargetState {
  expression: Float32Array;
  weights: Float32Array;  // Per-gene weights (importance)
}

/**
 * Compute fitness (lower is better) between simulated and target states
 *
 * fitness = sum_genes [ weight_g * (target_g - simulated_g)² ]
 */
export function computeFitness(
  simulated: CellState,
  target: TargetState
): number {
  let fitness = 0;
  const nGenes = target.expression.length;

  for (let i = 0; i < nGenes; i++) {
    const diff = (target.expression[i] ?? 0) - (simulated.expression[i] ?? 0);
    const weight = target.weights[i] ?? 1;
    fitness += weight * diff * diff;
  }

  return fitness;
}

/**
 * Compute fitness with optional lineage penalty
 */
export function computeFitnessWithLineage(
  simulated: CellState,
  target: TargetState,
  targetLineage: number | null,
  lineagePenalty: number = 1.0
): number {
  let fitness = computeFitness(simulated, target);

  // Add penalty if lineage doesn't match
  if (targetLineage !== null && simulated.lineage !== targetLineage) {
    fitness += lineagePenalty;
  }

  return fitness;
}

/**
 * Compute average fitness over an ensemble of simulated cells
 */
export function computeEnsembleFitness(
  simulated: CellState[],
  target: TargetState
): { mean: number; min: number; max: number; std: number } {
  const fitnesses = simulated.map(s => computeFitness(s, target));

  const mean = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
  const min = Math.min(...fitnesses);
  const max = Math.max(...fitnesses);

  const variance = fitnesses.reduce((sum, f) => sum + (f - mean) ** 2, 0) / fitnesses.length;
  const std = Math.sqrt(variance);

  return { mean, min, max, std };
}

/**
 * Create uniform weights (all genes equally important)
 */
export function createUniformWeights(nGenes: number): Float32Array {
  const weights = new Float32Array(nGenes);
  weights.fill(1);
  return weights;
}

/**
 * Create weights that emphasize specific gene groups
 */
export function createGroupWeights(
  nGenes: number,
  groups: { [key: string]: number[] },
  groupWeights: { [key: string]: number }
): Float32Array {
  const weights = new Float32Array(nGenes);
  weights.fill(0.1);  // Default low weight for unspecified genes

  for (const [groupName, indices] of Object.entries(groups)) {
    const weight = groupWeights[groupName] ?? 1;
    indices.forEach(i => {
      if (i < nGenes) weights[i] = weight;
    });
  }

  return weights;
}

/**
 * Create weights that emphasize specific genes by index
 */
export function createCustomWeights(
  nGenes: number,
  geneWeights: Map<number, number>,
  defaultWeight: number = 0.1
): Float32Array {
  const weights = new Float32Array(nGenes);
  weights.fill(defaultWeight);

  for (const [idx, weight] of geneWeights) {
    if (idx < nGenes) weights[idx] = weight;
  }

  return weights;
}

/**
 * Normalize weights to sum to 1
 */
export function normalizeWeights(weights: Float32Array): Float32Array {
  const sum = weights.reduce((a, b) => a + b, 0);
  if (sum === 0) return weights;

  const normalized = new Float32Array(weights.length);
  for (let i = 0; i < weights.length; i++) {
    normalized[i] = weights[i] / sum;
  }
  return normalized;
}

/**
 * Compute per-gene errors for visualization
 */
export function computePerGeneError(
  simulated: CellState,
  target: TargetState
): { errors: Float32Array; maxError: number; totalError: number } {
  const nGenes = target.expression.length;
  const errors = new Float32Array(nGenes);
  let maxError = 0;
  let totalError = 0;

  for (let i = 0; i < nGenes; i++) {
    const diff = Math.abs((target.expression[i] ?? 0) - (simulated.expression[i] ?? 0));
    errors[i] = diff;
    if (diff > maxError) maxError = diff;
    totalError += diff * (target.weights[i] ?? 1);
  }

  return { errors, maxError, totalError };
}

/**
 * Create target state from expression values and optional weights
 */
export function createTargetState(
  expression: Float32Array | number[],
  weights?: Float32Array | number[]
): TargetState {
  const expr = expression instanceof Float32Array ? expression : new Float32Array(expression);
  const w = weights
    ? (weights instanceof Float32Array ? weights : new Float32Array(weights))
    : createUniformWeights(expr.length);

  return { expression: expr, weights: w };
}

/**
 * Check if fitness has converged (below threshold)
 */
export function hasConverged(fitness: number, threshold: number = 0.1): boolean {
  return fitness < threshold;
}

/**
 * Compute correlation between simulated and target expression
 */
export function computeCorrelation(
  simulated: CellState,
  target: TargetState
): number {
  const nGenes = target.expression.length;

  // Compute means
  let simMean = 0, tarMean = 0;
  for (let i = 0; i < nGenes; i++) {
    simMean += simulated.expression[i] ?? 0;
    tarMean += target.expression[i] ?? 0;
  }
  simMean /= nGenes;
  tarMean /= nGenes;

  // Compute correlation
  let cov = 0, simVar = 0, tarVar = 0;
  for (let i = 0; i < nGenes; i++) {
    const simDiff = (simulated.expression[i] ?? 0) - simMean;
    const tarDiff = (target.expression[i] ?? 0) - tarMean;
    cov += simDiff * tarDiff;
    simVar += simDiff * simDiff;
    tarVar += tarDiff * tarDiff;
  }

  if (simVar === 0 || tarVar === 0) return 0;
  return cov / Math.sqrt(simVar * tarVar);
}
