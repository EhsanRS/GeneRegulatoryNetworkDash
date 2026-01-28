/**
 * Web Worker for background CMA-ES optimization
 *
 * Runs the optimization loop in a separate thread to keep
 * the UI responsive during long optimization runs.
 */

import { CMAES } from './optimizer';
import { InferenceEngine } from './inference-engine';
import type { GRNData } from './inference-engine';
import {
  decode,
  getDimensions,
  getBounds,
  getDefaultVector,
  type EncodingConfig,
  type SimulationParams,
} from './encoding';
import { computeFitness, type TargetState } from './objective';

// Message types
export interface WorkerStartMessage {
  type: 'start';
  grnData: GRNData;
  target: {
    expression: number[];
    weights: number[];
  };
  config: {
    includeGlobal: boolean;
    includeKnockouts: boolean;
    includeMorphogens: boolean;
    includeModifiers: boolean;  // Include gene modifiers (overexpression/inhibition)
    maxGenerations: number;
    populationSize: number;
    sigma: number;
    maxTime: number;  // Simulation time
    ensembleSize: number;  // Number of cells per evaluation
    targetLineage: number;  // 0 = ensemble (all positions), 1-6 = specific lineage
    uniformMorphogens: boolean;  // Apply morphogens uniformly to all cells
  };
  knockoutCandidates: number[];  // Gene indices that can be knocked out
  modifierCandidates: number[];  // Gene indices that can have modifiers applied
}

export interface WorkerStopMessage {
  type: 'stop';
}

export interface WorkerPauseMessage {
  type: 'pause';
}

export interface WorkerResumeMessage {
  type: 'resume';
}

export type WorkerInMessage = WorkerStartMessage | WorkerStopMessage | WorkerPauseMessage | WorkerResumeMessage;

export interface WorkerProgressMessage {
  type: 'progress';
  generation: number;
  bestFitness: number;
  meanFitness: number;
  sigma: number;
  bestParams: number[];
  correlation: number;
}

export interface WorkerDoneMessage {
  type: 'done';
  result: {
    params: number[];
    fitness: number;
    generation: number;
    converged: boolean;
  };
  decodedParams: SimulationParams;
}

export interface WorkerErrorMessage {
  type: 'error';
  error: string;
}

export type WorkerOutMessage = WorkerProgressMessage | WorkerDoneMessage | WorkerErrorMessage;

// Worker state
let engine: InferenceEngine | null = null;
let optimizer: CMAES | null = null;
let encodingConfig: EncodingConfig | null = null;
let targetState: TargetState | null = null;
let maxGenerations = 200;
let simMaxTime = 12;
let ensembleSize = 5;
let targetLineage = 0;  // 0 = ensemble (all positions), 1-6 = specific lineage
let uniformMorphogens = true;  // Apply morphogens uniformly
let isPaused = false;
let shouldStop = false;

// Phase flag for two-phase optimization
let useVariancePenalty = false;

/**
 * Evaluate fitness for a parameter vector
 *
 * Phase 1 (exploration): Simple average fitness - fast
 * Phase 2 (fine-tuning): Adds variance penalty to avoid bimodal distributions
 *
 * Returns { fitness: number for optimization, rawFitness: number for display }
 */
function evaluateFitnessDetailed(paramVector: Float64Array): { fitness: number; rawFitness: number } {
  if (!engine || !encodingConfig || !targetState) {
    throw new Error('Worker not initialized');
  }

  const params = decode(paramVector, encodingConfig);

  // Run simulation - either for specific lineage or ensemble
  let results: { expression: Float32Array; lineage: number }[];
  if (targetLineage > 0) {
    // Single cell at optimal position for target lineage
    const result = engine.runForLineage(params, targetLineage, simMaxTime, uniformMorphogens);
    results = [result];
  } else {
    // Ensemble across all positions
    results = engine.runEnsemble(params, ensembleSize, simMaxTime, uniformMorphogens);
  }

  // Compute average fitness across results
  let totalFitness = 0;
  for (const result of results) {
    totalFitness += computeFitness(result, targetState);
  }
  const meanFitness = totalFitness / results.length;

  // Phase 1: Just return average fitness (fast exploration)
  if (!useVariancePenalty) {
    return { fitness: meanFitness, rawFitness: meanFitness };
  }

  // Phase 2: Add variance penalty to discourage bimodality
  const nGenes = targetState.expression.length;
  let variancePenalty = 0;
  const varianceWeight = 0.3;

  for (let g = 0; g < nGenes; g++) {
    const weight = targetState.weights[g] ?? 0;
    if (weight === 0) continue;

    // Compute mean and variance for this gene across cells
    let geneSum = 0;
    for (const result of results) {
      geneSum += result.expression[g] ?? 0;
    }
    const geneMean = geneSum / results.length;

    let geneVariance = 0;
    for (const result of results) {
      const diff = (result.expression[g] ?? 0) - geneMean;
      geneVariance += diff * diff;
    }
    geneVariance /= results.length;

    variancePenalty += weight * geneVariance;
  }

  // Return both: penalized for optimization, raw for display
  return {
    fitness: meanFitness + varianceWeight * variancePenalty,
    rawFitness: meanFitness
  };
}

// Simple wrapper for optimization (returns penalized fitness)
function evaluateFitness(paramVector: Float64Array): number {
  return evaluateFitnessDetailed(paramVector).fitness;
}

/**
 * Compute correlation for a parameter vector
 */
function computeCorrelation(paramVector: Float64Array): number {
  if (!engine || !encodingConfig || !targetState) return 0;

  const params = decode(paramVector, encodingConfig);
  const result = engine.runSimulation(params, Math.PI, simMaxTime);

  const simExpr = result.expression;
  const tarExpr = targetState.expression;
  const nGenes = tarExpr.length;

  let simMean = 0, tarMean = 0;
  for (let i = 0; i < nGenes; i++) {
    simMean += simExpr[i] ?? 0;
    tarMean += tarExpr[i] ?? 0;
  }
  simMean /= nGenes;
  tarMean /= nGenes;

  let cov = 0, simVar = 0, tarVar = 0;
  for (let i = 0; i < nGenes; i++) {
    const simDiff = (simExpr[i] ?? 0) - simMean;
    const tarDiff = (tarExpr[i] ?? 0) - tarMean;
    cov += simDiff * tarDiff;
    simVar += simDiff * simDiff;
    tarVar += tarDiff * tarDiff;
  }

  if (simVar === 0 || tarVar === 0) return 0;
  return cov / Math.sqrt(simVar * tarVar);
}

/**
 * Main optimization loop - Two phase approach
 *
 * Phase 1: Fast exploration using simple average fitness
 * Phase 2: Fine-tuning with variance penalty to avoid bimodal solutions
 */
async function runOptimization(): Promise<void> {
  if (!optimizer || !engine || !encodingConfig || !targetState) {
    throw new Error('Worker not initialized');
  }

  // Phase 1: Main exploration (no variance penalty)
  const finetuneGenerations = Math.min(50, Math.floor(maxGenerations * 0.2));
  const explorationGenerations = maxGenerations - finetuneGenerations;

  useVariancePenalty = false;

  for (let gen = 0; gen < maxGenerations; gen++) {
    // Check for stop signal
    if (shouldStop) {
      break;
    }

    // Wait while paused
    while (isPaused && !shouldStop) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    if (shouldStop) break;

    // Switch to Phase 2 (fine-tuning) after exploration
    if (gen === explorationGenerations && !useVariancePenalty) {
      useVariancePenalty = true;
      // Re-evaluate current best with new fitness to reset optimizer state
      optimizer.resetSigma(0.1);  // Reduce step size for fine-tuning
    }

    // Sample population
    const population = optimizer.samplePopulation();

    // Evaluate fitness for each candidate
    const fitnesses: number[] = [];
    for (const candidate of population) {
      fitnesses.push(evaluateFitness(candidate));
    }

    // Update optimizer
    optimizer.update(population, fitnesses);

    // Get current best
    const best = optimizer.getBest();
    const meanFitness = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
    const correlation = computeCorrelation(best.params);

    // Get raw fitness (without variance penalty) for display
    const { rawFitness } = evaluateFitnessDetailed(best.params);

    // Send progress update (report raw fitness for comparability)
    const progressMsg: WorkerProgressMessage = {
      type: 'progress',
      generation: gen,
      bestFitness: rawFitness,  // Report raw fitness for display
      meanFitness,
      sigma: optimizer.getSigma(),
      bestParams: Array.from(best.params),
      correlation,
    };
    self.postMessage(progressMsg);

    // Check convergence (only in phase 1, always complete phase 2)
    if (!useVariancePenalty && (best.converged || best.fitness < 0.01)) {
      // Early convergence - skip to fine-tuning phase
      useVariancePenalty = true;
      optimizer.resetSigma(0.1);
    }

    // Yield to allow message processing
    if (gen % 5 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  // Send final result
  const best = optimizer.getBest();
  const decodedParams = decode(best.params, encodingConfig);
  const { rawFitness } = evaluateFitnessDetailed(best.params);

  const doneMsg: WorkerDoneMessage = {
    type: 'done',
    result: {
      params: Array.from(best.params),
      fitness: rawFitness,  // Report raw fitness for display
      generation: best.generation,
      converged: best.converged,
    },
    decodedParams,
  };
  self.postMessage(doneMsg);
}

/**
 * Handle incoming messages
 */
self.onmessage = async (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;

  try {
    switch (msg.type) {
      case 'start': {
        // Initialize engine
        engine = new InferenceEngine(msg.grnData);

        // Create target state
        targetState = {
          expression: new Float32Array(msg.target.expression),
          weights: new Float32Array(msg.target.weights),
        };

        // Create encoding config
        encodingConfig = {
          includeGlobal: msg.config.includeGlobal,
          includeKnockouts: msg.config.includeKnockouts,
          includeMorphogens: msg.config.includeMorphogens,
          includeModifiers: msg.config.includeModifiers,
          geneNames: engine.getGeneNames(),
          knockoutGeneIndices: msg.knockoutCandidates,
          modifierGeneIndices: msg.modifierCandidates,
        };

        maxGenerations = msg.config.maxGenerations;
        simMaxTime = msg.config.maxTime;
        ensembleSize = msg.config.ensembleSize;
        targetLineage = msg.config.targetLineage;
        uniformMorphogens = msg.config.uniformMorphogens;

        // Initialize optimizer
        const dims = getDimensions(encodingConfig);
        const bounds = getBounds(encodingConfig);
        const initialMean = getDefaultVector(encodingConfig);

        optimizer = new CMAES({
          dimensions: dims,
          populationSize: msg.config.populationSize,
          sigma: msg.config.sigma,
          bounds,
          initialMean,
        });

        shouldStop = false;
        isPaused = false;

        // Run optimization
        await runOptimization();
        break;
      }

      case 'stop':
        shouldStop = true;
        break;

      case 'pause':
        isPaused = true;
        break;

      case 'resume':
        isPaused = false;
        break;
    }
  } catch (error) {
    const errorMsg: WorkerErrorMessage = {
      type: 'error',
      error: error instanceof Error ? error.message : String(error),
    };
    self.postMessage(errorMsg);
  }
};
