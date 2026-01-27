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
    maxGenerations: number;
    populationSize: number;
    sigma: number;
    maxTime: number;  // Simulation time
    ensembleSize: number;  // Number of cells per evaluation
  };
  knockoutCandidates: number[];  // Gene indices that can be knocked out
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
let isPaused = false;
let shouldStop = false;

/**
 * Evaluate fitness for a parameter vector
 */
function evaluateFitness(paramVector: Float64Array): number {
  if (!engine || !encodingConfig || !targetState) {
    throw new Error('Worker not initialized');
  }

  const params = decode(paramVector, encodingConfig);

  // Run ensemble simulation
  const results = engine.runEnsemble(params, ensembleSize, simMaxTime);

  // Average fitness across ensemble
  let totalFitness = 0;
  for (const result of results) {
    totalFitness += computeFitness(result, targetState);
  }

  return totalFitness / results.length;
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
 * Main optimization loop
 */
async function runOptimization(): Promise<void> {
  if (!optimizer || !engine || !encodingConfig || !targetState) {
    throw new Error('Worker not initialized');
  }

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

    // Send progress update
    const progressMsg: WorkerProgressMessage = {
      type: 'progress',
      generation: gen,
      bestFitness: best.fitness,
      meanFitness,
      sigma: optimizer.getSigma(),
      bestParams: Array.from(best.params),
      correlation,
    };
    self.postMessage(progressMsg);

    // Check convergence
    if (best.converged || best.fitness < 0.01) {
      break;
    }

    // Yield to allow message processing
    if (gen % 5 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  // Send final result
  const best = optimizer.getBest();
  const decodedParams = decode(best.params, encodingConfig);

  const doneMsg: WorkerDoneMessage = {
    type: 'done',
    result: {
      params: Array.from(best.params),
      fitness: best.fitness,
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
          geneNames: engine.getGeneNames(),
          knockoutGeneIndices: msg.knockoutCandidates,
        };

        maxGenerations = msg.config.maxGenerations;
        simMaxTime = msg.config.maxTime;
        ensembleSize = msg.config.ensembleSize;

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
