/**
 * Simulation wrapper for parameter inference
 *
 * Runs a simplified single-cell simulation to steady state
 * for fitness evaluation during optimization.
 */

import type { SimulationParams, GlobalParams } from './encoding';

export interface GRNEdge {
  source: string;
  target: string;
  weight: number;
  type: string;
}

export interface GRNData {
  gene_names: string[];
  edges: GRNEdge[];
}

export interface GeneGroups {
  tf_prog: number[];
  tf_lin: number[];
  ligand: number[];
  receptor: number[];
  target: number[];
  housekeeping: number[];
  other: number[];
}

export interface CellState {
  expression: Float32Array;
  lineage: number;
}

/**
 * Build gene groups from gene names
 */
export function buildGeneGroups(geneNames: string[]): GeneGroups {
  const groups: GeneGroups = {
    tf_prog: [],
    tf_lin: [],
    ligand: [],
    receptor: [],
    target: [],
    housekeeping: [],
    other: [],
  };

  geneNames.forEach((name, i) => {
    if (name.startsWith('TF_PROG')) groups.tf_prog.push(i);
    else if (name.startsWith('TF_LIN')) groups.tf_lin.push(i);
    else if (name.startsWith('LIG')) groups.ligand.push(i);
    else if (name.startsWith('REC')) groups.receptor.push(i);
    else if (name.startsWith('TARG')) groups.target.push(i);
    else if (name.startsWith('HK')) groups.housekeeping.push(i);
    else groups.other.push(i);
  });

  return groups;
}

/**
 * Build weight matrix from edge data
 */
export function buildWeightMatrix(
  edges: GRNEdge[],
  geneNames: string[],
  inhibitionMult: number
): { W: Float32Array; edgeInfo: { srcIdx: number; tgtIdx: number; weight: number; type: string }[] } {
  const nGenes = geneNames.length;
  const W = new Float32Array(nGenes * nGenes);
  const edgeInfo: { srcIdx: number; tgtIdx: number; weight: number; type: string }[] = [];

  edges.forEach(e => {
    const srcIdx = geneNames.indexOf(e.source);
    const tgtIdx = geneNames.indexOf(e.target);
    if (srcIdx >= 0 && tgtIdx >= 0) {
      const w = e.weight < 0 ? e.weight * inhibitionMult : e.weight;
      W[tgtIdx * nGenes + srcIdx] = w;
      edgeInfo.push({ srcIdx, tgtIdx, weight: e.weight, type: e.type });
    }
  });

  return { W, edgeInfo };
}

/**
 * Hill function for gene regulation
 */
function hill(x: number, k: number, n: number): number {
  const xPos = Math.max(x, 0);
  const xPow = Math.pow(xPos, n);
  const kPow = Math.pow(k, n);
  return xPow / (kPow + xPow + 1e-8);
}

/**
 * Inference engine for single-cell simulation
 */
export class InferenceEngine {
  private geneNames: string[];
  private nGenes: number;
  private groups: GeneGroups;
  private baseEdgeInfo: { srcIdx: number; tgtIdx: number; weight: number; type: string }[];

  constructor(grnData: GRNData) {
    this.geneNames = grnData.gene_names;
    this.nGenes = this.geneNames.length;
    this.groups = buildGeneGroups(this.geneNames);

    // Store base edge info (without inhibition multiplier applied)
    const edges = grnData.edges;
    this.baseEdgeInfo = [];
    edges.forEach(e => {
      const srcIdx = this.geneNames.indexOf(e.source);
      const tgtIdx = this.geneNames.indexOf(e.target);
      if (srcIdx >= 0 && tgtIdx >= 0) {
        this.baseEdgeInfo.push({ srcIdx, tgtIdx, weight: e.weight, type: e.type });
      }
    });
  }

  getGeneNames(): string[] {
    return this.geneNames;
  }

  getGeneGroups(): GeneGroups {
    return this.groups;
  }

  getNumGenes(): number {
    return this.nGenes;
  }

  /**
   * Get indices of genes that can be knocked out
   * (TFs, ligands, and receptors - not targets or housekeeping)
   */
  getKnockoutCandidates(): number[] {
    return [
      ...this.groups.tf_prog,
      ...this.groups.tf_lin,
      ...this.groups.ligand,
      ...this.groups.receptor,
    ];
  }

  /**
   * Run simulation with given parameters to steady state
   * Returns final expression state
   */
  runSimulation(
    params: SimulationParams,
    spatialAngle: number = Math.PI,  // Default: middle of tissue
    maxTime: number = 12,
    uniformMorphogens: boolean = false,  // If true, morphogens affect all cells equally
    dt: number = 0.1,
    noiseLevel: number = 0.01
  ): CellState {
    const { global: g, knockouts, morphogens, geneModifiers } = params;

    // Build weight matrix with current inhibition multiplier
    const W = new Float32Array(this.nGenes * this.nGenes);
    this.baseEdgeInfo.forEach(({ srcIdx, tgtIdx, weight }) => {
      const w = weight < 0 ? weight * g.inhibitionMult : weight;
      W[tgtIdx * this.nGenes + srcIdx] = w;
    });

    // Build bias and decay arrays
    const bias = new Float32Array(this.nGenes);
    const decay = new Float32Array(this.nGenes);

    this.groups.tf_prog.forEach(i => { bias[i] = g.progBias; decay[i] = g.progDecay; });
    this.groups.tf_lin.forEach(i => { bias[i] = g.linBias; decay[i] = g.linDecay; });
    this.groups.ligand.forEach(i => { bias[i] = g.ligandBias; decay[i] = 0.2; });
    this.groups.receptor.forEach(i => { bias[i] = g.receptorBias; decay[i] = 0.1; });
    this.groups.target.forEach(i => { bias[i] = 0.02; decay[i] = 0.25; });
    this.groups.housekeeping.forEach(i => { bias[i] = 0.5; decay[i] = 0.1; });
    this.groups.other.forEach(i => { bias[i] = 0.05; decay[i] = 0.2; });

    // Initialize expression
    const expression = new Float32Array(this.nGenes);
    this.groups.tf_prog.forEach(i => { expression[i] = g.initProgTF + Math.random() * 0.2; });
    this.groups.tf_lin.forEach(i => { expression[i] = g.initLinTF + Math.random() * 0.1; });
    this.groups.ligand.forEach(i => { expression[i] = 0.2 + Math.random() * 0.1; });
    this.groups.receptor.forEach(i => { expression[i] = 0.2 + Math.random() * 0.1; });
    this.groups.target.forEach(i => { expression[i] = 0.1 + Math.random() * 0.05; });
    this.groups.housekeeping.forEach(i => { expression[i] = 1.0 + Math.random() * 0.2; });
    this.groups.other.forEach(i => { expression[i] = 0.1 + Math.random() * 0.05; });

    // Apply knockouts
    knockouts.forEach(i => { expression[i] = 0; });

    // Lineage sector configuration (from trajectory.ts)
    const lineageSectorSizes = [1.5, 0.8, 1.2, 0.6, 1.0, 0.9];
    const totalSize = lineageSectorSizes.reduce((a, b) => a + b, 0);
    const lineageSectorStarts: number[] = [];
    let cumAngle = 0;
    lineageSectorSizes.forEach((size) => {
      lineageSectorStarts.push(cumAngle);
      cumAngle += (size / totalSize) * 2 * Math.PI;
    });

    const nLineages = 6;
    const tfsPerLineage = 2;

    // Simulation loop
    const h = new Float32Array(this.nGenes);
    const f = new Float32Array(this.nGenes);

    for (let time = 0; time < maxTime; time += dt) {
      // Compute morphogen signal for this cell's spatial position
      const morphogen = new Float32Array(this.nGenes);

      for (let lin = 0; lin < nLineages; lin++) {
        const m = morphogens[lin];
        if (!m || !m.enabled) continue;

        const receptorIdx = this.groups.receptor[lin % this.groups.receptor.length];
        if (receptorIdx !== undefined && knockouts.has(receptorIdx)) continue;

        const receptorExpr = receptorIdx !== undefined ? (expression[receptorIdx] ?? 0) : 1;
        if (receptorExpr < 0.1) continue;

        const tfIndices = this.groups.tf_lin.slice(lin * tfsPerLineage, (lin + 1) * tfsPerLineage);

        const sectorStart = lineageSectorStarts[lin] ?? 0;
        const sectorSize = (lineageSectorSizes[lin] ?? 1) / totalSize * 2 * Math.PI;
        const sectorCenter = sectorStart + sectorSize / 2;

        // Spatial strength: 1.0 if uniform, otherwise Gaussian falloff from sector center
        let spatialStrength = 1.0;
        if (!uniformMorphogens) {
          let angleDiff = Math.abs(spatialAngle - sectorCenter);
          if (angleDiff > Math.PI) angleDiff = 2 * Math.PI - angleDiff;
          const width = sectorSize / 2;
          spatialStrength = Math.exp(-(angleDiff * angleDiff) / (width * width * 0.5));
        }
        const timeStrength = Math.min(1, time / g.morphogenTime);
        const receptorGain = Math.min(1, receptorExpr / 0.5);
        const strength = spatialStrength * timeStrength * g.morphogenStrength * receptorGain * m.strength;

        tfIndices.forEach(i => {
          morphogen[i] = strength;
        });
      }

      // Compute Hill function on current expression
      for (let i = 0; i < this.nGenes; i++) {
        h[i] = hill(expression[i] ?? 0, g.hillK, g.hillN);
      }

      // Compute regulatory input
      for (let i = 0; i < this.nGenes; i++) {
        let input = bias[i] ?? 0;
        for (let j = 0; j < this.nGenes; j++) {
          input += (h[j] ?? 0) * (W[i * this.nGenes + j] ?? 0);
        }
        input += morphogen[i] ?? 0;

        // Apply gene modifier (simulates inhibitors/activators)
        // 0 = fully inhibited, 1 = normal, 2 = overexpressed
        const modifier = geneModifiers?.get(i) ?? 1.0;
        input *= modifier;

        f[i] = input - (decay[i] ?? 0) * (expression[i] ?? 0);
      }

      // Update expression with noise
      for (let i = 0; i < this.nGenes; i++) {
        if (knockouts.has(i)) {
          expression[i] = 0;
          continue;
        }
        const noise = (Math.random() - 0.5) * 2 * noiseLevel * Math.sqrt(dt);
        expression[i] = Math.max(0, Math.min(6, expression[i] + dt * f[i] + noise));
      }
    }

    // Determine lineage
    const lineage = this.assignLineage(expression);

    return { expression, lineage };
  }

  /**
   * Assign lineage based on expression state
   */
  private assignLineage(expression: Float32Array): number {
    const nLineages = 6;
    const tfsPerLineage = 2;
    const scores = new Float32Array(nLineages);

    for (let lin = 0; lin < nLineages; lin++) {
      const startIdx = lin * tfsPerLineage;
      const endIdx = (lin + 1) * tfsPerLineage;
      const tfIndices = this.groups.tf_lin.slice(startIdx, endIdx);
      let sum = 0;
      tfIndices.forEach(i => { sum += expression[i] ?? 0; });
      scores[lin] = sum / tfsPerLineage;
    }

    let maxScore = 0;
    let maxLin = 0;
    for (let lin = 0; lin < nLineages; lin++) {
      if (scores[lin] > maxScore) {
        maxScore = scores[lin];
        maxLin = lin;
      }
    }

    return maxScore > 1.2 ? maxLin + 1 : 0;
  }

  /**
   * Run multiple simulations with different spatial angles
   * to get average behavior across the tissue
   */
  runEnsemble(
    params: SimulationParams,
    nSamples: number = 10,
    maxTime: number = 12,
    uniformMorphogens: boolean = false
  ): CellState[] {
    const results: CellState[] = [];

    for (let i = 0; i < nSamples; i++) {
      const angle = (i / nSamples) * 2 * Math.PI;
      const state = this.runSimulation(params, angle, maxTime, uniformMorphogens);
      results.push(state);
    }

    return results;
  }

  /**
   * Run simulation for a specific lineage target
   * Uses spatial angle that should favor that lineage
   */
  runForLineage(
    params: SimulationParams,
    targetLineage: number,
    maxTime: number = 12,
    uniformMorphogens: boolean = false
  ): CellState {
    // Calculate ideal angle for target lineage
    const lineageSectorSizes = [1.5, 0.8, 1.2, 0.6, 1.0, 0.9];
    const totalSize = lineageSectorSizes.reduce((a, b) => a + b, 0);
    let cumAngle = 0;
    for (let i = 0; i < targetLineage - 1 && i < lineageSectorSizes.length; i++) {
      cumAngle += (lineageSectorSizes[i] / totalSize) * 2 * Math.PI;
    }
    const sectorSize = (lineageSectorSizes[targetLineage - 1] ?? 1) / totalSize * 2 * Math.PI;
    const idealAngle = cumAngle + sectorSize / 2;

    return this.runSimulation(params, idealAngle, maxTime, uniformMorphogens);
  }
}
