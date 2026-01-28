import grnData from '../synth_grn.json';

interface GRNEdge {
  source: string;
  target: string;
  weight: number;
  type: string;
}

interface GRNData {
  gene_names: string[];
  edges: GRNEdge[];
}

interface Cell {
  id: number;
  expression: Float32Array;
  lineage: number;
  birthTime: number;
  x: number;
  y: number;
  vx: number; // velocity for force-directed layout
  vy: number;
  trajectory: { x: number; y: number; time: number }[]; // position history
  spatialAngle: number; // Position in "tissue" that determines morphogen exposure
}

// Gene indices
const data = grnData as GRNData;
const geneNames = data.gene_names;
const nGenes = geneNames.length;

// Build gene groups
const geneGroups = {
  tf_prog: [] as number[],
  tf_lin: [] as number[],
  ligand: [] as number[],
  receptor: [] as number[],
  target: [] as number[],
  housekeeping: [] as number[],
  other: [] as number[],
};

geneNames.forEach((name, i) => {
  if (name.startsWith('TF_PROG')) geneGroups.tf_prog.push(i);
  else if (name.startsWith('TF_LIN')) geneGroups.tf_lin.push(i);
  else if (name.startsWith('LIG')) geneGroups.ligand.push(i);
  else if (name.startsWith('REC')) geneGroups.receptor.push(i);
  else if (name.startsWith('TARG')) geneGroups.target.push(i);
  else if (name.startsWith('HK')) geneGroups.housekeeping.push(i);
  else geneGroups.other.push(i);
});

// Knockout state - set of gene indices that are knocked out
const knockouts = new Set<number>();

// Morphogen enable state - which lineage morphogens are active (all enabled by default)
const morphogenEnabled = [true, true, true, true, true, true]; // One per lineage (0-5)

// Per-lineage morphogen strength multipliers (1.0 = normal, 2.0 = double strength, 0.5 = half)
const morphogenStrengths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // One per lineage (0-5)

// Uniform morphogen application - if true, morphogens affect all cells equally (no spatial gradient)
let uniformMorphogens = false;

// Per-gene activity modifiers (1.0 = normal, 0 = fully inhibited, 2.0 = activated)
// These modify the effective bias and incoming regulation for each gene
const geneModifiers: Record<number, number> = {}; // geneIndex -> modifier (default 1.0)

// Network edge control - disable specific edge types
const edgeTypeEnabled: Record<string, boolean> = {
  prog_to_lineage: true,  // Progenitor TFs activating lineage TFs (priming)
  prog_mutual: true,       // Progenitor mutual activation
  lineage_self: true,      // Lineage TF self-activation (positive feedback)
  lineage_partner: true,   // Lineage TF partner activation (within same lineage)
  lineage_inhibit: true,   // Cross-lineage inhibition
};

// Advanced simulation parameters (mutable via UI)
const params = {
  // Initial expression levels
  initProgTF: 1.5,
  initLinTF: 0, // Zero - lineage TFs start silent, require signal to activate
  initLigand: 0.2,
  initReceptor: 0.2,
  initTarget: 0.1,
  initHousekeeping: 1.0,
  // Biases
  progBias: 0.08,
  linBias: 0, // Zero - lineage TFs strictly require external signal to activate
  ligandBias: 0.05,
  receptorBias: 0.2,
  targetBias: 0.02,
  hkBias: 0.5,
  // Decay rates
  progDecay: 0.25,
  linDecay: 0.15,
  ligandDecay: 0.2,
  receptorDecay: 0.1,
  targetDecay: 0.25,
  hkDecay: 0.1,
  // Weight multiplier
  inhibitionMult: 2.5,
  // Morphogen
  morphogenTime: 4,
  morphogenStrength: 0.8,
  // Hill function
  hillN: 4,
  hillK: 1.0,
};

// Preset configurations
const presets: Record<string, Partial<typeof params>> = {
  default: { ...params },
  highPlasticity: {
    inhibitionMult: 1.5,
    progDecay: 0.15,
    linDecay: 0.1,
    morphogenStrength: 0.5,
  },
  strongCommitment: {
    inhibitionMult: 4.0,
    progDecay: 0.35,
    linDecay: 0.2,
    hillN: 6,
  },
  delayedDifferentiation: {
    morphogenTime: 8,
    morphogenStrength: 0.6,
    initProgTF: 2.0,
    progBias: 0.12,
  },
};

// Build weight matrix
const W = new Float32Array(nGenes * nGenes);
const baseWeights = new Float32Array(nGenes * nGenes); // Store original weights
const bias = new Float32Array(nGenes);
const decay = new Float32Array(nGenes);

// Store edge info for selective enabling/disabling
const edgeInfo: { srcIdx: number; tgtIdx: number; weight: number; type: string }[] = [];
data.edges.forEach(e => {
  const srcIdx = geneNames.indexOf(e.source);
  const tgtIdx = geneNames.indexOf(e.target);
  if (srcIdx >= 0 && tgtIdx >= 0) {
    baseWeights[tgtIdx * nGenes + srcIdx] = e.weight;
    edgeInfo.push({ srcIdx, tgtIdx, weight: e.weight, type: e.type });
  }
});

// Apply parameters to bias, decay, and weight arrays
function applyParameters(): void {
  // Reset weights
  W.fill(0);

  // Apply edges, respecting edge type enable/disable flags
  edgeInfo.forEach(({ srcIdx, tgtIdx, weight, type }) => {
    // Skip disabled edge types
    if (edgeTypeEnabled[type] === false) return;

    // Apply inhibition multiplier to negative weights
    const w = weight < 0 ? weight * params.inhibitionMult : weight;
    W[tgtIdx * nGenes + srcIdx] = w;
  });

  // Set biases and decay rates from params
  geneGroups.tf_prog.forEach(i => { bias[i] = params.progBias; decay[i] = params.progDecay; });
  geneGroups.tf_lin.forEach(i => { bias[i] = params.linBias; decay[i] = params.linDecay; });
  geneGroups.ligand.forEach(i => { bias[i] = params.ligandBias; decay[i] = params.ligandDecay; });
  geneGroups.receptor.forEach(i => { bias[i] = params.receptorBias; decay[i] = params.receptorDecay; });
  geneGroups.target.forEach(i => { bias[i] = params.targetBias; decay[i] = params.targetDecay; });
  geneGroups.housekeeping.forEach(i => { bias[i] = params.hkBias; decay[i] = params.hkDecay; });
  geneGroups.other.forEach(i => { bias[i] = 0.05; decay[i] = 0.2; });
}

// Initial application of parameters
applyParameters();

// Lineage colors
const lineageColors = [
  '#8b949e', // 0: progenitor
  '#f85149', // 1
  '#58a6ff', // 2
  '#3fb950', // 3
  '#a371f7', // 4
  '#d29922', // 5
  '#db61a2', // 6
];

const lineageNames = [
  'Progenitor',
  'Lineage 1',
  'Lineage 2',
  'Lineage 3',
  'Lineage 4',
  'Lineage 5',
  'Lineage 6',
];

// Simulation state
let cells: Cell[] = [];
let time = 0;
let isPlaying = false;
let animationId: number | null = null;
let lastFrameTime = 0;

// Reference simulation (default parameters, for comparison)
let referenceCells: Cell[] = [];
let showReference = true;
const referenceAlpha = 0.25; // Transparency for ghost cells

// Parameters
let nCells = 100;
let speed = 0.1;
let noiseLevel = 0.1;
let colorBy: 'lineage' | 'time' | 'gene' = 'lineage';
let selectedGene = 0;
let selectedGeneGroup: 'tf_prog' | 'tf_lin' | 'ligands' | 'receptors' | 'targets' | 'housekeeping' | 'other' | 'custom' = 'tf_prog';

let maxTime = 24; // hours (now mutable)
const dt = 0.1; // simulation timestep

// Canvas contexts
let cellCanvas: HTMLCanvasElement;
let cellCtx: CanvasRenderingContext2D;
let exprCanvas: HTMLCanvasElement;
let exprCtx: CanvasRenderingContext2D;
let proportionsCanvas: HTMLCanvasElement;
let proportionsCtx: CanvasRenderingContext2D;

// Dynamics panel state
let activeDynamicsTab: 'expression' | 'proportions' = 'expression';
const customSelectedGenes = new Set<number>();

// Lineage proportion history
interface ProportionSnapshot {
  time: number;
  counts: number[]; // Count per lineage (0-6, where 0 is undifferentiated)
  total: number;
}
const proportionHistory: ProportionSnapshot[] = [];

// Zoom and pan state for cell canvas
let cellZoom = 1.0;
let cellPanX = 0;
let cellPanY = 0;
let cellIsDragging = false;
let cellDragStartX = 0;
let cellDragStartY = 0;

// Zoom and pan state for expression canvas
let exprZoom = 1.0;
let exprPanX = 0;
let exprPanY = 0;
let exprIsDragging = false;
let exprDragStartX = 0;
let exprDragStartY = 0;

// Manual attractors for steering cells into unusual regions (visual aid)
interface Attractor {
  x: number;
  y: number;
  strength: number;
}
const manualAttractors: Attractor[] = [];
let attractorStrength = 0.3;
let positionBiasX = 0;
let positionBiasY = 0;
let attractorMode = false;

// External signals - direct input to the GRN equations (biologically meaningful)
// These add constant input terms to specific genes, simulating drugs/growth factors
const externalSignals: Record<number, number> = {}; // geneIndex -> signal strength (-1 to 1)

// Per-lineage external forcing - directly activate/inhibit lineage TF pairs
const lineageSignals: number[] = [0, 0, 0, 0, 0, 0]; // One signal per lineage (0 = none, + = activate, - = inhibit)

// Selected cell for detailed view (store ID, not reference)
let selectedCellId: number | null = null;
// Expression history for the selected cell (tracks expression over time)
interface CellExpressionSnapshot {
  time: number;
  expression: Float32Array;
  lineage: number;
}
const selectedCellHistory: CellExpressionSnapshot[] = [];

// Tracked genes for the selected cell expression plot
const trackedGenes: Set<number> = new Set();

// Helper to get selected cell from ID
function getSelectedCell(): Cell | null {
  if (selectedCellId === null) return null;
  return cells.find(c => c.id === selectedCellId) ?? null;
}

// Expression history for plotting
const expressionHistory: { time: number; means: Record<string, number[]> }[] = [];

// Helper functions
function hill(x: number, k: number, n: number): number {
  const xPos = Math.max(x, 0);
  const xPow = Math.pow(xPos, n);
  const kPow = Math.pow(k, n);
  return xPow / (kPow + xPow + 1e-8);
}

function getLineageColor(idx: number): string {
  return lineageColors[idx] ?? lineageColors[0]!;
}

function getLineageName(idx: number): string {
  return lineageNames[idx] ?? `Lineage ${idx}`;
}

function getGeneName(idx: number): string {
  return geneNames[idx] ?? `Gene_${idx}`;
}

// Default parameters for reference simulation (never modified)
const defaultParams = {
  initProgTF: 1.5,
  initLinTF: 0,
  initLigand: 0.2,
  initReceptor: 0.2,
  initTarget: 0.1,
  initHousekeeping: 1.0,
  progBias: 0.08,
  linBias: 0,
  ligandBias: 0.05,
  receptorBias: 0.2,
  targetBias: 0.02,
  hkBias: 0.5,
  progDecay: 0.25,
  linDecay: 0.15,
  ligandDecay: 0.2,
  receptorDecay: 0.1,
  targetDecay: 0.25,
  hkDecay: 0.1,
  inhibitionMult: 2.5,
  morphogenTime: 4,
  morphogenStrength: 0.8,
  hillN: 4,
  hillK: 1.0,
};

// Pre-compute default weights for reference simulation
const defaultW = new Float32Array(nGenes * nGenes);
const defaultBias = new Float32Array(nGenes);
const defaultDecay = new Float32Array(nGenes);

function initDefaultWeights(): void {
  defaultW.fill(0);
  edgeInfo.forEach(({ srcIdx, tgtIdx, weight }) => {
    const w = weight < 0 ? weight * defaultParams.inhibitionMult : weight;
    defaultW[tgtIdx * nGenes + srcIdx] = w;
  });
  geneGroups.tf_prog.forEach(i => { defaultBias[i] = defaultParams.progBias; defaultDecay[i] = defaultParams.progDecay; });
  geneGroups.tf_lin.forEach(i => { defaultBias[i] = defaultParams.linBias; defaultDecay[i] = defaultParams.linDecay; });
  geneGroups.ligand.forEach(i => { defaultBias[i] = defaultParams.ligandBias; defaultDecay[i] = defaultParams.ligandDecay; });
  geneGroups.receptor.forEach(i => { defaultBias[i] = defaultParams.receptorBias; defaultDecay[i] = defaultParams.receptorDecay; });
  geneGroups.target.forEach(i => { defaultBias[i] = defaultParams.targetBias; defaultDecay[i] = defaultParams.targetDecay; });
  geneGroups.housekeeping.forEach(i => { defaultBias[i] = defaultParams.hkBias; defaultDecay[i] = defaultParams.hkDecay; });
  geneGroups.other.forEach(i => { defaultBias[i] = 0.05; defaultDecay[i] = 0.2; });
}

function initReferenceCell(id: number, totalCells: number): Cell {
  const expression = new Float32Array(nGenes);
  // Use default params
  geneGroups.tf_prog.forEach(i => { expression[i] = defaultParams.initProgTF + Math.random() * 0.2; });
  geneGroups.tf_lin.forEach(i => { expression[i] = defaultParams.initLinTF + Math.random() * 0.1; });
  geneGroups.ligand.forEach(i => { expression[i] = defaultParams.initLigand + Math.random() * 0.1; });
  geneGroups.receptor.forEach(i => { expression[i] = defaultParams.initReceptor + Math.random() * 0.1; });
  geneGroups.target.forEach(i => { expression[i] = defaultParams.initTarget + Math.random() * 0.05; });
  geneGroups.housekeeping.forEach(i => { expression[i] = defaultParams.initHousekeeping + Math.random() * 0.2; });
  geneGroups.other.forEach(i => { expression[i] = 0.1 + Math.random() * 0.05; });

  const spatialAngle = (id / totalCells) * 2 * Math.PI + (Math.random() - 0.5) * 0.3;
  const initX = (Math.random() - 0.5) * 0.3;
  const initY = (Math.random() - 0.5) * 0.3 + 0.3;

  return {
    id,
    expression,
    lineage: 0,
    birthTime: 0,
    x: initX,
    y: initY,
    vx: 0,
    vy: 0,
    trajectory: [{ x: initX, y: initY, time: 0 }],
    spatialAngle,
  };
}

function simulateReferenceStep(cell: Cell, morphogen: Float32Array): void {
  const h = new Float32Array(nGenes);
  const f = new Float32Array(nGenes);

  for (let i = 0; i < nGenes; i++) {
    h[i] = hill(cell.expression[i] ?? 0, defaultParams.hillK, defaultParams.hillN);
  }

  for (let i = 0; i < nGenes; i++) {
    let input = defaultBias[i] ?? 0;
    for (let j = 0; j < nGenes; j++) {
      input += (h[j] ?? 0) * (defaultW[i * nGenes + j] ?? 0);
    }
    input += morphogen[i] ?? 0;
    f[i] = input - (defaultDecay[i] ?? 0) * (cell.expression[i] ?? 0);
  }

  for (let i = 0; i < nGenes; i++) {
    const noise = (Math.random() - 0.5) * 2 * noiseLevel * Math.sqrt(dt);
    cell.expression[i] = Math.max(0, Math.min(6, (cell.expression[i] ?? 0) + dt * (f[i] ?? 0) + noise));
  }
}

function computeReferenceCellPosition(cell: Cell): void {
  const progMean = geneGroups.tf_prog.reduce((s, i) => s + (cell.expression[i] ?? 0), 0) / geneGroups.tf_prog.length;
  const nLineages = 6;
  const tfsPerLineage = 2;

  let targetX = 0;
  let targetY = 0;

  for (let lin = 0; lin < nLineages; lin++) {
    const angle = ((lin - 2.5) / 5) * Math.PI * 0.7 - Math.PI / 2;
    const tfIndices = geneGroups.tf_lin.slice(lin * tfsPerLineage, (lin + 1) * tfsPerLineage);
    const score = tfIndices.reduce((s, i) => s + (cell.expression[i] ?? 0), 0) / tfsPerLineage;
    const strength = Math.pow(score, 1.5) * 0.3;
    targetX += Math.cos(angle) * strength;
    targetY += Math.sin(angle) * strength;
  }

  const progStrength = Math.pow(progMean, 1.5) * 0.4;
  targetY += progStrength;

  const attractionStrength = 0.15;
  cell.vx += (targetX - cell.x) * attractionStrength;
  cell.vy += (targetY - cell.y) * attractionStrength;
  cell.vx += (Math.random() - 0.5) * 0.02;
  cell.vy += (Math.random() - 0.5) * 0.02;
  cell.vx *= 0.85;
  cell.vy *= 0.85;
  cell.x += cell.vx;
  cell.y += cell.vy;

  const lastPoint = cell.trajectory[cell.trajectory.length - 1];
  if (!lastPoint || time - lastPoint.time >= 0.5) {
    cell.trajectory.push({ x: cell.x, y: cell.y, time });
    if (cell.trajectory.length > 100) cell.trajectory.shift();
  }
}

function initCell(id: number, totalCells: number): Cell {
  const expression = new Float32Array(nGenes);

  // Initialize expression from params
  geneGroups.tf_prog.forEach(i => { expression[i] = params.initProgTF + Math.random() * 0.2; });
  geneGroups.tf_lin.forEach(i => { expression[i] = params.initLinTF + Math.random() * 0.1; });
  geneGroups.ligand.forEach(i => { expression[i] = params.initLigand + Math.random() * 0.1; });
  geneGroups.receptor.forEach(i => { expression[i] = params.initReceptor + Math.random() * 0.1; });
  geneGroups.target.forEach(i => { expression[i] = params.initTarget + Math.random() * 0.05; });
  geneGroups.housekeeping.forEach(i => { expression[i] = params.initHousekeeping + Math.random() * 0.2; });
  geneGroups.other.forEach(i => { expression[i] = 0.1 + Math.random() * 0.05; });

  // Apply knockouts - set knocked out genes to 0
  knockouts.forEach(i => { expression[i] = 0; });

  // Assign spatial position (angle around a circle) - determines morphogen exposure
  // Distribute cells evenly around the circle with some noise
  const spatialAngle = (id / totalCells) * 2 * Math.PI + (Math.random() - 0.5) * 0.3;

  // Initial position: clustered near center (progenitor state) with small random offset
  const initX = (Math.random() - 0.5) * 0.3;
  const initY = (Math.random() - 0.5) * 0.3 + 0.3; // Slightly above center (progenitor region)

  return {
    id,
    expression,
    lineage: 0,
    birthTime: 0,
    x: initX,
    y: initY,
    vx: 0,
    vy: 0,
    trajectory: [{ x: initX, y: initY, time: 0 }],
    spatialAngle,
  };
}

function simulateStep(cell: Cell, morphogen: Float32Array): void {
  const h = new Float32Array(nGenes);
  const f = new Float32Array(nGenes);

  // Hill function on expression using params
  for (let i = 0; i < nGenes; i++) {
    h[i] = hill(cell.expression[i] ?? 0, params.hillK, params.hillN);
  }

  // Compute regulatory input
  for (let i = 0; i < nGenes; i++) {
    let input = bias[i] ?? 0;
    for (let j = 0; j < nGenes; j++) {
      const hVal = h[j] ?? 0;
      const wVal = W[i * nGenes + j] ?? 0;
      input += hVal * wVal;
    }
    input += morphogen[i] ?? 0;

    // Apply gene modifier (simulates inhibitors/activators)
    const modifier = geneModifiers[i] ?? 1.0;
    input *= modifier;

    // Add external signal (direct input to gene, like drug/growth factor)
    // This is mathematically part of the ODE: dx/dt = input + externalSignal - decay*x
    const extSignal = externalSignals[i] ?? 0;
    input += extSignal;

    const decayVal = decay[i] ?? 0;
    const exprVal = cell.expression[i] ?? 0;
    f[i] = input - decayVal * exprVal;
  }

  // Update expression with noise
  for (let i = 0; i < nGenes; i++) {
    // Knocked out genes stay at 0
    if (knockouts.has(i)) {
      cell.expression[i] = 0;
      continue;
    }
    const noise = (Math.random() - 0.5) * 2 * noiseLevel * Math.sqrt(dt);
    const fVal = f[i] ?? 0;
    const oldVal = cell.expression[i] ?? 0;
    cell.expression[i] = Math.max(0, Math.min(6, oldVal + dt * fVal + noise));
  }
}

function assignLineage(cell: Cell): void {
  // Compute lineage scores based on lineage TF expression
  const nLineages = 6;
  const tfsPerLineage = 2;
  const scores = new Float32Array(nLineages);

  for (let lin = 0; lin < nLineages; lin++) {
    const startIdx = lin * tfsPerLineage;
    const endIdx = (lin + 1) * tfsPerLineage;
    const tfIndices = geneGroups.tf_lin.slice(startIdx, endIdx);
    let sum = 0;
    tfIndices.forEach(i => { sum += cell.expression[i] ?? 0; });
    scores[lin] = sum / tfsPerLineage;
  }

  // Find max score
  let maxScore = 0;
  let maxLin = 0;
  for (let lin = 0; lin < nLineages; lin++) {
    const score = scores[lin] ?? 0;
    if (score > maxScore) {
      maxScore = score;
      maxLin = lin;
    }
  }

  // Assign lineage if above threshold
  cell.lineage = maxScore > 1.2 ? maxLin + 1 : 0;
}

function computeCellPosition(cell: Cell): void {
  // Force-directed layout based on expression state
  const progMean = geneGroups.tf_prog.reduce((s, i) => s + (cell.expression[i] ?? 0), 0) / geneGroups.tf_prog.length;
  const nLineages = 6;
  const tfsPerLineage = 2;

  // Compute target position based on expression (attractor)
  let targetX = 0;
  let targetY = 0;

  // Lineage-specific attractors arranged in a fan below the progenitor region
  for (let lin = 0; lin < nLineages; lin++) {
    // Lineages fan out at bottom: angles from -60° to +60° below center
    const angle = ((lin - 2.5) / 5) * Math.PI * 0.7 - Math.PI / 2; // Fan from -63° to +63°
    const tfIndices = geneGroups.tf_lin.slice(lin * tfsPerLineage, (lin + 1) * tfsPerLineage);
    const score = tfIndices.reduce((s, i) => s + (cell.expression[i] ?? 0), 0) / tfsPerLineage;

    // Attraction strength based on lineage TF expression
    const strength = Math.pow(score, 1.5) * 0.3;
    targetX += Math.cos(angle) * strength;
    targetY += Math.sin(angle) * strength;
  }

  // Progenitor attractor at top center
  const progStrength = Math.pow(progMean, 1.5) * 0.4;
  targetY += progStrength; // Pull toward top when progenitor genes are high

  // Apply force toward target (spring-like attraction)
  const attractionStrength = 0.15;
  const dx = targetX - cell.x;
  const dy = targetY - cell.y;
  cell.vx += dx * attractionStrength;
  cell.vy += dy * attractionStrength;

  // Apply manual attractors (for steering cells into unusual regions)
  manualAttractors.forEach(attr => {
    const adx = attr.x - cell.x;
    const ady = attr.y - cell.y;
    const dist = Math.sqrt(adx * adx + ady * ady) + 0.01;
    // Inverse-distance attraction with falloff
    const force = attr.strength * attractorStrength / (1 + dist * 2);
    cell.vx += (adx / dist) * force;
    cell.vy += (ady / dist) * force;
  });

  // Apply direct position bias (global push in X/Y direction)
  cell.vx += positionBiasX * 0.05;
  cell.vy += positionBiasY * 0.05;

  // Add small random force for exploration
  cell.vx += (Math.random() - 0.5) * 0.02;
  cell.vy += (Math.random() - 0.5) * 0.02;

  // Damping
  const damping = 0.85;
  cell.vx *= damping;
  cell.vy *= damping;

  // Update position
  cell.x += cell.vx;
  cell.y += cell.vy;

  // Record trajectory (every 0.5 time units)
  const lastPoint = cell.trajectory[cell.trajectory.length - 1];
  if (!lastPoint || time - lastPoint.time >= 0.5) {
    cell.trajectory.push({ x: cell.x, y: cell.y, time });
    // Keep trajectory length manageable
    if (cell.trajectory.length > 100) {
      cell.trajectory.shift();
    }
  }
}

function resetSimulation(): void {
  time = 0;
  cells = [];
  referenceCells = [];
  expressionHistory.length = 0;
  proportionHistory.length = 0;
  stepAccumulator = 0;
  selectedCellId = null;
  selectedCellHistory.length = 0;
  trackedGenes.clear();

  // Apply current parameters before creating cells
  applyParameters();

  // Initialize default weights for reference simulation
  initDefaultWeights();

  for (let i = 0; i < nCells; i++) {
    cells.push(initCell(i, nCells));
    referenceCells.push(initReferenceCell(i, nCells));
  }

  // Initial positions
  cells.forEach(cell => {
    computeCellPosition(cell);
  });
  referenceCells.forEach(cell => {
    computeReferenceCellPosition(cell);
  });

  updateDisplay();
}

// Asymmetric lineage sector sizes (some lineages have larger domains)
const lineageSectorSizes = [1.5, 0.8, 1.2, 0.6, 1.0, 0.9]; // Relative sizes
const lineageSectorStarts: number[] = [];
let cumAngle = 0;
const totalSize = lineageSectorSizes.reduce((a, b) => a + b, 0);
lineageSectorSizes.forEach((size, i) => {
  lineageSectorStarts.push(cumAngle);
  cumAngle += (size / totalSize) * 2 * Math.PI;
});

// Lineage-specific proliferation rates (some lineages grow faster)
const lineageProlifRates = [0.02, 0.015, 0.025, 0.01, 0.018, 0.012];

function divideCell(parent: Cell): Cell {
  const child: Cell = {
    id: cells.length,
    expression: new Float32Array(parent.expression),
    lineage: parent.lineage,
    birthTime: time,
    x: parent.x + (Math.random() - 0.5) * 0.05, // Slight offset from parent
    y: parent.y + (Math.random() - 0.5) * 0.05,
    vx: 0,
    vy: 0,
    trajectory: [{ x: parent.x, y: parent.y, time }], // Start trajectory from parent position
    spatialAngle: parent.spatialAngle + (Math.random() - 0.5) * 0.2, // Slight spatial noise
  };
  // Add noise to child expression
  for (let i = 0; i < nGenes; i++) {
    const val = child.expression[i] ?? 0;
    child.expression[i] = Math.max(0, val + (Math.random() - 0.5) * 0.1);
  }
  return child;
}

function simulationStep(): void {
  const nLineages = 6;
  const tfsPerLineage = 2;

  // Cell proliferation - committed cells can divide
  const newCells: Cell[] = [];
  const maxCells = nCells * 5; // Cap at 5x initial population

  if (cells.length < maxCells && time > 2) {
    cells.forEach(cell => {
      if (cell.lineage > 0) {
        // Committed cells can divide based on lineage-specific rate
        const rate = lineageProlifRates[cell.lineage - 1] ?? 0.01;
        if (Math.random() < rate * dt) {
          newCells.push(divideCell(cell));
        }
      } else if (time > 6) {
        // Progenitors divide slowly after initial phase
        if (Math.random() < 0.005 * dt) {
          newCells.push(divideCell(cell));
        }
      }
    });
  }
  cells.push(...newCells);

  // Simulate each cell with cell-specific morphogen based on spatial position
  cells.forEach(cell => {
    // Create morphogen signal specific to this cell's spatial position
    const morphogen = new Float32Array(nGenes);

    // Asymmetric lineage sectors - different sizes create different lineage proportions
    // Morphogens now work through receptors: Morphogen -> Receptor -> Lineage TF
    for (let lin = 0; lin < nLineages; lin++) {
      // Skip if this morphogen is disabled
      if (!morphogenEnabled[lin]) continue;

      // Get the receptor for this lineage (1 receptor per lineage, cycling if fewer receptors)
      const receptorIdx = geneGroups.receptor[lin % geneGroups.receptor.length];

      // Skip if receptor is knocked out (expression will be 0, can't transduce signal)
      if (receptorIdx !== undefined && knockouts.has(receptorIdx)) continue;

      // Check if receptor is expressed in this cell (must be > threshold to transduce)
      const receptorExpr = receptorIdx !== undefined ? (cell.expression[receptorIdx] ?? 0) : 1;
      if (receptorExpr < 0.1) continue; // Receptor must be expressed to transduce signal

      const tfIndices = geneGroups.tf_lin.slice(lin * tfsPerLineage, (lin + 1) * tfsPerLineage);

      // Get sector center and width
      const sectorStart = lineageSectorStarts[lin] ?? 0;
      const sectorSize = (lineageSectorSizes[lin] ?? 1) / totalSize * 2 * Math.PI;
      const sectorCenter = sectorStart + sectorSize / 2;

      // Spatial strength: 1.0 if uniform, otherwise Gaussian falloff from sector center
      let spatialStrength = 1.0;
      if (!uniformMorphogens) {
        // Distance from cell to this lineage's sector center (wrapped)
        let angleDiff = Math.abs(cell.spatialAngle - sectorCenter);
        if (angleDiff > Math.PI) angleDiff = 2 * Math.PI - angleDiff;

        // Morphogen strength falls off with angular distance
        // Wider sectors have broader morphogen gradients
        const width = sectorSize / 2;
        spatialStrength = Math.exp(-(angleDiff * angleDiff) / (width * width * 0.5));
      }

      // Time-dependent component - morphogen activates gradually
      const timeStrength = Math.min(1, time / params.morphogenTime);

      // Signal strength is modulated by receptor expression level and per-lineage multiplier
      const receptorGain = Math.min(1, receptorExpr / 0.5); // Saturates at receptor expr = 0.5
      const lineageMultiplier = morphogenStrengths[lin] ?? 1.0;
      const strength = spatialStrength * timeStrength * params.morphogenStrength * receptorGain * lineageMultiplier;
      tfIndices.forEach(i => {
        morphogen[i] = strength;
      });
    }

    // Add external lineage signals - direct forcing of lineage TFs (bypasses spatial/receptor constraints)
    // This simulates adding synthetic activators/inhibitors that directly affect lineage TF expression
    for (let lin = 0; lin < nLineages; lin++) {
      const signal = lineageSignals[lin] ?? 0;
      if (signal === 0) continue;
      const tfIndices = geneGroups.tf_lin.slice(lin * tfsPerLineage, (lin + 1) * tfsPerLineage);
      tfIndices.forEach(i => {
        // Add to morphogen signal (positive = activate, negative = inhibit)
        morphogen[i] = (morphogen[i] ?? 0) + signal * 0.5;
      });
    }

    simulateStep(cell, morphogen);
    assignLineage(cell);
    computeCellPosition(cell);
  });

  // Record expression history - capture ALL genes in each group
  if (expressionHistory.length === 0 || time - (expressionHistory[expressionHistory.length - 1]?.time ?? 0) >= 0.5) {
    const means: Record<string, number[]> = {};

    means.tf_prog = geneGroups.tf_prog.map(i => {
      return cells.reduce((s, c) => s + (c.expression[i] ?? 0), 0) / cells.length;
    });

    means.tf_lin = geneGroups.tf_lin.map(i => {
      return cells.reduce((s, c) => s + (c.expression[i] ?? 0), 0) / cells.length;
    });

    means.ligand = geneGroups.ligand.map(i => {
      return cells.reduce((s, c) => s + (c.expression[i] ?? 0), 0) / cells.length;
    });

    means.receptor = geneGroups.receptor.map(i => {
      return cells.reduce((s, c) => s + (c.expression[i] ?? 0), 0) / cells.length;
    });

    means.target = geneGroups.target.map(i => {
      return cells.reduce((s, c) => s + (c.expression[i] ?? 0), 0) / cells.length;
    });

    means.housekeeping = geneGroups.housekeeping.map(i => {
      return cells.reduce((s, c) => s + (c.expression[i] ?? 0), 0) / cells.length;
    });

    means.other = geneGroups.other.map(i => {
      return cells.reduce((s, c) => s + (c.expression[i] ?? 0), 0) / cells.length;
    });

    // Also store raw means by gene index for custom mode
    means.raw = geneNames.map((_, i) => {
      return cells.reduce((s, c) => s + (c.expression[i] ?? 0), 0) / cells.length;
    });

    expressionHistory.push({ time, means });

    // Record lineage proportions
    const counts = new Array(7).fill(0) as number[];
    cells.forEach(cell => {
      const current = counts[cell.lineage];
      if (current !== undefined) {
        counts[cell.lineage] = current + 1;
      }
    });
    proportionHistory.push({ time, counts, total: cells.length });
  }

  // Simulate reference cells (with default parameters, no external signals)
  if (showReference) {
    referenceCells.forEach(cell => {
      // Create morphogen signal for reference cell (using default params)
      const morphogen = new Float32Array(nGenes);

      for (let lin = 0; lin < nLineages; lin++) {
        const receptorIdx = geneGroups.receptor[lin % geneGroups.receptor.length];
        const receptorExpr = receptorIdx !== undefined ? (cell.expression[receptorIdx] ?? 0) : 1;
        if (receptorExpr < 0.1) continue;

        const tfIndices = geneGroups.tf_lin.slice(lin * tfsPerLineage, (lin + 1) * tfsPerLineage);
        const sectorStart = lineageSectorStarts[lin] ?? 0;
        const sectorSize = (lineageSectorSizes[lin] ?? 1) / totalSize * 2 * Math.PI;
        const sectorCenter = sectorStart + sectorSize / 2;

        // Spatial strength: 1.0 if uniform, otherwise Gaussian falloff from sector center
        let spatialStrength = 1.0;
        if (!uniformMorphogens) {
          let angleDiff = Math.abs(cell.spatialAngle - sectorCenter);
          if (angleDiff > Math.PI) angleDiff = 2 * Math.PI - angleDiff;
          const width = sectorSize / 2;
          spatialStrength = Math.exp(-(angleDiff * angleDiff) / (width * width * 0.5));
        }
        const timeStrength = Math.min(1, time / defaultParams.morphogenTime);
        const receptorGain = Math.min(1, receptorExpr / 0.5);
        const strength = spatialStrength * timeStrength * defaultParams.morphogenStrength * receptorGain;

        tfIndices.forEach(i => {
          morphogen[i] = strength;
        });
      }

      simulateReferenceStep(cell, morphogen);
      // Assign lineage for reference cell
      const scores = new Float32Array(nLineages);
      for (let lin = 0; lin < nLineages; lin++) {
        const tfIndices = geneGroups.tf_lin.slice(lin * tfsPerLineage, (lin + 1) * tfsPerLineage);
        let sum = 0;
        tfIndices.forEach(i => { sum += cell.expression[i] ?? 0; });
        scores[lin] = sum / tfsPerLineage;
      }
      let maxScore = 0;
      let maxLin = 0;
      for (let lin = 0; lin < nLineages; lin++) {
        if ((scores[lin] ?? 0) > maxScore) {
          maxScore = scores[lin] ?? 0;
          maxLin = lin;
        }
      }
      cell.lineage = maxScore > 1.2 ? maxLin + 1 : 0;

      computeReferenceCellPosition(cell);
    });
  }

  // Track selected cell expression history
  const selectedCell = getSelectedCell();
  if (selectedCell) {
    // Record snapshot every 0.5 time units
    const lastSnapshot = selectedCellHistory[selectedCellHistory.length - 1];
    if (!lastSnapshot || time - lastSnapshot.time >= 0.5) {
      selectedCellHistory.push({
        time,
        expression: new Float32Array(selectedCell.expression),
        lineage: selectedCell.lineage,
      });
      // Keep history manageable
      if (selectedCellHistory.length > 100) {
        selectedCellHistory.shift();
      }
    }
  }

  time += dt;
}

function getCellColor(cell: Cell): string {
  if (colorBy === 'lineage') {
    return getLineageColor(cell.lineage);
  } else if (colorBy === 'time') {
    const t = Math.min(1, time / maxTime);
    const r = Math.round(88 + t * 100);
    const g = Math.round(166 - t * 80);
    const b = Math.round(255 - t * 100);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Gene expression
    const expr = cell.expression[selectedGene] ?? 0;
    const normalized = Math.min(1, expr / 3);
    const r = Math.round(13 + normalized * 242);
    const g = Math.round(17 + normalized * 168);
    const b = Math.round(23 + normalized * 50);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

function drawCells(): void {
  // Skip if panel is collapsed
  const cellRow = document.getElementById('row-cell-state');
  if (cellRow?.classList.contains('collapsed')) return;

  const width = cellCanvas.width / (window.devicePixelRatio || 1);
  const height = cellCanvas.height / (window.devicePixelRatio || 1);
  if (width <= 0 || height <= 0) return;

  cellCtx.fillStyle = '#0d1117';
  cellCtx.fillRect(0, 0, width, height);

  // Base bounds (fixed coordinate system for consistency)
  let minX = -1.5, maxX = 1.5, minY = -1.5, maxY = 1.5;

  // Apply zoom - smaller zoom means zoomed out (see more), larger zoom means zoomed in
  const zoomedRangeX = (maxX - minX) / cellZoom;
  const zoomedRangeY = (maxY - minY) / cellZoom;
  const centerX = (minX + maxX) / 2 + cellPanX;
  const centerY = (minY + maxY) / 2 + cellPanY;

  minX = centerX - zoomedRangeX / 2;
  maxX = centerX + zoomedRangeX / 2;
  minY = centerY - zoomedRangeY / 2;
  maxY = centerY + zoomedRangeY / 2;

  // Draw grid
  cellCtx.strokeStyle = '#21262d';
  cellCtx.lineWidth = 1;

  for (let i = 0; i <= 10; i++) {
    const x = (i / 10) * width;
    const y = (i / 10) * height;
    cellCtx.beginPath();
    cellCtx.moveTo(x, 0);
    cellCtx.lineTo(x, height);
    cellCtx.stroke();
    cellCtx.beginPath();
    cellCtx.moveTo(0, y);
    cellCtx.lineTo(width, y);
    cellCtx.stroke();
  }

  const padding = 40;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  // Helper to convert cell coordinates to screen coordinates
  const toScreen = (x: number, y: number) => ({
    px: padding + ((x - minX) / (maxX - minX)) * plotWidth,
    py: padding + (1 - (y - minY) / (maxY - minY)) * plotHeight
  });

  // Draw reference cells first (ghost simulation with default params)
  if (showReference && referenceCells.length > 0) {
    // Draw reference trajectories
    referenceCells.forEach(cell => {
      if (cell.trajectory.length < 2) return;

      cellCtx.beginPath();
      cellCtx.strokeStyle = '#444'; // Muted gray for reference
      cellCtx.lineWidth = 1;
      cellCtx.globalAlpha = referenceAlpha * 0.5;

      cell.trajectory.forEach((point, i) => {
        const { px, py } = toScreen(point.x, point.y);
        if (i === 0) {
          cellCtx.moveTo(px, py);
        } else {
          cellCtx.lineTo(px, py);
        }
      });

      const { px: currPx, py: currPy } = toScreen(cell.x, cell.y);
      cellCtx.lineTo(currPx, currPy);
      cellCtx.stroke();
      cellCtx.globalAlpha = 1;
    });

    // Draw reference cells
    referenceCells.forEach(cell => {
      const { px, py } = toScreen(cell.x, cell.y);

      cellCtx.beginPath();
      cellCtx.arc(px, py, 3, 0, Math.PI * 2);
      cellCtx.globalAlpha = referenceAlpha;
      cellCtx.fillStyle = getLineageColor(cell.lineage);
      cellCtx.fill();
      cellCtx.globalAlpha = 1;
    });
  }

  // Draw trajectories first (behind cells)
  cells.forEach(cell => {
    if (cell.trajectory.length < 2) return;

    const color = getCellColor(cell);
    cellCtx.beginPath();
    cellCtx.strokeStyle = color;
    cellCtx.lineWidth = 1;
    cellCtx.globalAlpha = 0.4;

    cell.trajectory.forEach((point, i) => {
      const { px, py } = toScreen(point.x, point.y);
      if (i === 0) {
        cellCtx.moveTo(px, py);
      } else {
        cellCtx.lineTo(px, py);
      }
    });

    // Connect to current position
    const { px: currPx, py: currPy } = toScreen(cell.x, cell.y);
    cellCtx.lineTo(currPx, currPy);
    cellCtx.stroke();
    cellCtx.globalAlpha = 1;
  });

  // Draw cells on top
  cells.forEach(cell => {
    const { px, py } = toScreen(cell.x, cell.y);
    const isSelected = selectedCellId !== null && cell.id === selectedCellId;

    cellCtx.beginPath();
    cellCtx.arc(px, py, isSelected ? 6 : 4, 0, Math.PI * 2);
    cellCtx.fillStyle = getCellColor(cell);
    cellCtx.fill();
    cellCtx.strokeStyle = isSelected ? '#ffffff' : '#0d1117';
    cellCtx.lineWidth = isSelected ? 2 : 1;
    cellCtx.stroke();

    // Draw selection ring for selected cell
    if (isSelected) {
      cellCtx.beginPath();
      cellCtx.arc(px, py, 10, 0, Math.PI * 2);
      cellCtx.strokeStyle = '#ffffff';
      cellCtx.lineWidth = 2;
      cellCtx.setLineDash([3, 3]);
      cellCtx.stroke();
      cellCtx.setLineDash([]);
    }
  });

  // Draw lineage attractor labels
  const nLineages = 6;
  cellCtx.font = '10px sans-serif';
  cellCtx.fillStyle = '#8b949e';
  cellCtx.textAlign = 'center';

  for (let lin = 0; lin < nLineages; lin++) {
    const angle = ((lin - 2.5) / 5) * Math.PI * 0.7 - Math.PI / 2;
    const labelDist = 0.8;
    const lx = Math.cos(angle) * labelDist;
    const ly = Math.sin(angle) * labelDist;
    const { px, py } = toScreen(lx, ly);
    cellCtx.fillText(`L${lin + 1}`, px, py);
  }

  // Draw progenitor label at top
  const { px: progPx, py: progPy } = toScreen(0, 0.6);
  cellCtx.fillText('Prog', progPx, progPy - 10);

  // Draw manual attractors
  manualAttractors.forEach((attr, i) => {
    const { px, py } = toScreen(attr.x, attr.y);

    // Draw attractor marker (target symbol)
    cellCtx.strokeStyle = '#db61a2';
    cellCtx.lineWidth = 2;

    // Outer circle
    cellCtx.beginPath();
    cellCtx.arc(px, py, 12, 0, Math.PI * 2);
    cellCtx.stroke();

    // Inner circle
    cellCtx.beginPath();
    cellCtx.arc(px, py, 6, 0, Math.PI * 2);
    cellCtx.stroke();

    // Crosshair
    cellCtx.beginPath();
    cellCtx.moveTo(px - 16, py);
    cellCtx.lineTo(px + 16, py);
    cellCtx.moveTo(px, py - 16);
    cellCtx.lineTo(px, py + 16);
    cellCtx.stroke();

    // Number label
    cellCtx.fillStyle = '#db61a2';
    cellCtx.font = 'bold 10px sans-serif';
    cellCtx.fillText(`A${i + 1}`, px + 14, py - 10);
  });
}

function drawExpression(): void {
  // Skip if panel is collapsed
  const exprRow = document.getElementById('row-gene-expr');
  if (exprRow?.classList.contains('collapsed')) return;

  const dpr = window.devicePixelRatio || 1;
  const width = exprCanvas.width / dpr;
  const height = exprCanvas.height / dpr;
  if (width <= 0 || height <= 0) return;

  // Reset transform and clear entire canvas
  exprCtx.setTransform(1, 0, 0, 1, 0, 0);
  exprCtx.clearRect(0, 0, exprCanvas.width, exprCanvas.height);
  exprCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  exprCtx.fillStyle = '#0d1117';
  exprCtx.fillRect(0, 0, width, height);

  if (expressionHistory.length < 2) return;

  const padding = { left: 50, right: 20, top: 20, bottom: 30 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Draw axes
  exprCtx.strokeStyle = '#30363d';
  exprCtx.lineWidth = 1;
  exprCtx.beginPath();
  exprCtx.moveTo(padding.left, padding.top);
  exprCtx.lineTo(padding.left, height - padding.bottom);
  exprCtx.lineTo(width - padding.right, height - padding.bottom);
  exprCtx.stroke();

  // Axis labels
  exprCtx.fillStyle = '#8b949e';
  exprCtx.font = '10px sans-serif';
  exprCtx.textAlign = 'center';
  exprCtx.fillText('Time (hours)', width / 2, height - 5);

  exprCtx.save();
  exprCtx.translate(12, height / 2);
  exprCtx.rotate(-Math.PI / 2);
  exprCtx.fillText('Expression', 0, 0);
  exprCtx.restore();

  // Map UI names to data keys
  const getDataKey = (group: string): string => {
    switch (group) {
      case 'ligands': return 'ligand';
      case 'receptors': return 'receptor';
      case 'targets': return 'target';
      default: return group;
    }
  };

  // Auto-scale expression axis based on data
  let baseMaxExpr = 2;
  const key = getDataKey(selectedGeneGroup);
  if (selectedGeneGroup === 'custom') {
    // For custom mode, check raw data for selected genes
    expressionHistory.forEach(h => {
      const raw = h.means.raw;
      if (raw) {
        customSelectedGenes.forEach(geneIdx => {
          const v = raw[geneIdx];
          if (v !== undefined && v > baseMaxExpr) baseMaxExpr = v;
        });
      }
    });
  } else {
    expressionHistory.forEach(h => {
      const means = h.means[key];
      if (means) {
        means.forEach(v => { if (v > baseMaxExpr) baseMaxExpr = v; });
      }
    });
  }
  baseMaxExpr = Math.ceil(baseMaxExpr * 1.1); // Add 10% headroom

  // Apply zoom to time and expression ranges
  const baseMaxT = Math.max(time, 1);
  const timeRange = baseMaxT / exprZoom;
  const exprRange = baseMaxExpr / exprZoom;

  // Calculate visible ranges with pan offset
  const timeCenterBase = baseMaxT / 2;
  const exprCenterBase = baseMaxExpr / 2;

  const minT = Math.max(0, timeCenterBase + exprPanX - timeRange / 2);
  const maxT = minT + timeRange;
  const minExpr = Math.max(0, exprCenterBase - exprPanY - exprRange / 2);
  const maxExpr = minExpr + exprRange;

  // Time scale labels
  const timeStep = Math.max(1, Math.ceil(timeRange / 6));
  for (let t = Math.ceil(minT / timeStep) * timeStep; t <= maxT; t += timeStep) {
    const x = padding.left + ((t - minT) / (maxT - minT)) * plotWidth;
    if (x >= padding.left && x <= width - padding.right) {
      exprCtx.fillStyle = '#8b949e';
      exprCtx.fillText(String(Math.round(t)), x, height - padding.bottom + 15);
    }
  }

  // Expression scale labels
  const exprStep = Math.max(0.5, Math.ceil(exprRange / 4 * 2) / 2);
  for (let e = Math.ceil(minExpr / exprStep) * exprStep; e <= maxExpr; e += exprStep) {
    const y = height - padding.bottom - ((e - minExpr) / (maxExpr - minExpr)) * plotHeight;
    if (y >= padding.top && y <= height - padding.bottom) {
      exprCtx.fillStyle = '#8b949e';
      exprCtx.textAlign = 'right';
      exprCtx.fillText(e.toFixed(1), padding.left - 5, y + 3);
    }
  }

  // Get genes to plot - expanded color palette for more genes
  let genesToPlot: number[];
  let geneLabels: string[];
  const geneColors = [
    '#f85149', '#58a6ff', '#3fb950', '#a371f7', '#d29922', '#db61a2', // 6 base colors
    '#ff7b72', '#79c0ff', '#7ee787', '#d2a8ff', '#e3b341', '#ff9bce', // 6 lighter variants
    '#ffa657', '#a5d6ff', '#aff5b4', '#cab8ff', '#f0c674', '#ffadda', // 6 more variants
    '#ffc658', '#56d4dd', '#2ea043', '#8957e5'  // 4 extra
  ];

  switch (selectedGeneGroup) {
    case 'tf_prog':
      genesToPlot = geneGroups.tf_prog;
      geneLabels = genesToPlot.map(i => getGeneName(i));
      break;
    case 'tf_lin':
      genesToPlot = geneGroups.tf_lin;
      geneLabels = genesToPlot.map(i => getGeneName(i));
      break;
    case 'ligands':
      genesToPlot = geneGroups.ligand;
      geneLabels = genesToPlot.map(i => getGeneName(i));
      break;
    case 'receptors':
      genesToPlot = geneGroups.receptor;
      geneLabels = genesToPlot.map(i => getGeneName(i));
      break;
    case 'targets':
      genesToPlot = geneGroups.target.slice(0, 20); // First 20 targets
      geneLabels = genesToPlot.map(i => getGeneName(i));
      break;
    case 'housekeeping':
      genesToPlot = geneGroups.housekeeping.slice(0, 20); // First 20 HK genes
      geneLabels = genesToPlot.map(i => getGeneName(i));
      break;
    case 'other':
      genesToPlot = geneGroups.other;
      geneLabels = genesToPlot.map(i => getGeneName(i));
      break;
    case 'custom':
      genesToPlot = Array.from(customSelectedGenes);
      geneLabels = genesToPlot.map(i => getGeneName(i));
      break;
    default:
      genesToPlot = geneGroups.tf_prog;
      geneLabels = genesToPlot.map(i => getGeneName(i));
  }

  // For custom mode, we need to draw using raw expression data
  const isCustomMode = selectedGeneGroup === 'custom';

  // Draw expression lines - no limit on gene count
  const dataKey = getDataKey(selectedGeneGroup);
  genesToPlot.forEach((geneAbsIdx, plotIdx) => {
    exprCtx.beginPath();
    exprCtx.strokeStyle = geneColors[plotIdx % geneColors.length] ?? '#8b949e';
    exprCtx.lineWidth = 1.5;

    let started = false;
    expressionHistory.forEach((h) => {
      let val: number | undefined;
      if (isCustomMode) {
        // For custom mode, use raw data with absolute gene index
        const raw = h.means.raw;
        val = raw?.[geneAbsIdx];
      } else {
        // For group mode, use group data with relative index
        const means = h.means[dataKey];
        val = means?.[plotIdx];
      }
      if (val === undefined) return;

      // Use zoom-adjusted coordinate system
      const x = padding.left + ((h.time - minT) / (maxT - minT)) * plotWidth;
      const y = height - padding.bottom - ((val - minExpr) / (maxExpr - minExpr)) * plotHeight;

      // Clip to plot area
      if (h.time < minT || h.time > maxT) return;

      if (!started) {
        exprCtx.moveTo(x, y);
        started = true;
      } else {
        exprCtx.lineTo(x, y);
      }
    });

    exprCtx.stroke();
  });

  // Draw legend - compact layout for many genes
  const legendX = width - padding.right - 80;
  const legendY = padding.top + 5;
  const legendCols = genesToPlot.length > 10 ? 2 : 1;
  const legendItemsPerCol = Math.ceil(genesToPlot.length / legendCols);

  geneLabels.forEach((label, i) => {
    const col = Math.floor(i / legendItemsPerCol);
    const row = i % legendItemsPerCol;
    const x = legendX - col * 75;
    const y = legendY + row * 11;
    exprCtx.fillStyle = geneColors[i % geneColors.length] ?? '#8b949e';
    exprCtx.fillRect(x, y - 3, 8, 8);
    exprCtx.fillStyle = '#e6edf3';
    exprCtx.font = '8px sans-serif';
    exprCtx.textAlign = 'left';
    exprCtx.fillText(label.replace('TF_', '').replace('LIG_', 'L').replace('TARG_', 'T'), x + 10, y + 4);
  });
}

function drawProportions(): void {
  // Skip if panel is collapsed or tab is not active
  const exprRow = document.getElementById('row-gene-expr');
  if (exprRow?.classList.contains('collapsed')) return;
  if (activeDynamicsTab !== 'proportions') return;

  // Ensure canvas is properly sized
  const dpr = window.devicePixelRatio || 1;
  const rect = proportionsCanvas.getBoundingClientRect();
  const canvasW = Math.floor(rect.width * dpr);
  const canvasH = Math.floor(rect.height * dpr);

  if (canvasW > 0 && canvasH > 0 && (proportionsCanvas.width !== canvasW || proportionsCanvas.height !== canvasH)) {
    proportionsCanvas.width = canvasW;
    proportionsCanvas.height = canvasH;
  }

  const width = proportionsCanvas.width / dpr;
  const height = proportionsCanvas.height / dpr;
  if (width <= 0 || height <= 0) return;

  // Reset transform and clear entire canvas
  proportionsCtx.setTransform(1, 0, 0, 1, 0, 0);
  proportionsCtx.clearRect(0, 0, proportionsCanvas.width, proportionsCanvas.height);
  proportionsCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  proportionsCtx.fillStyle = '#0d1117';
  proportionsCtx.fillRect(0, 0, width, height);

  if (proportionHistory.length < 2) return;

  const padding = { left: 50, right: 90, top: 20, bottom: 30 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Draw axes
  proportionsCtx.strokeStyle = '#30363d';
  proportionsCtx.lineWidth = 1;
  proportionsCtx.beginPath();
  proportionsCtx.moveTo(padding.left, padding.top);
  proportionsCtx.lineTo(padding.left, height - padding.bottom);
  proportionsCtx.lineTo(width - padding.right, height - padding.bottom);
  proportionsCtx.stroke();

  // Axis labels
  proportionsCtx.fillStyle = '#8b949e';
  proportionsCtx.font = '10px sans-serif';
  proportionsCtx.textAlign = 'center';
  proportionsCtx.fillText('Time (hours)', (padding.left + width - padding.right) / 2, height - 5);

  proportionsCtx.save();
  proportionsCtx.translate(12, height / 2);
  proportionsCtx.rotate(-Math.PI / 2);
  proportionsCtx.fillText('Cell Count', 0, 0);
  proportionsCtx.restore();

  // Calculate time range (use zoom/pan from expr canvas)
  const baseMaxT = Math.max(time, 1);
  const timeRange = baseMaxT / exprZoom;
  const timeCenterBase = baseMaxT / 2;
  const minT = Math.max(0, timeCenterBase + exprPanX - timeRange / 2);
  const maxT = minT + timeRange;

  // Find max count for Y scale
  let maxCount = 10;
  proportionHistory.forEach(h => {
    h.counts.forEach(c => { if (c > maxCount) maxCount = c; });
  });
  maxCount = Math.ceil(maxCount * 1.1);

  // Time scale labels
  const timeStep = Math.max(1, Math.ceil(timeRange / 6));
  for (let t = Math.ceil(minT / timeStep) * timeStep; t <= maxT; t += timeStep) {
    const x = padding.left + ((t - minT) / (maxT - minT)) * plotWidth;
    if (x >= padding.left && x <= width - padding.right) {
      proportionsCtx.fillStyle = '#8b949e';
      proportionsCtx.textAlign = 'center';
      proportionsCtx.fillText(String(Math.round(t)), x, height - padding.bottom + 15);
    }
  }

  // Count scale labels
  const countStep = Math.max(5, Math.ceil(maxCount / 5 / 5) * 5);
  for (let c = 0; c <= maxCount; c += countStep) {
    const y = height - padding.bottom - (c / maxCount) * plotHeight;
    proportionsCtx.fillStyle = '#8b949e';
    proportionsCtx.textAlign = 'right';
    proportionsCtx.fillText(String(c), padding.left - 5, y + 3);
  }

  // Draw lines for each lineage with clear colors and thick lines
  const nLineages = 7; // 0 = undiff + 6 lineages

  for (let lin = 0; lin < nLineages; lin++) {
    proportionsCtx.beginPath();
    proportionsCtx.strokeStyle = getLineageColor(lin);
    proportionsCtx.lineWidth = 2.5;
    proportionsCtx.lineCap = 'round';
    proportionsCtx.lineJoin = 'round';

    let started = false;

    proportionHistory.forEach((h) => {
      if (h.time < minT || h.time > maxT) return;

      const x = padding.left + ((h.time - minT) / (maxT - minT)) * plotWidth;
      const count = h.counts[lin] ?? 0;
      const y = height - padding.bottom - (count / maxCount) * plotHeight;

      if (!started) {
        proportionsCtx.moveTo(x, y);
        started = true;
      } else {
        proportionsCtx.lineTo(x, y);
      }
    });

    proportionsCtx.stroke();
  }

  // Draw legend on the right with clear formatting
  const legendX = width - padding.right + 8;
  const legendY = padding.top + 5;

  lineageNames.forEach((name, i) => {
    const y = legendY + i * 16;
    proportionsCtx.fillStyle = getLineageColor(i);
    proportionsCtx.fillRect(legendX, y - 4, 10, 10);
    proportionsCtx.fillStyle = '#e6edf3';
    proportionsCtx.font = '9px sans-serif';
    proportionsCtx.textAlign = 'left';
    proportionsCtx.fillText(name, legendX + 14, y + 4);
  });
}

function updateLineageBars(): void {
  const counts = new Array(7).fill(0) as number[];
  cells.forEach(cell => {
    const current = counts[cell.lineage];
    if (current !== undefined) {
      counts[cell.lineage] = current + 1;
    }
  });

  const container = document.getElementById('lineage-bars')!;
  container.replaceChildren();

  counts.forEach((count, i) => {
    const bar = document.createElement('div');
    bar.className = 'lineage-bar';

    const color = document.createElement('span');
    color.className = 'lineage-color';
    color.style.background = getLineageColor(i);

    const name = document.createElement('span');
    name.className = 'lineage-name';
    name.textContent = getLineageName(i);

    const countSpan = document.createElement('span');
    countSpan.className = 'lineage-count';
    countSpan.textContent = String(count);

    const fill = document.createElement('div');
    fill.className = 'lineage-fill';

    const fillInner = document.createElement('div');
    fillInner.className = 'lineage-fill-inner';
    fillInner.style.width = `${(count / nCells) * 100}%`;
    fillInner.style.background = getLineageColor(i);
    fill.appendChild(fillInner);

    bar.appendChild(color);
    bar.appendChild(name);
    bar.appendChild(countSpan);
    bar.appendChild(fill);
    container.appendChild(bar);
  });
}

function updateDisplay(): void {
  // Update time display
  document.getElementById('time-display')!.textContent = time.toFixed(1);
  const progress = (time / maxTime) * 100;
  document.getElementById('progress-fill')!.style.width = `${Math.min(100, progress)}%`;

  // Update cell count
  const cellCountEl = document.getElementById('cell-count');
  if (cellCountEl) cellCountEl.textContent = String(cells.length);

  // Draw (no resize here - only on window resize/panel toggle)
  drawCells();
  drawExpression();
  drawProportions();
  updateLineageBars();

  // Update selected cell details and expression chart
  if (selectedCellId !== null) {
    if (isPlaying) {
      updateCellDetails(); // Only rebuild HTML during play to avoid flicker
    }
    drawSelectedCellExpression();
  }
}

function resizeCanvases(): void {
  const dpr = window.devicePixelRatio || 1;

  // Check if cell state panel is collapsed
  const cellRow = document.getElementById('row-cell-state');
  if (cellRow && !cellRow.classList.contains('collapsed')) {
    // Get canvas's computed size from CSS flex layout
    const cellRect = cellCanvas.getBoundingClientRect();
    const cellW = Math.floor(cellRect.width * dpr);
    const cellH = Math.floor(cellRect.height * dpr);

    if (cellW > 0 && cellH > 0 && (cellCanvas.width !== cellW || cellCanvas.height !== cellH)) {
      cellCanvas.width = cellW;
      cellCanvas.height = cellH;
      cellCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  // Check if gene expression panel is collapsed
  const exprRow = document.getElementById('row-gene-expr');
  if (exprRow && !exprRow.classList.contains('collapsed')) {
    // Get canvas's computed size from CSS flex layout
    const exprRect = exprCanvas.getBoundingClientRect();
    const exprW = Math.floor(exprRect.width * dpr);
    const exprH = Math.floor(exprRect.height * dpr);

    if (exprW > 0 && exprH > 0 && (exprCanvas.width !== exprW || exprCanvas.height !== exprH)) {
      exprCanvas.width = exprW;
      exprCanvas.height = exprH;
      exprCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    // Also resize proportions canvas (same panel)
    const propRect = proportionsCanvas.getBoundingClientRect();
    const propW = Math.floor(propRect.width * dpr);
    const propH = Math.floor(propRect.height * dpr);

    if (propW > 0 && propH > 0 && (proportionsCanvas.width !== propW || proportionsCanvas.height !== propH)) {
      proportionsCanvas.width = propW;
      proportionsCanvas.height = propH;
      proportionsCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  // Check if selected cell panel is collapsed
  const selRow = document.getElementById('row-selected-cell');
  if (selRow && !selRow.classList.contains('collapsed')) {
    const selCanvas = document.getElementById('selected-cell-canvas') as HTMLCanvasElement;
    if (selCanvas) {
      const selContainer = selCanvas.parentElement;
      if (selContainer) {
        const selRect = selContainer.getBoundingClientRect();
        const selW = Math.floor(selRect.width * dpr);
        const selH = Math.floor(selRect.height * dpr);

        if (selW > 0 && selH > 0 && (selCanvas.width !== selW || selCanvas.height !== selH)) {
          selCanvas.width = selW;
          selCanvas.height = selH;
          selCanvas.style.width = `${selRect.width}px`;
          selCanvas.style.height = `${selRect.height}px`;
          const selCtx = selCanvas.getContext('2d');
          if (selCtx) selCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
      }
    }
  }
}

// Accumulator for fractional steps when speed < 1
let stepAccumulator = 0;

function animate(timestamp: number): void {
  if (!isPlaying) return;

  const elapsed = timestamp - lastFrameTime;

  if (elapsed >= 16) { // ~60fps
    lastFrameTime = timestamp;

    if (time < maxTime) {
      // Speed controls simulation rate
      // speed=1 means 1 step per frame, speed=0.1 means 1 step every 10 frames
      stepAccumulator += speed;
      const stepsThisFrame = Math.floor(stepAccumulator);
      stepAccumulator -= stepsThisFrame;

      for (let i = 0; i < stepsThisFrame && time < maxTime; i++) {
        simulationStep();
      }
      updateDisplay();
    } else {
      stopSimulation();
    }
  }

  animationId = requestAnimationFrame(animate);
}

function startSimulation(): void {
  if (isPlaying) return;
  isPlaying = true;
  lastFrameTime = performance.now();
  animationId = requestAnimationFrame(animate);

  const btn = document.getElementById('btn-play')!;
  btn.textContent = '⏸ Pause';
  btn.classList.add('playing');
}

function stopSimulation(): void {
  isPlaying = false;
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  const btn = document.getElementById('btn-play')!;
  btn.textContent = '▶ Play';
  btn.classList.remove('playing');
}

function setupCustomGeneSelector(): void {
  const container = document.getElementById('custom-gene-checkboxes');
  if (!container) return;

  const groups = [
    { name: 'Progenitor TFs', genes: geneGroups.tf_prog },
    { name: 'Lineage TFs', genes: geneGroups.tf_lin },
    { name: 'Ligands', genes: geneGroups.ligand },
    { name: 'Receptors', genes: geneGroups.receptor },
    { name: 'Targets', genes: geneGroups.target.slice(0, 10) },
    { name: 'Housekeeping', genes: geneGroups.housekeeping.slice(0, 5) },
  ];

  container.replaceChildren();

  groups.forEach(group => {
    const groupDiv = document.createElement('div');
    groupDiv.className = 'gene-group';

    const titleDiv = document.createElement('div');
    titleDiv.className = 'gene-group-title';
    titleDiv.textContent = group.name;
    groupDiv.appendChild(titleDiv);

    const itemsDiv = document.createElement('div');
    itemsDiv.className = 'gene-items';

    group.genes.forEach(geneIdx => {
      const name = (geneNames[geneIdx] ?? '').replace('TF_PROG_', 'P').replace('TF_LIN_', 'L').replace('LIG_', '').replace('REC_', 'R').replace('TARG_', 'T').replace('HK_', 'H');

      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.dataset.gene = String(geneIdx);
      checkbox.checked = customSelectedGenes.has(geneIdx);

      checkbox.addEventListener('change', () => {
        if (checkbox.checked) {
          customSelectedGenes.add(geneIdx);
        } else {
          customSelectedGenes.delete(geneIdx);
        }
        updateDisplay();
      });

      const span = document.createElement('span');
      span.textContent = name;

      label.appendChild(checkbox);
      label.appendChild(span);
      itemsDiv.appendChild(label);
    });

    groupDiv.appendChild(itemsDiv);
    container.appendChild(groupDiv);
  });

  // Clear button
  document.getElementById('btn-clear-custom-genes')?.addEventListener('click', () => {
    customSelectedGenes.clear();
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      (cb as HTMLInputElement).checked = false;
    });
    updateDisplay();
  });

  // All TFs button
  document.getElementById('btn-select-all-tfs')?.addEventListener('click', () => {
    geneGroups.tf_prog.forEach(i => customSelectedGenes.add(i));
    geneGroups.tf_lin.forEach(i => customSelectedGenes.add(i));
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      const geneIdx = parseInt((cb as HTMLInputElement).dataset.gene ?? '-1', 10);
      if (geneGroups.tf_prog.includes(geneIdx) || geneGroups.tf_lin.includes(geneIdx)) {
        (cb as HTMLInputElement).checked = true;
      }
    });
    updateDisplay();
  });
}

function setupLegend(): void {
  const legend = document.getElementById('legend')!;
  legend.replaceChildren();

  lineageNames.forEach((name, i) => {
    const item = document.createElement('div');
    item.className = 'legend-item';

    const color = document.createElement('span');
    color.className = 'legend-color';
    color.style.background = getLineageColor(i);

    const label = document.createElement('span');
    label.textContent = name;

    item.appendChild(color);
    item.appendChild(label);
    legend.appendChild(item);
  });
}

function setupGeneSelect(): void {
  const select = document.getElementById('gene-select') as HTMLSelectElement;
  geneNames.forEach((name, i) => {
    const option = document.createElement('option');
    option.value = String(i);
    option.textContent = name;
    select.appendChild(option);
  });

  select.addEventListener('change', () => {
    selectedGene = parseInt(select.value, 10);
    updateDisplay();
  });
}

function setupSteeringControls(): void {
  // Lineage signal sliders (direct GRN manipulation)
  for (let lin = 0; lin < 6; lin++) {
    const slider = document.getElementById(`lin-signal-${lin}`) as HTMLInputElement;
    const valEl = document.getElementById(`val-lin-signal-${lin}`);
    if (slider && valEl) {
      slider.addEventListener('input', () => {
        const val = parseFloat(slider.value);
        lineageSignals[lin] = val;
        valEl.textContent = val.toFixed(1);
      });
    }
  }

  // Reset lineage signals button
  const resetLinSignalsBtn = document.getElementById('btn-reset-lin-signals');
  if (resetLinSignalsBtn) {
    resetLinSignalsBtn.addEventListener('click', () => {
      for (let lin = 0; lin < 6; lin++) {
        lineageSignals[lin] = 0;
        const slider = document.getElementById(`lin-signal-${lin}`) as HTMLInputElement;
        const valEl = document.getElementById(`val-lin-signal-${lin}`);
        if (slider) slider.value = '0';
        if (valEl) valEl.textContent = '0.0';
      }
    });
  }

  // Position bias sliders
  const biasXSlider = document.getElementById('bias-x') as HTMLInputElement;
  const biasYSlider = document.getElementById('bias-y') as HTMLInputElement;
  const biasXVal = document.getElementById('val-bias-x');
  const biasYVal = document.getElementById('val-bias-y');

  if (biasXSlider && biasXVal) {
    biasXSlider.addEventListener('input', () => {
      positionBiasX = parseFloat(biasXSlider.value);
      biasXVal.textContent = positionBiasX.toFixed(1);
    });
  }

  if (biasYSlider && biasYVal) {
    biasYSlider.addEventListener('input', () => {
      positionBiasY = parseFloat(biasYSlider.value);
      biasYVal.textContent = positionBiasY.toFixed(1);
    });
  }

  // Attractor strength slider
  const strengthSlider = document.getElementById('attractor-strength') as HTMLInputElement;
  const strengthVal = document.getElementById('val-attractor-strength');
  if (strengthSlider && strengthVal) {
    strengthSlider.addEventListener('input', () => {
      attractorStrength = parseFloat(strengthSlider.value);
      strengthVal.textContent = attractorStrength.toFixed(2);
    });
  }

  // Attractor mode toggle
  const attractorModeBtn = document.getElementById('btn-attractor-mode');
  if (attractorModeBtn) {
    attractorModeBtn.addEventListener('click', () => {
      attractorMode = !attractorMode;
      attractorModeBtn.classList.toggle('active', attractorMode);
      cellCanvas.classList.toggle('attractor-mode', attractorMode);
      cellCanvas.style.cursor = attractorMode ? 'crosshair' : 'grab';
    });
  }

  // Clear attractors button
  const clearAttractorsBtn = document.getElementById('btn-clear-attractors');
  if (clearAttractorsBtn) {
    clearAttractorsBtn.addEventListener('click', () => {
      manualAttractors.length = 0;
      updateAttractorCount();
      updateDisplay();
    });
  }
}

function updateAttractorCount(): void {
  const countEl = document.getElementById('attractor-count');
  if (countEl) {
    countEl.textContent = manualAttractors.length > 0
      ? `${manualAttractors.length} attractor${manualAttractors.length > 1 ? 's' : ''} placed`
      : '';
  }
}

function screenToCell(clientX: number, clientY: number): { x: number; y: number } {
  const rect = cellCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const canvasWidth = cellCanvas.width / dpr;
  const canvasHeight = cellCanvas.height / dpr;
  const padding = 40;
  const plotWidth = canvasWidth - padding * 2;
  const plotHeight = canvasHeight - padding * 2;

  // Get current view bounds (same as in drawCells)
  let minX = -1.5, maxX = 1.5, minY = -1.5, maxY = 1.5;
  const zoomedRangeX = (maxX - minX) / cellZoom;
  const zoomedRangeY = (maxY - minY) / cellZoom;
  const centerX = (minX + maxX) / 2 + cellPanX;
  const centerY = (minY + maxY) / 2 + cellPanY;
  minX = centerX - zoomedRangeX / 2;
  maxX = centerX + zoomedRangeX / 2;
  minY = centerY - zoomedRangeY / 2;
  maxY = centerY + zoomedRangeY / 2;

  // Convert screen position to cell coordinates
  const px = clientX - rect.left;
  const py = clientY - rect.top;
  const x = minX + ((px - padding) / plotWidth) * (maxX - minX);
  const y = maxY - ((py - padding) / plotHeight) * (maxY - minY);

  return { x, y };
}

function findNearestCell(x: number, y: number, maxDistance: number = 0.15): Cell | null {
  let nearest: Cell | null = null;
  let minDist = maxDistance;

  cells.forEach(cell => {
    const dx = cell.x - x;
    const dy = cell.y - y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < minDist) {
      minDist = dist;
      nearest = cell;
    }
  });

  return nearest;
}

function selectCell(cell: Cell | null): void {
  selectedCellId = cell ? cell.id : null;
  selectedCellHistory.length = 0;

  // Update label
  const label = document.getElementById('selected-cell-label');
  if (label) {
    label.textContent = cell ? `Cell ${cell.id} - ${getLineageName(cell.lineage)}` : 'No cell selected';
    label.style.color = cell ? getLineageColor(cell.lineage) : '';
  }

  if (cell) {
    // Record initial snapshot
    selectedCellHistory.push({
      time,
      expression: new Float32Array(cell.expression),
      lineage: cell.lineage,
    });
    // Auto-track some default genes
    if (trackedGenes.size === 0) {
      // Add first 2 progenitor TFs and first 2 lineage TFs by default
      geneGroups.tf_prog.slice(0, 2).forEach(i => trackedGenes.add(i));
      geneGroups.tf_lin.slice(0, 4).forEach(i => trackedGenes.add(i));
    }
  }

  updateCellDetails();
  updateGeneSelector();
  drawSelectedCellExpression();
  updateDisplay();
}

function updateCellDetails(): void {
  const container = document.getElementById('cell-details')!;
  const cell = getSelectedCell();

  if (!cell) {
    container.innerHTML = '<p class="placeholder">Click on a cell to select it</p>';
    return;
  }

  // Build expression table for key genes
  let html = `
    <div class="cell-info">
      <div class="cell-info-row">
        <span class="cell-info-label">Cell ID:</span>
        <span class="cell-info-value">${cell.id}</span>
      </div>
      <div class="cell-info-row">
        <span class="cell-info-label">Lineage:</span>
        <span class="cell-info-value" style="color: ${getLineageColor(cell.lineage)}">${getLineageName(cell.lineage)}</span>
      </div>
      <div class="cell-info-row">
        <span class="cell-info-label">Age:</span>
        <span class="cell-info-value">${(time - cell.birthTime).toFixed(1)}h</span>
      </div>
    </div>
    <div class="cell-expression">
      <div class="expression-title">Expression Levels</div>
      <div class="expression-group">
        <div class="expression-group-title">Progenitor TFs</div>
        ${geneGroups.tf_prog.map(i => {
          const val = cell.expression[i] ?? 0;
          const name = (geneNames[i] ?? '').replace('TF_PROG_', 'P');
          return `<div class="expr-row"><span>${name}</span><div class="expr-bar"><div class="expr-fill" style="width: ${Math.min(100, val / 3 * 100)}%; background: var(--accent-cyan)"></div></div><span class="expr-val">${val.toFixed(2)}</span></div>`;
        }).join('')}
      </div>
      <div class="expression-group">
        <div class="expression-group-title">Lineage TFs</div>
        ${geneGroups.tf_lin.slice(0, 6).map(i => {
          const val = cell.expression[i] ?? 0;
          const name = (geneNames[i] ?? '').replace('TF_LIN_', 'L');
          const linIdx = Math.floor(geneGroups.tf_lin.indexOf(i) / 2) + 1;
          return `<div class="expr-row"><span>${name}</span><div class="expr-bar"><div class="expr-fill" style="width: ${Math.min(100, val / 3 * 100)}%; background: ${getLineageColor(linIdx)}"></div></div><span class="expr-val">${val.toFixed(2)}</span></div>`;
        }).join('')}
      </div>
    </div>
    <button id="btn-deselect-cell" class="deselect-btn">Deselect</button>
  `;

  container.innerHTML = html;

  // Add deselect handler
  document.getElementById('btn-deselect-cell')?.addEventListener('click', () => {
    selectCell(null);
  });
}

// Draw expression over time for the selected cell
let selectedCellCanvas: HTMLCanvasElement | null = null;
let selectedCellCtx: CanvasRenderingContext2D | null = null;

function drawSelectedCellExpression(): void {
  // Skip if panel is collapsed
  const selRow = document.getElementById('row-selected-cell');
  if (selRow?.classList.contains('collapsed')) return;

  if (!selectedCellCanvas || !selectedCellCtx) {
    selectedCellCanvas = document.getElementById('selected-cell-canvas') as HTMLCanvasElement;
    if (!selectedCellCanvas) return;
    selectedCellCtx = selectedCellCanvas.getContext('2d');
    if (!selectedCellCtx) return;
  }

  const dpr = window.devicePixelRatio || 1;
  const container = selectedCellCanvas.parentElement;
  if (container) {
    const rect = container.getBoundingClientRect();
    const w = Math.floor(rect.width * dpr);
    const h = Math.floor(rect.height * dpr);
    if (w > 0 && h > 0 && (selectedCellCanvas.width !== w || selectedCellCanvas.height !== h)) {
      selectedCellCanvas.width = w;
      selectedCellCanvas.height = h;
      selectedCellCanvas.style.width = `${rect.width}px`;
      selectedCellCanvas.style.height = `${rect.height}px`;
    }
  }

  const ctx = selectedCellCtx;
  const width = selectedCellCanvas.width / dpr;
  const height = selectedCellCanvas.height / dpr;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, selectedCellCanvas.width, selectedCellCanvas.height);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, width, height);

  if (trackedGenes.size === 0) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(selectedCellId !== null ? 'Select genes to track →' : 'Select a cell to track', width / 2, height / 2);
    return;
  }

  if (selectedCellHistory.length === 0) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No data yet', width / 2, height / 2);
    return;
  }

  const padding = { left: 40, right: 10, top: 10, bottom: 25 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Draw axes
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  // Find time and expression ranges
  const minT = selectedCellHistory[0]?.time ?? 0;
  const maxT = selectedCellHistory[selectedCellHistory.length - 1]?.time ?? 1;
  let maxExpr = 2;
  selectedCellHistory.forEach(snap => {
    trackedGenes.forEach(gi => {
      const v = snap.expression[gi] ?? 0;
      if (v > maxExpr) maxExpr = v;
    });
  });
  maxExpr = Math.ceil(maxExpr * 1.1);

  // Time labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`${minT.toFixed(1)}h`, padding.left, height - 5);
  ctx.fillText(`${maxT.toFixed(1)}h`, width - padding.right, height - 5);

  // Expression labels
  ctx.textAlign = 'right';
  ctx.fillText('0', padding.left - 3, height - padding.bottom);
  ctx.fillText(maxExpr.toFixed(1), padding.left - 3, padding.top + 8);

  // Draw lines for each tracked gene
  const geneColors = [
    '#f85149', '#58a6ff', '#3fb950', '#a371f7', '#d29922', '#db61a2',
    '#ff7b72', '#79c0ff', '#7ee787', '#d2a8ff', '#e3b341', '#ff9bce',
  ];
  let colorIdx = 0;
  trackedGenes.forEach(geneIdx => {
    ctx.beginPath();
    ctx.strokeStyle = geneColors[colorIdx % geneColors.length] ?? '#8b949e';
    ctx.lineWidth = 1.5;
    colorIdx++;

    let started = false;
    selectedCellHistory.forEach(snap => {
      const val = snap.expression[geneIdx] ?? 0;
      const x = padding.left + ((snap.time - minT) / (maxT - minT + 0.01)) * plotWidth;
      const y = height - padding.bottom - (val / maxExpr) * plotHeight;

      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });

  // Draw legend
  ctx.font = '8px sans-serif';
  ctx.textAlign = 'left';
  colorIdx = 0;
  let legendX = padding.left + 5;
  let legendY = padding.top + 10;
  trackedGenes.forEach(geneIdx => {
    const name = (geneNames[geneIdx] ?? '').replace('TF_PROG_', 'P').replace('TF_LIN_', 'L').replace('LIG_', 'Lg').replace('REC_', 'R');
    ctx.fillStyle = geneColors[colorIdx % geneColors.length] ?? '#8b949e';
    ctx.fillRect(legendX, legendY - 6, 8, 8);
    ctx.fillStyle = '#e6edf3';
    ctx.fillText(name, legendX + 10, legendY);
    legendX += 45;
    if (legendX > width - 50) {
      legendX = padding.left + 5;
      legendY += 12;
    }
    colorIdx++;
  });
}

function updateGeneSelector(): void {
  const container = document.getElementById('gene-selector-panel');
  if (!container) return;

  // Build gene list grouped by type
  const groups = [
    { name: 'Progenitor TFs', genes: geneGroups.tf_prog },
    { name: 'Lineage TFs', genes: geneGroups.tf_lin },
    { name: 'Ligands', genes: geneGroups.ligand },
    { name: 'Receptors', genes: geneGroups.receptor },
  ];

  let html = '<div class="gene-selector-groups">';
  groups.forEach(group => {
    html += `<div class="gene-group"><div class="gene-group-title">${group.name}</div><div class="gene-checkboxes">`;
    group.genes.forEach(geneIdx => {
      const name = (geneNames[geneIdx] ?? '').replace('TF_PROG_', 'P').replace('TF_LIN_', 'L').replace('LIG_', '').replace('REC_', '');
      const checked = trackedGenes.has(geneIdx) ? 'checked' : '';
      html += `<label class="gene-checkbox"><input type="checkbox" data-gene="${geneIdx}" ${checked}><span>${name}</span></label>`;
    });
    html += '</div></div>';
  });
  html += '</div>';
  html += '<div class="gene-selector-actions"><button id="btn-clear-tracked">Clear All</button><button id="btn-track-all-tf">All TFs</button></div>';

  container.innerHTML = html;

  // Add event listeners
  container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', (e) => {
      const geneIdx = parseInt((e.target as HTMLInputElement).dataset.gene ?? '-1', 10);
      if (geneIdx >= 0) {
        if ((e.target as HTMLInputElement).checked) {
          trackedGenes.add(geneIdx);
        } else {
          trackedGenes.delete(geneIdx);
        }
        drawSelectedCellExpression();
      }
    });
  });

  document.getElementById('btn-clear-tracked')?.addEventListener('click', () => {
    trackedGenes.clear();
    updateGeneSelector();
    drawSelectedCellExpression();
  });

  document.getElementById('btn-track-all-tf')?.addEventListener('click', () => {
    geneGroups.tf_prog.forEach(i => trackedGenes.add(i));
    geneGroups.tf_lin.forEach(i => trackedGenes.add(i));
    updateGeneSelector();
    drawSelectedCellExpression();
  });
}

function setupCanvasZoomPan(): void {
  // Cell canvas zoom and pan
  cellCanvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    cellZoom = Math.max(0.1, Math.min(10, cellZoom * zoomFactor));
    updateDisplay();
  });

  let cellDidDrag = false; // Track if actual dragging occurred

  cellCanvas.addEventListener('mousedown', (e) => {
    if (attractorMode) {
      // Place attractor
      const pos = screenToCell(e.clientX, e.clientY);
      manualAttractors.push({ x: pos.x, y: pos.y, strength: 1.0 });
      updateAttractorCount();
      updateDisplay();
      return;
    }
    cellIsDragging = true;
    cellDidDrag = false;
    cellDragStartX = e.clientX;
    cellDragStartY = e.clientY;
    cellCanvas.style.cursor = 'grabbing';
  });

  cellCanvas.addEventListener('mousemove', (e) => {
    if (!cellIsDragging) return;
    const dx = e.clientX - cellDragStartX;
    const dy = e.clientY - cellDragStartY;
    // Only count as drag if moved more than a few pixels
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      cellDidDrag = true;
    }
    // Convert pixel movement to coordinate movement (use same scale for both axes)
    const dpr = window.devicePixelRatio || 1;
    const canvasWidth = cellCanvas.width / dpr;
    const canvasHeight = cellCanvas.height / dpr;
    // Use consistent scale: 3 units across the view
    const scaleX = 3 / cellZoom / canvasWidth;
    const scaleY = 3 / cellZoom / canvasHeight;
    cellPanX -= dx * scaleX;
    cellPanY += dy * scaleY;
    cellDragStartX = e.clientX;
    cellDragStartY = e.clientY;
    updateDisplay();
  });

  cellCanvas.addEventListener('mouseup', (e) => {
    const wasDragging = cellIsDragging;
    cellIsDragging = false;
    if (!attractorMode) cellCanvas.style.cursor = 'grab';

    // If we didn't drag, treat as a click to select cell
    if (wasDragging && !cellDidDrag && !attractorMode) {
      const pos = screenToCell(e.clientX, e.clientY);
      const cell = findNearestCell(pos.x, pos.y);
      selectCell(cell);
    }
  });

  cellCanvas.addEventListener('mouseleave', () => {
    cellIsDragging = false;
    if (!attractorMode) cellCanvas.style.cursor = 'grab';
  });

  cellCanvas.style.cursor = 'grab';

  // Expression canvas zoom and pan
  exprCanvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    exprZoom = Math.max(0.1, Math.min(10, exprZoom * zoomFactor));
    updateDisplay();
  });

  exprCanvas.addEventListener('mousedown', (e) => {
    exprIsDragging = true;
    exprDragStartX = e.clientX;
    exprDragStartY = e.clientY;
    exprCanvas.style.cursor = 'grabbing';
  });

  exprCanvas.addEventListener('mousemove', (e) => {
    if (!exprIsDragging) return;
    const dx = e.clientX - exprDragStartX;
    const dy = e.clientY - exprDragStartY;
    // Convert pixel movement to normalized pan units (consistent sensitivity)
    const dpr = window.devicePixelRatio || 1;
    const canvasWidth = exprCanvas.width / dpr;
    const canvasHeight = exprCanvas.height / dpr;
    // Use normalized pan values that get scaled in draw function
    const scaleX = 1 / exprZoom / canvasWidth * Math.max(time, 1);
    const scaleY = 1 / exprZoom / canvasHeight * 4; // ~4 expression units
    exprPanX -= dx * scaleX;
    exprPanY -= dy * scaleY;
    exprDragStartX = e.clientX;
    exprDragStartY = e.clientY;
    updateDisplay();
  });

  exprCanvas.addEventListener('mouseup', () => {
    exprIsDragging = false;
    exprCanvas.style.cursor = 'grab';
  });

  exprCanvas.addEventListener('mouseleave', () => {
    exprIsDragging = false;
    exprCanvas.style.cursor = 'grab';
  });

  exprCanvas.style.cursor = 'grab';
}

function resetCellZoom(): void {
  cellZoom = 1.0;
  cellPanX = 0;
  cellPanY = 0;
  updateDisplay();
}

function resetExprZoom(): void {
  exprZoom = 1.0;
  exprPanX = 0;
  exprPanY = 0;
  updateDisplay();
}

function setupAccordions(): void {
  // Set up all accordion headers
  document.querySelectorAll('.accordion-header').forEach(header => {
    const targetId = header.getAttribute('data-target');
    if (!targetId) return;
    const content = document.getElementById(targetId);
    if (!content) return;

    header.addEventListener('click', () => {
      content.classList.toggle('collapsed');
      header.classList.toggle('collapsed');
    });

    // Initialize collapsed state for header icon if content is collapsed
    if (content.classList.contains('collapsed')) {
      header.classList.add('collapsed');
    }
  });
}

function setupMaxTimeControl(): void {
  const maxTimeSlider = document.getElementById('max-time') as HTMLInputElement;
  const maxTimeVal = document.getElementById('max-time-val');
  if (maxTimeSlider && maxTimeVal) {
    maxTimeSlider.addEventListener('input', () => {
      maxTime = parseInt(maxTimeSlider.value, 10);
      maxTimeVal.textContent = String(maxTime);
    });
  }
}

function setupGeneModifiers(): void {
  // Helper to create a modifier row
  const createModifierRow = (container: HTMLElement, geneIdx: number, displayName: string) => {
    const row = document.createElement('div');
    row.className = 'modifier-row';

    const label = document.createElement('label');
    label.textContent = displayName;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '2';
    slider.step = '0.1';
    slider.value = '1';
    slider.dataset.gene = String(geneIdx);

    const valueSpan = document.createElement('span');
    valueSpan.textContent = '1.0';

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      geneModifiers[geneIdx] = val;
      valueSpan.textContent = val.toFixed(1);
    });

    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(valueSpan);
    container.appendChild(row);
  };

  // Progenitor TFs
  const progContainer = document.getElementById('modifier-prog-list');
  if (progContainer) {
    geneGroups.tf_prog.forEach(geneIdx => {
      const name = geneNames[geneIdx] ?? `Gene_${geneIdx}`;
      const shortName = name.replace('TF_PROG_', 'P');
      createModifierRow(progContainer, geneIdx, shortName);
    });
  }

  // Lineage TFs
  const linContainer = document.getElementById('modifier-lin-list');
  if (linContainer) {
    geneGroups.tf_lin.forEach(geneIdx => {
      const name = geneNames[geneIdx] ?? `Gene_${geneIdx}`;
      const shortName = name.replace('TF_LIN_', 'L');
      createModifierRow(linContainer, geneIdx, shortName);
    });
  }

  // Ligands
  const ligandContainer = document.getElementById('modifier-ligand-list');
  if (ligandContainer) {
    geneGroups.ligand.forEach(geneIdx => {
      const name = geneNames[geneIdx] ?? `Gene_${geneIdx}`;
      const shortName = name.replace('LIG_', '');
      createModifierRow(ligandContainer, geneIdx, shortName);
    });
  }

  // Receptors
  const receptorContainer = document.getElementById('modifier-receptor-list');
  if (receptorContainer) {
    geneGroups.receptor.forEach(geneIdx => {
      const name = geneNames[geneIdx] ?? `Gene_${geneIdx}`;
      const shortName = name.replace('REC_', '');
      createModifierRow(receptorContainer, geneIdx, shortName);
    });
  }

  // Reset modifiers button
  const resetBtn = document.getElementById('btn-reset-modifiers');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      // Clear all modifiers (reset to 1.0)
      Object.keys(geneModifiers).forEach(key => {
        delete geneModifiers[parseInt(key, 10)];
      });
      // Reset all sliders to 1.0
      document.querySelectorAll('.modifier-row input[type="range"]').forEach(slider => {
        const inp = slider as HTMLInputElement;
        inp.value = '1';
        const valueSpan = inp.parentElement?.querySelector('span');
        if (valueSpan) valueSpan.textContent = '1.0';
      });
    });
  }
}

function setupAdvancedControls(): void {

  // Helper to bind slider to param
  const bindSlider = (id: string, paramKey: keyof typeof params, valueId: string, formatter: (v: number) => string = v => v.toFixed(2)) => {
    const slider = document.getElementById(id) as HTMLInputElement;
    const valueEl = document.getElementById(valueId);
    if (!slider || !valueEl) return;

    // Set initial value
    slider.value = String(params[paramKey]);
    valueEl.textContent = formatter(params[paramKey]);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      (params as Record<string, number>)[paramKey] = val;
      valueEl.textContent = formatter(val);
    });
  };

  // Initial expression sliders
  bindSlider('param-init-prog', 'initProgTF', 'val-init-prog');
  bindSlider('param-init-lin', 'initLinTF', 'val-init-lin');
  bindSlider('param-init-ligand', 'initLigand', 'val-init-ligand');

  // Regulatory sliders
  bindSlider('param-prog-bias', 'progBias', 'val-prog-bias');
  bindSlider('param-lin-bias', 'linBias', 'val-lin-bias');
  bindSlider('param-prog-decay', 'progDecay', 'val-prog-decay');
  bindSlider('param-lin-decay', 'linDecay', 'val-lin-decay');
  bindSlider('param-inhibition', 'inhibitionMult', 'val-inhibition');

  // Morphogen sliders
  bindSlider('param-morph-time', 'morphogenTime', 'val-morph-time', v => `${v}h`);
  bindSlider('param-morph-strength', 'morphogenStrength', 'val-morph-strength');

  // Hill function sliders
  bindSlider('param-hill-n', 'hillN', 'val-hill-n', v => String(Math.round(v)));
  bindSlider('param-hill-k', 'hillK', 'val-hill-k');

  // Presets dropdown
  const presetSelect = document.getElementById('param-presets') as HTMLSelectElement;
  if (presetSelect) {
    presetSelect.addEventListener('change', () => {
      const presetName = presetSelect.value;
      const preset = presets[presetName];
      if (preset) {
        // Apply preset values
        Object.entries(preset).forEach(([key, value]) => {
          (params as Record<string, number>)[key] = value as number;
        });
        // Update all slider values
        updateSliderValues();
      }
    });
  }

  // Apply & Reset button
  const applyBtn = document.getElementById('btn-apply-params');
  if (applyBtn) {
    applyBtn.addEventListener('click', () => {
      stopSimulation();
      resetSimulation();
    });
  }
}

function setupKnockoutControls(): void {
  const progContainer = document.getElementById('knockout-prog-list');
  const linContainer = document.getElementById('knockout-lin-list');
  const ligandContainer = document.getElementById('knockout-ligand-list');
  const receptorContainer = document.getElementById('knockout-receptor-list');

  if (!progContainer || !linContainer) return;

  // Helper to create knockout checkbox
  const createKnockoutItem = (container: HTMLElement, geneIdx: number, shortName: string) => {
    const label = document.createElement('label');
    label.className = 'knockout-item';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.dataset.gene = String(geneIdx);

    const span = document.createElement('span');
    span.textContent = shortName;

    label.appendChild(checkbox);
    label.appendChild(span);
    container.appendChild(label);

    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        knockouts.add(geneIdx);
      } else {
        knockouts.delete(geneIdx);
      }
      updateKnockoutCount();
    });
  };

  // Create checkboxes for progenitor TFs
  geneGroups.tf_prog.forEach(geneIdx => {
    const name = geneNames[geneIdx] ?? `Gene_${geneIdx}`;
    const shortName = name.replace('TF_PROG_', 'P');
    createKnockoutItem(progContainer, geneIdx, shortName);
  });

  // Create checkboxes for lineage TFs
  geneGroups.tf_lin.forEach(geneIdx => {
    const name = geneNames[geneIdx] ?? `Gene_${geneIdx}`;
    const shortName = name.replace('TF_LIN_', 'L');
    createKnockoutItem(linContainer, geneIdx, shortName);
  });

  // Create checkboxes for ligands
  if (ligandContainer) {
    geneGroups.ligand.forEach(geneIdx => {
      const name = geneNames[geneIdx] ?? `Gene_${geneIdx}`;
      const shortName = name.replace('LIG_', '');
      createKnockoutItem(ligandContainer, geneIdx, shortName);
    });
  }

  // Create checkboxes for receptors
  if (receptorContainer) {
    geneGroups.receptor.forEach(geneIdx => {
      const name = geneNames[geneIdx] ?? `Gene_${geneIdx}`;
      const shortName = name.replace('REC_', '');
      createKnockoutItem(receptorContainer, geneIdx, shortName);
    });
  }

  // Create toggles for morphogen signals (one per lineage)
  const morphogenContainer = document.getElementById('morphogen-list');
  if (morphogenContainer) {
    for (let lin = 0; lin < 6; lin++) {
      const label = document.createElement('label');
      label.className = 'knockout-item morphogen-item enabled';

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = true; // Enabled by default
      checkbox.dataset.lineage = String(lin);

      const span = document.createElement('span');
      span.textContent = `M${lin + 1}`;

      label.appendChild(checkbox);
      label.appendChild(span);
      morphogenContainer.appendChild(label);

      checkbox.addEventListener('change', () => {
        morphogenEnabled[lin] = checkbox.checked;
        label.classList.toggle('enabled', checkbox.checked);
        updateKnockoutCount();
      });
    }
  }

  // Set up morphogen strength sliders
  for (let lin = 0; lin < 6; lin++) {
    const slider = document.getElementById(`morph-str-${lin}`) as HTMLInputElement;
    const valueEl = document.getElementById(`morph-val-${lin}`);
    if (slider && valueEl) {
      slider.addEventListener('input', () => {
        const val = parseFloat(slider.value);
        morphogenStrengths[lin] = val;
        valueEl.textContent = val.toFixed(1);
      });
    }
  }

  // Create toggles for network edge types
  const edgeTypeContainer = document.getElementById('edge-type-list');
  if (edgeTypeContainer) {
    const edgeTypeLabels: Record<string, string> = {
      prog_to_lineage: 'Prog→Lin',
      prog_mutual: 'Prog↔Prog',
      lineage_self: 'Lin→Self',
      lineage_partner: 'Lin↔Partner',
      lineage_inhibit: 'Lin⊣Lin',
    };

    Object.keys(edgeTypeEnabled).forEach(edgeType => {
      const label = document.createElement('label');
      label.className = 'knockout-item edge-type-item enabled';

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = true;
      checkbox.dataset.edgeType = edgeType;

      const span = document.createElement('span');
      span.textContent = edgeTypeLabels[edgeType] ?? edgeType;

      label.appendChild(checkbox);
      label.appendChild(span);
      edgeTypeContainer.appendChild(label);

      checkbox.addEventListener('change', () => {
        edgeTypeEnabled[edgeType] = checkbox.checked;
        label.classList.toggle('enabled', checkbox.checked);
        updateKnockoutCount();
      });
    });
  }

  // Clear all knockouts button
  const clearBtn = document.getElementById('btn-clear-knockouts');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      // Clear gene knockouts
      knockouts.clear();
      document.querySelectorAll('.knockout-item:not(.morphogen-item):not(.edge-type-item) input').forEach(cb => {
        (cb as HTMLInputElement).checked = false;
      });
      // Reset morphogens to enabled and default strength
      for (let i = 0; i < 6; i++) {
        morphogenEnabled[i] = true;
        morphogenStrengths[i] = 1.0;
        const slider = document.getElementById(`morph-str-${i}`) as HTMLInputElement;
        const valueEl = document.getElementById(`morph-val-${i}`);
        if (slider) slider.value = '1';
        if (valueEl) valueEl.textContent = '1.0';
      }
      document.querySelectorAll('.morphogen-item').forEach(item => {
        const cb = item.querySelector('input') as HTMLInputElement;
        if (cb) cb.checked = true;
        item.classList.add('enabled');
      });
      // Reset edge types to enabled
      Object.keys(edgeTypeEnabled).forEach(key => {
        edgeTypeEnabled[key] = true;
      });
      document.querySelectorAll('.edge-type-item').forEach(item => {
        const cb = item.querySelector('input') as HTMLInputElement;
        if (cb) cb.checked = true;
        item.classList.add('enabled');
      });
      updateKnockoutCount();
    });
  }

  // Collapsible toggle
  const header = document.getElementById('knockout-header');
  const content = document.getElementById('knockout-content');
  if (header && content) {
    header.addEventListener('click', () => {
      content.classList.toggle('collapsed');
      header.classList.toggle('collapsed');
    });
  }
}

function updateKnockoutCount(): void {
  const countEl = document.getElementById('knockout-count');
  if (countEl) {
    const disabledMorphogens = morphogenEnabled.filter(e => !e).length;
    const disabledEdgeTypes = Object.values(edgeTypeEnabled).filter(e => !e).length;
    const total = knockouts.size + disabledMorphogens + disabledEdgeTypes;
    countEl.textContent = total > 0 ? `(${total})` : '';
  }
}

function updateSliderValues(): void {
  const updateSlider = (id: string, paramKey: keyof typeof params, formatter: (v: number) => string = v => v.toFixed(2)) => {
    const slider = document.getElementById(id) as HTMLInputElement;
    const valueEl = document.getElementById(id.replace('param-', 'val-'));
    if (slider && valueEl) {
      slider.value = String(params[paramKey]);
      valueEl.textContent = formatter(params[paramKey]);
    }
  };

  updateSlider('param-init-prog', 'initProgTF');
  updateSlider('param-init-lin', 'initLinTF');
  updateSlider('param-init-ligand', 'initLigand');
  updateSlider('param-prog-bias', 'progBias');
  updateSlider('param-lin-bias', 'linBias');
  updateSlider('param-prog-decay', 'progDecay');
  updateSlider('param-lin-decay', 'linDecay');
  updateSlider('param-inhibition', 'inhibitionMult');
  updateSlider('param-morph-time', 'morphogenTime', v => `${v}h`);
  updateSlider('param-morph-strength', 'morphogenStrength');
  updateSlider('param-hill-n', 'hillN', v => String(Math.round(v)));
  updateSlider('param-hill-k', 'hillK');
}

// Apply parameters from URL (from inference page)
function applyURLParams(): void {
  const urlParams = new URLSearchParams(window.location.search);
  const encodedParams = urlParams.get('params');

  if (!encodedParams) return;

  try {
    const json = atob(encodedParams);
    const data = JSON.parse(json);

    // Apply global parameters
    if (data.g) {
      const g = data.g;
      if (typeof g.hillN === 'number') params.hillN = g.hillN;
      if (typeof g.hillK === 'number') params.hillK = g.hillK;
      if (typeof g.progBias === 'number') params.progBias = g.progBias;
      if (typeof g.linBias === 'number') params.linBias = g.linBias;
      if (typeof g.progDecay === 'number') params.progDecay = g.progDecay;
      if (typeof g.linDecay === 'number') params.linDecay = g.linDecay;
      if (typeof g.inhibitionMult === 'number') params.inhibitionMult = g.inhibitionMult;
      if (typeof g.morphogenTime === 'number') params.morphogenTime = g.morphogenTime;
      if (typeof g.morphogenStrength === 'number') params.morphogenStrength = g.morphogenStrength;
      if (typeof g.initProgTF === 'number') params.initProgTF = g.initProgTF;
      if (typeof g.initLinTF === 'number') params.initLinTF = g.initLinTF;
      if (typeof g.ligandBias === 'number') params.ligandBias = g.ligandBias;
      if (typeof g.receptorBias === 'number') params.receptorBias = g.receptorBias;
    }

    // Apply knockouts
    if (data.k && Array.isArray(data.k)) {
      knockouts.clear();
      data.k.forEach((idx: number) => knockouts.add(idx));
    }

    // Apply morphogen settings
    if (data.m && Array.isArray(data.m)) {
      data.m.forEach((m: { enabled: boolean; strength: number }, i: number) => {
        if (i < morphogenEnabled.length) {
          morphogenEnabled[i] = m.enabled;
          morphogenStrengths[i] = m.strength;
        }
      });
    }

    // Apply gene modifiers (from inference optimization)
    if (data.mod && Array.isArray(data.mod)) {
      // Clear existing modifiers
      Object.keys(geneModifiers).forEach(key => {
        delete geneModifiers[parseInt(key, 10)];
      });
      // Apply new modifiers - data.mod is array of [geneIdx, modifier] tuples
      data.mod.forEach(([geneIdx, modifier]: [number, number]) => {
        if (typeof geneIdx === 'number' && typeof modifier === 'number') {
          geneModifiers[geneIdx] = modifier;
        }
      });
    }

    // Update UI sliders to reflect imported params
    updateAdvancedSlidersFromParams();

    // Update knockout UI
    updateKnockoutUI();

    // Update morphogen UI
    updateMorphogenUI();

    // Update gene modifier UI
    updateModifierUI();

    console.log('Applied parameters from inference page');
  } catch (e) {
    console.warn('Failed to parse URL params:', e);
  }
}

// Update knockout checkboxes from state
function updateKnockoutUI(): void {
  document.querySelectorAll('.knockout-checkbox').forEach(cb => {
    const geneIdx = parseInt((cb as HTMLInputElement).dataset.gene ?? '-1', 10);
    if (geneIdx >= 0) {
      (cb as HTMLInputElement).checked = knockouts.has(geneIdx);
    }
  });
  updateKnockoutCount();
}

// Update morphogen controls from state
function updateMorphogenUI(): void {
  // Update enable checkboxes
  document.querySelectorAll('#morphogen-list .knockout-checkbox').forEach(cb => {
    const idx = parseInt((cb as HTMLInputElement).dataset.morphogen ?? '-1', 10);
    if (idx >= 0 && idx < morphogenEnabled.length) {
      (cb as HTMLInputElement).checked = morphogenEnabled[idx];
    }
  });

  // Update strength sliders
  for (let i = 0; i < morphogenStrengths.length; i++) {
    const slider = document.getElementById(`morph-str-${i}`) as HTMLInputElement;
    const valSpan = document.getElementById(`morph-val-${i}`);
    if (slider) {
      slider.value = morphogenStrengths[i].toString();
    }
    if (valSpan) {
      valSpan.textContent = morphogenStrengths[i].toFixed(1);
    }
  }
}

// Update gene modifier sliders from state
function updateModifierUI(): void {
  document.querySelectorAll('.modifier-row input[type="range"]').forEach(slider => {
    const inp = slider as HTMLInputElement;
    const geneIdx = parseInt(inp.dataset.gene ?? '-1', 10);
    if (geneIdx >= 0) {
      const modifier = geneModifiers[geneIdx] ?? 1.0;
      inp.value = modifier.toString();
      const valueSpan = inp.parentElement?.querySelector('span');
      if (valueSpan) {
        valueSpan.textContent = modifier.toFixed(1);
      }
    }
  });
}

// Update advanced sliders from current params
function updateAdvancedSlidersFromParams(): void {
  const sliderMap: { id: string; param: keyof typeof params; format?: (v: number) => string }[] = [
    { id: 'param-init-prog', param: 'initProgTF' },
    { id: 'param-init-lin', param: 'initLinTF' },
    { id: 'param-init-ligand', param: 'initLigand' },
    { id: 'param-prog-bias', param: 'progBias' },
    { id: 'param-lin-bias', param: 'linBias' },
    { id: 'param-prog-decay', param: 'progDecay' },
    { id: 'param-lin-decay', param: 'linDecay' },
    { id: 'param-inhibition', param: 'inhibitionMult' },
    { id: 'param-morph-time', param: 'morphogenTime', format: v => `${v}h` },
    { id: 'param-morph-strength', param: 'morphogenStrength' },
    { id: 'param-hill-n', param: 'hillN', format: v => String(Math.round(v)) },
    { id: 'param-hill-k', param: 'hillK' },
  ];

  for (const { id, param, format } of sliderMap) {
    const slider = document.getElementById(id) as HTMLInputElement;
    const valSpan = document.getElementById(id.replace('param-', 'val-'));

    if (slider) {
      slider.value = params[param].toString();
    }
    if (valSpan) {
      const val = params[param];
      valSpan.textContent = format ? format(val) : val.toFixed(2);
    }
  }
}

function init(): void {
  cellCanvas = document.getElementById('cell-canvas') as HTMLCanvasElement;
  cellCtx = cellCanvas.getContext('2d')!;
  exprCanvas = document.getElementById('expression-canvas') as HTMLCanvasElement;
  exprCtx = exprCanvas.getContext('2d')!;
  proportionsCanvas = document.getElementById('proportions-canvas') as HTMLCanvasElement;
  proportionsCtx = proportionsCanvas.getContext('2d')!;

  // Setup controls
  const nCellsSlider = document.getElementById('n-cells') as HTMLInputElement;
  const speedSlider = document.getElementById('speed') as HTMLInputElement;
  const noiseSlider = document.getElementById('noise') as HTMLInputElement;

  nCellsSlider.addEventListener('input', () => {
    nCells = parseInt(nCellsSlider.value, 10);
    document.getElementById('n-cells-val')!.textContent = String(nCells);
  });

  speedSlider.addEventListener('input', () => {
    speed = parseFloat(speedSlider.value);
    document.getElementById('speed-val')!.textContent = `${speed}x`;
  });

  noiseSlider.addEventListener('input', () => {
    noiseLevel = parseFloat(noiseSlider.value);
    document.getElementById('noise-val')!.textContent = noiseLevel.toFixed(2);
  });

  // Play/pause button
  document.getElementById('btn-play')!.addEventListener('click', () => {
    if (isPlaying) {
      stopSimulation();
    } else {
      startSimulation();
    }
  });

  // Reset button
  document.getElementById('btn-reset')!.addEventListener('click', () => {
    stopSimulation();
    resetSimulation();
  });

  // Color by radio buttons
  document.querySelectorAll('input[name="color-by"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      colorBy = (e.target as HTMLInputElement).value as 'lineage' | 'time' | 'gene';
      const geneSelect = document.getElementById('gene-select-container')!;
      if (colorBy === 'gene') {
        geneSelect.classList.remove('hidden');
      } else {
        geneSelect.classList.add('hidden');
      }
      updateDisplay();
    });
  });

  // Gene group buttons
  document.querySelectorAll('.gene-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.gene-btn').forEach(b => b.classList.remove('active'));
      (e.target as HTMLElement).classList.add('active');
      selectedGeneGroup = (e.target as HTMLElement).dataset.genes as typeof selectedGeneGroup;

      // Show/hide custom gene selector
      const customSelector = document.getElementById('custom-gene-selector');
      if (customSelector) {
        if (selectedGeneGroup === 'custom') {
          customSelector.classList.remove('hidden');
        } else {
          customSelector.classList.add('hidden');
        }
      }
      updateDisplay();
    });
  });

  // Dynamics tab switching
  document.querySelectorAll('.dynamics-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      document.querySelectorAll('.dynamics-tab').forEach(t => t.classList.remove('active'));
      (e.target as HTMLElement).classList.add('active');
      activeDynamicsTab = (e.target as HTMLElement).dataset.tab as 'expression' | 'proportions';

      // Show/hide tab content
      document.getElementById('expression-tab-content')?.classList.toggle('active', activeDynamicsTab === 'expression');
      document.getElementById('proportions-tab-content')?.classList.toggle('active', activeDynamicsTab === 'proportions');

      updateDisplay();
    });
  });

  // Setup custom gene selector
  setupCustomGeneSelector();

  // Setup
  setupLegend();
  setupGeneSelect();
  setupAccordions();
  setupMaxTimeControl();
  setupGeneModifiers();
  setupAdvancedControls();
  setupKnockoutControls();
  setupSteeringControls();
  setupCanvasZoomPan();

  // Zoom reset buttons
  const resetCellZoomBtn = document.getElementById('btn-reset-cell-zoom');
  if (resetCellZoomBtn) {
    resetCellZoomBtn.addEventListener('click', resetCellZoom);
  }

  const resetExprZoomBtn = document.getElementById('btn-reset-expr-zoom');
  if (resetExprZoomBtn) {
    resetExprZoomBtn.addEventListener('click', resetExprZoom);
  }

  // Main apply button (always visible)
  const applyMainBtn = document.getElementById('btn-apply-main');
  if (applyMainBtn) {
    applyMainBtn.addEventListener('click', () => {
      stopSimulation();
      resetSimulation();
    });
  }

  // Reference toggle
  const showRefCheckbox = document.getElementById('show-reference') as HTMLInputElement;
  if (showRefCheckbox) {
    showRefCheckbox.addEventListener('change', () => {
      showReference = showRefCheckbox.checked;
      updateDisplay();
    });
  }

  // Uniform morphogens toggle
  const uniformMorphogensCheckbox = document.getElementById('uniform-morphogens') as HTMLInputElement;
  if (uniformMorphogensCheckbox) {
    uniformMorphogensCheckbox.addEventListener('change', () => {
      uniformMorphogens = uniformMorphogensCheckbox.checked;
    });
  }

  // Panel toggle buttons (minimize/expand)
  document.querySelectorAll('.panel-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = (btn as HTMLElement).dataset.target;
      if (!targetId) return;
      const row = document.getElementById(targetId);
      if (!row) return;
      const isCollapsed = row.classList.toggle('collapsed');
      btn.textContent = isCollapsed ? '+' : '−';
      btn.setAttribute('title', isCollapsed ? 'Expand' : 'Minimize');
      // Trigger resize after layout settles
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          resizeCanvases();
          updateDisplay();
        });
      });
    });
  });

  // Initial canvas sizing
  resizeCanvases();

  // Check for imported parameters from inference page
  applyURLParams();

  resetSimulation();
  updateCellDetails(); // Initialize cell details panel
  drawSelectedCellExpression(); // Initialize selected cell expression panel

  // Handle resize
  window.addEventListener('resize', () => {
    resizeCanvases();
    updateDisplay();
  });
}

document.addEventListener('DOMContentLoaded', init);
