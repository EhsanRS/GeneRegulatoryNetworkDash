/**
 * Parameter Inference Page
 *
 * Allows users to define a target cell state and uses CMA-ES optimization
 * to find simulation parameters that produce that state.
 */

import grnData from '../synth_grn.json';
import { InferenceEngine, type GRNData, type GeneGroups } from './inference/inference-engine';
import {
  decode,
  getDimensions,
  type SimulationParams,
  type EncodingConfig,
  exportParamsJSON,
  encodeParamsURL,
  GLOBAL_PARAM_BOUNDS,
  GLOBAL_PARAM_KEYS,
} from './inference/encoding';
import type {
  WorkerStartMessage,
  WorkerOutMessage,
  WorkerProgressMessage,
  WorkerDoneMessage,
} from './inference/worker';

// Initialize inference engine
const data = grnData as GRNData;
const engine = new InferenceEngine(data);
const geneNames = engine.getGeneNames();
const geneGroups = engine.getGeneGroups();
const nGenes = engine.getNumGenes();
const knockoutCandidates = engine.getKnockoutCandidates();

// State
let worker: Worker | null = null;
let isRunning = false;
let targetExpression = new Float32Array(nGenes);
let targetWeights = new Float32Array(nGenes).fill(1);
let bestParams: SimulationParams | null = null;

// Chart state
let fitnessHistory: { gen: number; fitness: number; meanFitness: number }[] = [];
let bestSimulatedExpression: Float32Array | null = null;

// Canvas contexts
let fitnessCanvas: HTMLCanvasElement;
let fitnessCtx: CanvasRenderingContext2D;
let comparisonCanvas: HTMLCanvasElement;
let comparisonCtx: CanvasRenderingContext2D;
let scatterCanvas: HTMLCanvasElement;
let scatterCtx: CanvasRenderingContext2D;

// Weight preset modes
let weightPreset: 'uniform' | 'tfs-only' | 'select' | 'custom' = 'select';

// Selected genes for 'select' mode
const selectedGenes = new Set<number>();

// Initialize default target expression (progenitor state)
function initializeDefaultTarget(): void {
  // Start with moderate progenitor expression
  geneGroups.tf_prog.forEach(i => { targetExpression[i] = 1.5; });
  // Low lineage TF expression
  geneGroups.tf_lin.forEach(i => { targetExpression[i] = 0.2; });
  // Moderate ligand/receptor
  geneGroups.ligand.forEach(i => { targetExpression[i] = 0.3; });
  geneGroups.receptor.forEach(i => { targetExpression[i] = 0.3; });
  // Low targets
  geneGroups.target.forEach(i => { targetExpression[i] = 0.1; });
  // Housekeeping stable
  geneGroups.housekeeping.forEach(i => { targetExpression[i] = 1.0; });
  geneGroups.other.forEach(i => { targetExpression[i] = 0.1; });
}

// Initialize target expression sliders
function initializeTargetSliders(): void {
  const groups: { id: string; indices: number[]; label: string }[] = [
    { id: 'prog-targets', indices: geneGroups.tf_prog, label: 'Progenitor TFs' },
    { id: 'lin-targets', indices: geneGroups.tf_lin, label: 'Lineage TFs' },
    { id: 'ligand-targets', indices: geneGroups.ligand, label: 'Ligands' },
    { id: 'receptor-targets', indices: geneGroups.receptor, label: 'Receptors' },
  ];

  for (const group of groups) {
    const container = document.getElementById(group.id);
    if (!container) continue;

    // Clear existing content safely
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    for (const idx of group.indices) {
      const name = geneNames[idx] ?? `Gene_${idx}`;
      const shortName = name.replace('TF_PROG_', 'P').replace('TF_LIN', 'L').replace('LIG_', 'Lig').replace('REC_', 'Rec');

      const row = document.createElement('div');
      row.className = 'target-row';

      const label = document.createElement('label');
      label.title = name;
      label.textContent = shortName;

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.id = `target-${idx}`;
      slider.min = '0';
      slider.max = '3';
      slider.step = '0.1';
      slider.value = targetExpression[idx].toFixed(1);

      const valSpan = document.createElement('span');
      valSpan.className = 'target-val';
      valSpan.id = `val-target-${idx}`;
      valSpan.textContent = targetExpression[idx].toFixed(1);

      row.appendChild(label);
      row.appendChild(slider);
      row.appendChild(valSpan);
      container.appendChild(row);

      // Event listener
      slider.addEventListener('input', () => {
        const val = parseFloat(slider.value);
        targetExpression[idx] = val;
        valSpan.textContent = val.toFixed(1);
        updateCharts();
      });
    }
  }
}

// Update weights based on preset
function updateWeights(): void {
  switch (weightPreset) {
    case 'uniform':
      targetWeights.fill(1);
      break;
    case 'tfs-only':
      targetWeights.fill(0);
      geneGroups.tf_prog.forEach(i => { targetWeights[i] = 2; });
      geneGroups.tf_lin.forEach(i => { targetWeights[i] = 2; });
      break;
    case 'select':
      // Only selected genes have weight=1, others have weight=0
      targetWeights.fill(0);
      for (const idx of selectedGenes) {
        targetWeights[idx] = 1;
      }
      break;
    case 'custom':
      // Read from custom weight sliders
      const progWeight = parseFloat((document.getElementById('weight-prog') as HTMLInputElement).value);
      const linWeight = parseFloat((document.getElementById('weight-lin') as HTMLInputElement).value);
      const ligandWeight = parseFloat((document.getElementById('weight-ligand') as HTMLInputElement).value);
      const receptorWeight = parseFloat((document.getElementById('weight-receptor') as HTMLInputElement).value);

      targetWeights.fill(0.1);
      geneGroups.tf_prog.forEach(i => { targetWeights[i] = progWeight; });
      geneGroups.tf_lin.forEach(i => { targetWeights[i] = linWeight; });
      geneGroups.ligand.forEach(i => { targetWeights[i] = ligandWeight; });
      geneGroups.receptor.forEach(i => { targetWeights[i] = receptorWeight; });
      break;
  }
}

// Update selected gene count display
function updateSelectedCount(): void {
  const countSpan = document.getElementById('selected-gene-count');
  if (countSpan) {
    countSpan.textContent = selectedGenes.size.toString();
  }
}

// Toggle gene selection
function toggleGeneSelection(idx: number, chip: HTMLElement): void {
  if (selectedGenes.has(idx)) {
    selectedGenes.delete(idx);
    chip.classList.remove('selected');
  } else {
    selectedGenes.add(idx);
    chip.classList.add('selected');
  }
  updateSelectedCount();
  if (weightPreset === 'select') {
    updateWeights();
  }
}

// Initialize gene selection UI
function initializeGeneSelectionUI(): void {
  const groups: { containerId: string; groupKey: keyof GeneGroups; label: string }[] = [
    { containerId: 'select-prog', groupKey: 'tf_prog', label: 'prog' },
    { containerId: 'select-lin', groupKey: 'tf_lin', label: 'lin' },
    { containerId: 'select-ligand', groupKey: 'ligand', label: 'ligand' },
    { containerId: 'select-receptor', groupKey: 'receptor', label: 'receptor' },
  ];

  for (const { containerId, groupKey, label } of groups) {
    const container = document.getElementById(containerId);
    if (!container) continue;

    // Clear existing content
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    const indices = geneGroups[groupKey];
    for (const idx of indices) {
      const name = geneNames[idx] ?? `Gene_${idx}`;
      const shortName = name
        .replace('TF_PROG_', 'P')
        .replace('TF_LIN', 'L')
        .replace('LIG_', '')
        .replace('REC_', '');

      const chip = document.createElement('button');
      chip.className = 'gene-chip';
      chip.textContent = shortName;
      chip.title = name;
      chip.dataset.geneIdx = idx.toString();

      // Check if already selected
      if (selectedGenes.has(idx)) {
        chip.classList.add('selected');
      }

      chip.addEventListener('click', () => {
        toggleGeneSelection(idx, chip);
      });

      container.appendChild(chip);
    }
  }

  // Set up All/None buttons
  const allBtns = document.querySelectorAll('.select-all-btn');
  const noneBtns = document.querySelectorAll('.select-none-btn');

  allBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.getAttribute('data-group');
      const groupKey = getGroupKeyFromLabel(group);
      if (!groupKey) return;

      const indices = geneGroups[groupKey];
      for (const idx of indices) {
        selectedGenes.add(idx);
      }

      // Update chip visuals
      const containerId = `select-${group}`;
      const container = document.getElementById(containerId);
      container?.querySelectorAll('.gene-chip').forEach(chip => {
        chip.classList.add('selected');
      });

      updateSelectedCount();
      if (weightPreset === 'select') {
        updateWeights();
      }
    });
  });

  noneBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.getAttribute('data-group');
      const groupKey = getGroupKeyFromLabel(group);
      if (!groupKey) return;

      const indices = geneGroups[groupKey];
      for (const idx of indices) {
        selectedGenes.delete(idx);
      }

      // Update chip visuals
      const containerId = `select-${group}`;
      const container = document.getElementById(containerId);
      container?.querySelectorAll('.gene-chip').forEach(chip => {
        chip.classList.remove('selected');
      });

      updateSelectedCount();
      if (weightPreset === 'select') {
        updateWeights();
      }
    });
  });

  updateSelectedCount();
}

// Helper to map group label to geneGroups key
function getGroupKeyFromLabel(label: string | null): keyof GeneGroups | null {
  switch (label) {
    case 'prog': return 'tf_prog';
    case 'lin': return 'tf_lin';
    case 'ligand': return 'ligand';
    case 'receptor': return 'receptor';
    default: return null;
  }
}

// Initialize weight controls
function initializeWeightControls(): void {
  const presetBtns = document.querySelectorAll('.preset-btn');
  const customPanel = document.getElementById('weight-custom');
  const selectPanel = document.getElementById('weight-select');

  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      presetBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      weightPreset = btn.getAttribute('data-preset') as 'uniform' | 'tfs-only' | 'select' | 'custom';

      // Show/hide panels based on preset
      if (customPanel) {
        customPanel.classList.toggle('hidden', weightPreset !== 'custom');
      }
      if (selectPanel) {
        selectPanel.classList.toggle('hidden', weightPreset !== 'select');
      }

      updateWeights();
    });
  });

  // Custom weight sliders
  const weightSliders = ['weight-prog', 'weight-lin', 'weight-ligand', 'weight-receptor'];
  for (const id of weightSliders) {
    const slider = document.getElementById(id) as HTMLInputElement;
    const valSpan = document.getElementById(`val-${id}`) as HTMLSpanElement;

    if (slider && valSpan) {
      slider.addEventListener('input', () => {
        valSpan.textContent = parseFloat(slider.value).toFixed(1);
        if (weightPreset === 'custom') {
          updateWeights();
        }
      });
    }
  }
}

// Initialize accordion toggles
function initializeAccordions(): void {
  const headers = document.querySelectorAll('.accordion-header');
  headers.forEach(header => {
    header.addEventListener('click', () => {
      const targetId = header.getAttribute('data-target');
      if (!targetId) return;

      const content = document.getElementById(targetId);
      if (!content) return;

      const icon = header.querySelector('.accordion-icon');
      content.classList.toggle('collapsed');

      if (icon) {
        icon.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
      }
    });
  });
}

// Initialize canvases
function initializeCanvases(): void {
  fitnessCanvas = document.getElementById('fitness-chart') as HTMLCanvasElement;
  comparisonCanvas = document.getElementById('comparison-chart') as HTMLCanvasElement;
  scatterCanvas = document.getElementById('scatter-chart') as HTMLCanvasElement;

  if (fitnessCanvas) fitnessCtx = fitnessCanvas.getContext('2d')!;
  if (comparisonCanvas) comparisonCtx = comparisonCanvas.getContext('2d')!;
  if (scatterCanvas) scatterCtx = scatterCanvas.getContext('2d')!;

  // Handle resize
  window.addEventListener('resize', resizeCanvases);
  resizeCanvases();
}

function resizeCanvases(): void {
  const canvases = [fitnessCanvas, comparisonCanvas, scatterCanvas];
  for (const canvas of canvases) {
    if (!canvas) continue;
    const rect = canvas.parentElement?.getBoundingClientRect();
    if (rect) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
  }
  updateCharts();
}

// Draw fitness history chart
function drawFitnessChart(): void {
  if (!fitnessCtx || !fitnessCanvas) return;

  const ctx = fitnessCtx;
  const w = fitnessCanvas.width;
  const h = fitnessCanvas.height;

  // Clear
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  if (fitnessHistory.length === 0) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No data yet', w / 2, h / 2);
    return;
  }

  const padding = { top: 20, right: 20, bottom: 30, left: 50 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  // Find ranges
  const maxGen = Math.max(...fitnessHistory.map(d => d.gen));
  const maxFitness = Math.max(...fitnessHistory.map(d => Math.max(d.fitness, d.meanFitness))) * 1.1;
  const minFitness = 0;

  // Draw axes
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, h - padding.bottom);
  ctx.lineTo(w - padding.right, h - padding.bottom);
  ctx.stroke();

  // Draw grid lines
  ctx.strokeStyle = '#21262d';
  ctx.setLineDash([2, 4]);
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Draw mean fitness line
  ctx.strokeStyle = '#8b949e';
  ctx.lineWidth = 1;
  ctx.beginPath();
  fitnessHistory.forEach((d, i) => {
    const x = padding.left + (d.gen / maxGen) * plotW;
    const y = h - padding.bottom - ((d.meanFitness - minFitness) / (maxFitness - minFitness)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw best fitness line
  ctx.strokeStyle = '#39c5cf';
  ctx.lineWidth = 2;
  ctx.beginPath();
  fitnessHistory.forEach((d, i) => {
    const x = padding.left + (d.gen / maxGen) * plotW;
    const y = h - padding.bottom - ((d.fitness - minFitness) / (maxFitness - minFitness)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Generation', padding.left + plotW / 2, h - 5);

  ctx.save();
  ctx.translate(15, padding.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Fitness', 0, 0);
  ctx.restore();

  // Current values
  if (fitnessHistory.length > 0) {
    const last = fitnessHistory[fitnessHistory.length - 1];
    ctx.fillStyle = '#39c5cf';
    ctx.textAlign = 'right';
    ctx.fillText(`Best: ${last.fitness.toFixed(4)}`, w - padding.right, padding.top + 15);
    ctx.fillStyle = '#8b949e';
    ctx.fillText(`Mean: ${last.meanFitness.toFixed(4)}`, w - padding.right, padding.top + 30);
  }
}

// Draw expression comparison chart
function drawComparisonChart(): void {
  if (!comparisonCtx || !comparisonCanvas) return;

  const ctx = comparisonCtx;
  const w = comparisonCanvas.width;
  const h = comparisonCanvas.height;

  // Clear
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  const padding = { top: 20, right: 20, bottom: 40, left: 40 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  // Show TF genes only for clarity
  const displayGenes = [...geneGroups.tf_prog, ...geneGroups.tf_lin];
  const barWidth = plotW / displayGenes.length / 2.5;

  // Draw target bars
  ctx.fillStyle = '#a371f7';
  displayGenes.forEach((idx, i) => {
    const x = padding.left + (i / displayGenes.length) * plotW + barWidth * 0.5;
    const val = targetExpression[idx] ?? 0;
    const barH = (val / 3) * plotH;
    ctx.fillRect(x, h - padding.bottom - barH, barWidth, barH);
  });

  // Draw simulated bars (if available)
  if (bestSimulatedExpression) {
    ctx.fillStyle = '#3fb950';
    displayGenes.forEach((idx, i) => {
      const x = padding.left + (i / displayGenes.length) * plotW + barWidth * 1.7;
      const val = bestSimulatedExpression![idx] ?? 0;
      const barH = (val / 3) * plotH;
      ctx.fillRect(x, h - padding.bottom - barH, barWidth, barH);
    });
  }

  // X-axis labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';

  // Only label every few genes to avoid clutter
  const labelInterval = Math.max(1, Math.floor(displayGenes.length / 12));
  displayGenes.forEach((idx, i) => {
    if (i % labelInterval !== 0) return;
    const x = padding.left + (i / displayGenes.length) * plotW + barWidth * 1.1;
    const name = geneNames[idx]?.replace('TF_PROG_', 'P').replace('TF_LIN', 'L') ?? '';
    ctx.save();
    ctx.translate(x, h - padding.bottom + 5);
    ctx.rotate(Math.PI / 4);
    ctx.textAlign = 'left';
    ctx.fillText(name, 0, 0);
    ctx.restore();
  });

  // Y-axis
  ctx.fillStyle = '#8b949e';
  ctx.textAlign = 'right';
  ctx.fillText('0', padding.left - 5, h - padding.bottom);
  ctx.fillText('3', padding.left - 5, padding.top + 10);
}

// Draw scatter plot (target vs simulated)
function drawScatterChart(): void {
  if (!scatterCtx || !scatterCanvas) return;

  const ctx = scatterCtx;
  const w = scatterCanvas.width;
  const h = scatterCanvas.height;

  // Clear
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  // Draw diagonal (perfect correlation line)
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padding.left, h - padding.bottom);
  ctx.lineTo(w - padding.right, padding.top);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw axes
  ctx.strokeStyle = '#30363d';
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, h - padding.bottom);
  ctx.lineTo(w - padding.right, h - padding.bottom);
  ctx.stroke();

  if (!bestSimulatedExpression) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Run optimization to see results', w / 2, h / 2);
    return;
  }

  // Draw points
  const maxVal = 3;
  const displayGenes = [...geneGroups.tf_prog, ...geneGroups.tf_lin, ...geneGroups.ligand, ...geneGroups.receptor];

  for (const idx of displayGenes) {
    const target = targetExpression[idx] ?? 0;
    const simulated = bestSimulatedExpression[idx] ?? 0;

    const x = padding.left + (target / maxVal) * plotW;
    const y = h - padding.bottom - (simulated / maxVal) * plotH;

    // Color by gene type
    let color = '#8b949e';
    if (geneGroups.tf_prog.includes(idx)) color = '#f85149';
    else if (geneGroups.tf_lin.includes(idx)) color = '#58a6ff';
    else if (geneGroups.ligand.includes(idx)) color = '#3fb950';
    else if (geneGroups.receptor.includes(idx)) color = '#a371f7';

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // Labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Target Expression', padding.left + plotW / 2, h - 5);

  ctx.save();
  ctx.translate(15, padding.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Simulated Expression', 0, 0);
  ctx.restore();

  // Axis values
  ctx.textAlign = 'center';
  ctx.fillText('0', padding.left, h - padding.bottom + 15);
  ctx.fillText('3', w - padding.right, h - padding.bottom + 15);

  ctx.textAlign = 'right';
  ctx.fillText('0', padding.left - 5, h - padding.bottom);
  ctx.fillText('3', padding.left - 5, padding.top + 10);
}

// Update all charts
function updateCharts(): void {
  drawFitnessChart();
  drawComparisonChart();
  drawScatterChart();
}

// Update results display - using safe DOM manipulation
function updateResults(): void {
  const paramsDiv = document.getElementById('result-params');
  const knockoutsDiv = document.getElementById('result-knockouts');
  const morphogensDiv = document.getElementById('result-morphogens');

  // Helper to clear element
  function clearElement(el: HTMLElement | null): void {
    if (!el) return;
    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  // Helper to add placeholder
  function addPlaceholder(el: HTMLElement | null, text: string): void {
    if (!el) return;
    const p = document.createElement('p');
    p.className = 'placeholder';
    p.textContent = text;
    el.appendChild(p);
  }

  if (!bestParams) {
    clearElement(paramsDiv);
    clearElement(knockoutsDiv);
    clearElement(morphogensDiv);
    addPlaceholder(paramsDiv, 'Run optimization to see results');
    addPlaceholder(knockoutsDiv, '-');
    addPlaceholder(morphogensDiv, '-');
    return;
  }

  // Global parameters
  if (paramsDiv) {
    clearElement(paramsDiv);
    for (const key of GLOBAL_PARAM_KEYS) {
      const val = bestParams.global[key];
      const name = key.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase());

      const row = document.createElement('div');
      row.className = 'param-row';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'param-name';
      nameSpan.textContent = name;

      const valSpan = document.createElement('span');
      valSpan.className = 'param-value';
      valSpan.textContent = val.toFixed(3);

      row.appendChild(nameSpan);
      row.appendChild(valSpan);
      paramsDiv.appendChild(row);
    }
  }

  // Knockouts
  if (knockoutsDiv) {
    clearElement(knockoutsDiv);
    if (bestParams.knockouts.size === 0) {
      addPlaceholder(knockoutsDiv, 'None');
    } else {
      for (const idx of bestParams.knockouts) {
        const item = document.createElement('span');
        item.className = 'knockout-item';
        item.textContent = geneNames[idx] ?? `Gene_${idx}`;
        knockoutsDiv.appendChild(item);
      }
    }
  }

  // Morphogens
  if (morphogensDiv) {
    clearElement(morphogensDiv);
    bestParams.morphogens.forEach((m, i) => {
      const item = document.createElement('span');
      item.className = 'morphogen-item';
      item.textContent = `Lin${i + 1}: ${m.strength.toFixed(2)}`;
      morphogensDiv.appendChild(item);
    });
  }

  // Enable action buttons
  (document.getElementById('btn-apply') as HTMLButtonElement).disabled = false;
  (document.getElementById('btn-export') as HTMLButtonElement).disabled = false;
}

// Start optimization
function startOptimization(): void {
  if (isRunning) return;

  // Get config from UI
  const includeGlobal = (document.getElementById('opt-global') as HTMLInputElement).checked;
  const includeKnockouts = (document.getElementById('opt-knockouts') as HTMLInputElement).checked;
  const includeMorphogens = (document.getElementById('opt-morphogens') as HTMLInputElement).checked;
  const maxGenerations = parseInt((document.getElementById('max-generations') as HTMLInputElement).value);
  const populationSize = parseInt((document.getElementById('pop-size') as HTMLInputElement).value);
  const simTime = parseInt((document.getElementById('sim-time') as HTMLInputElement).value);

  // Update weights
  updateWeights();

  // Reset state
  fitnessHistory = [];
  bestSimulatedExpression = null;
  bestParams = null;

  // Create worker
  worker = new Worker(new URL('./inference/worker.ts', import.meta.url), { type: 'module' });

  worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
    const msg = event.data;

    switch (msg.type) {
      case 'progress':
        handleProgress(msg);
        break;
      case 'done':
        handleDone(msg);
        break;
      case 'error':
        console.error('Worker error:', msg.error);
        stopOptimization();
        break;
    }
  };

  worker.onerror = (error) => {
    console.error('Worker error:', error);
    stopOptimization();
  };

  // Send start message
  const startMsg: WorkerStartMessage = {
    type: 'start',
    grnData: data,
    target: {
      expression: Array.from(targetExpression),
      weights: Array.from(targetWeights),
    },
    config: {
      includeGlobal,
      includeKnockouts,
      includeMorphogens,
      maxGenerations,
      populationSize,
      sigma: 0.3,
      maxTime: simTime,
      ensembleSize: 5,
    },
    knockoutCandidates,
  };

  worker.postMessage(startMsg);

  // Update UI
  isRunning = true;
  updateUIRunning(true, maxGenerations);
}

function handleProgress(msg: WorkerProgressMessage): void {
  fitnessHistory.push({
    gen: msg.generation,
    fitness: msg.bestFitness,
    meanFitness: msg.meanFitness,
  });

  // Update stats
  const maxGen = parseInt((document.getElementById('max-generations') as HTMLInputElement).value);
  const genDisplay = document.getElementById('gen-display');
  const fitnessDisplay = document.getElementById('fitness-display');
  const corrDisplay = document.getElementById('corr-display');
  const sigmaDisplay = document.getElementById('sigma-display');

  if (genDisplay) genDisplay.textContent = `${msg.generation} / ${maxGen}`;
  if (fitnessDisplay) fitnessDisplay.textContent = msg.bestFitness.toFixed(4);
  if (corrDisplay) corrDisplay.textContent = msg.correlation.toFixed(3);
  if (sigmaDisplay) sigmaDisplay.textContent = msg.sigma.toFixed(4);

  // Get encoding config
  const includeGlobal = (document.getElementById('opt-global') as HTMLInputElement).checked;
  const includeKnockouts = (document.getElementById('opt-knockouts') as HTMLInputElement).checked;
  const includeMorphogens = (document.getElementById('opt-morphogens') as HTMLInputElement).checked;

  const config: EncodingConfig = {
    includeGlobal,
    includeKnockouts,
    includeMorphogens,
    geneNames,
    knockoutGeneIndices: knockoutCandidates,
  };

  // Decode and simulate to show expression
  const params = decode(new Float64Array(msg.bestParams), config);
  const simResult = engine.runSimulation(params, Math.PI, parseInt((document.getElementById('sim-time') as HTMLInputElement).value));
  bestSimulatedExpression = simResult.expression;
  bestParams = params;

  updateCharts();
}

function handleDone(msg: WorkerDoneMessage): void {
  bestParams = msg.decodedParams;

  // Run final simulation for display
  const simResult = engine.runSimulation(bestParams, Math.PI, parseInt((document.getElementById('sim-time') as HTMLInputElement).value));
  bestSimulatedExpression = simResult.expression;

  updateResults();
  updateCharts();
  stopOptimization();

  // Update status
  const statusLabel = document.getElementById('status-label');
  if (statusLabel) {
    statusLabel.textContent = msg.result.converged ? 'Converged' : 'Done';
    statusLabel.classList.remove('running');
    statusLabel.classList.add('done');
  }
}

function stopOptimization(): void {
  if (worker) {
    worker.postMessage({ type: 'stop' });
    worker.terminate();
    worker = null;
  }

  isRunning = false;
  updateUIRunning(false, 0);
}

function updateUIRunning(running: boolean, maxGen: number): void {
  const startBtn = document.getElementById('btn-start') as HTMLButtonElement;
  const stopBtn = document.getElementById('btn-stop') as HTMLButtonElement;
  const statusLabel = document.getElementById('status-label');

  startBtn.disabled = running;
  stopBtn.disabled = !running;

  if (running) {
    startBtn.classList.add('running');
    startBtn.textContent = 'Running...';
    if (statusLabel) {
      statusLabel.textContent = 'Running';
      statusLabel.classList.add('running');
      statusLabel.classList.remove('done');
    }
  } else {
    startBtn.classList.remove('running');
    startBtn.textContent = 'Start Optimization';
    if (statusLabel && !statusLabel.classList.contains('done')) {
      statusLabel.textContent = 'Ready';
      statusLabel.classList.remove('running');
    }
  }
}

// Initialize config slider displays
function initializeConfigSliders(): void {
  const sliders = [
    { id: 'max-generations', valId: 'max-gen-val', suffix: '' },
    { id: 'pop-size', valId: 'pop-size-val', suffix: '' },
    { id: 'sim-time', valId: 'sim-time-val', suffix: 'h' },
  ];

  for (const { id, valId, suffix } of sliders) {
    const slider = document.getElementById(id) as HTMLInputElement;
    const valSpan = document.getElementById(valId);

    if (slider && valSpan) {
      slider.addEventListener('input', () => {
        valSpan.textContent = slider.value + suffix;
      });
    }
  }
}

// Import modal handling
function initializeImportModal(): void {
  const modal = document.getElementById('import-modal');
  const btnImport = document.getElementById('btn-import');
  const btnConfirm = document.getElementById('btn-import-confirm');
  const btnCancel = document.getElementById('btn-import-cancel');
  const closeBtn = modal?.querySelector('.modal-close');
  const textarea = document.getElementById('import-textarea') as HTMLTextAreaElement;

  function showModal() {
    modal?.classList.remove('hidden');
  }

  function hideModal() {
    modal?.classList.add('hidden');
  }

  btnImport?.addEventListener('click', showModal);
  btnCancel?.addEventListener('click', hideModal);
  closeBtn?.addEventListener('click', hideModal);

  btnConfirm?.addEventListener('click', () => {
    try {
      const json = JSON.parse(textarea.value);
      if (json.expression && Array.isArray(json.expression)) {
        targetExpression = new Float32Array(json.expression);
        // Re-initialize sliders with new values
        initializeTargetSliders();
      }
      if (json.weights && Array.isArray(json.weights)) {
        targetWeights = new Float32Array(json.weights);
      }
      hideModal();
      updateCharts();
    } catch (e) {
      alert('Invalid JSON format');
    }
  });
}

// Apply to trajectory button
function initializeApplyButton(): void {
  const btn = document.getElementById('btn-apply');
  btn?.addEventListener('click', () => {
    if (!bestParams) return;

    const encoded = encodeParamsURL(bestParams);
    window.location.href = `/trajectory.html?params=${encoded}`;
  });
}

// Export button
function initializeExportButton(): void {
  const btn = document.getElementById('btn-export');
  btn?.addEventListener('click', () => {
    if (!bestParams) return;

    const json = exportParamsJSON(bestParams);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'inferred-params.json';
    a.click();

    URL.revokeObjectURL(url);
  });
}

// Main initialization
function init(): void {
  initializeDefaultTarget();
  initializeTargetSliders();
  initializeGeneSelectionUI();
  initializeWeightControls();
  initializeAccordions();
  initializeCanvases();
  initializeConfigSliders();
  initializeImportModal();
  initializeApplyButton();
  initializeExportButton();

  // Initialize weights based on default preset
  updateWeights();

  // Button handlers
  document.getElementById('btn-start')?.addEventListener('click', startOptimization);
  document.getElementById('btn-stop')?.addEventListener('click', stopOptimization);

  updateCharts();
}

// Start when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
