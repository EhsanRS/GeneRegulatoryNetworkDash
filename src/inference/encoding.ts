/**
 * Parameter encoding/decoding for CMA-ES optimization
 *
 * Maps between structured simulation parameters and flat vectors
 * that the optimizer can work with.
 */

export interface GlobalParams {
  hillN: number;      // Hill coefficient [1, 8]
  hillK: number;      // Hill threshold [0.1, 3.0]
  progBias: number;   // Progenitor bias [0, 0.2]
  linBias: number;    // Lineage bias [0, 0.1]
  progDecay: number;  // Progenitor decay [0.1, 0.5]
  linDecay: number;   // Lineage decay [0.05, 0.3]
  inhibitionMult: number;  // Inhibition multiplier [1, 5]
  morphogenTime: number;   // Morphogen activation time [1, 12]
  morphogenStrength: number;  // Morphogen strength [0.2, 1.5]
  initProgTF: number;  // Initial progenitor TF expression [0.5, 3]
  initLinTF: number;   // Initial lineage TF expression [0, 0.5]
  ligandBias: number;  // Ligand bias [0, 0.2]
  receptorBias: number;  // Receptor bias [0, 0.5]
}

export interface MorphogenPerLineage {
  enabled: boolean;
  strength: number;  // [0, 2]
}

export interface SimulationParams {
  global: GlobalParams;
  knockouts: Set<number>;  // Gene indices to knock out
  morphogens: MorphogenPerLineage[];  // Per-lineage morphogen settings
  geneModifiers: Map<number, number>;  // Gene index -> modifier (0=inhibited, 1=normal, 2=overexpressed)
}

export interface ParameterBounds {
  min: number;
  max: number;
  default: number;
}

// Global parameter definitions with bounds
export const GLOBAL_PARAM_BOUNDS: Record<keyof GlobalParams, ParameterBounds> = {
  hillN: { min: 1, max: 8, default: 4 },
  hillK: { min: 0.1, max: 3.0, default: 1.0 },
  progBias: { min: 0, max: 0.2, default: 0.08 },
  linBias: { min: 0, max: 0.1, default: 0 },
  progDecay: { min: 0.1, max: 0.5, default: 0.25 },
  linDecay: { min: 0.05, max: 0.3, default: 0.15 },
  inhibitionMult: { min: 1, max: 5, default: 2.5 },
  morphogenTime: { min: 1, max: 12, default: 4 },
  morphogenStrength: { min: 0.2, max: 1.5, default: 0.8 },
  initProgTF: { min: 0.5, max: 3, default: 1.5 },
  initLinTF: { min: 0, max: 0.5, default: 0 },
  ligandBias: { min: 0, max: 0.2, default: 0.05 },
  receptorBias: { min: 0, max: 0.5, default: 0.2 },
};

export const GLOBAL_PARAM_KEYS = Object.keys(GLOBAL_PARAM_BOUNDS) as (keyof GlobalParams)[];
export const NUM_LINEAGES = 6;

export interface EncodingConfig {
  includeGlobal: boolean;
  includeKnockouts: boolean;
  includeMorphogens: boolean;
  includeModifiers: boolean;  // Include gene modifiers (overexpression/inhibition)
  geneNames: string[];  // All gene names for knockout encoding
  knockoutGeneIndices?: number[];  // Specific genes to consider for knockouts (TFs, ligands, receptors)
  modifierGeneIndices?: number[];  // Specific genes to consider for modifiers
}

/**
 * Calculate total dimensions of the parameter vector
 */
export function getDimensions(config: EncodingConfig): number {
  let dims = 0;

  if (config.includeGlobal) {
    dims += GLOBAL_PARAM_KEYS.length;
  }

  if (config.includeKnockouts && config.knockoutGeneIndices) {
    dims += config.knockoutGeneIndices.length;
  }

  if (config.includeMorphogens) {
    dims += NUM_LINEAGES;  // One strength value per lineage
  }

  if (config.includeModifiers && config.modifierGeneIndices) {
    dims += config.modifierGeneIndices.length;  // Continuous [0, 2] per gene
  }

  return dims;
}

/**
 * Get bounds for all parameters as [min, max] tuples
 */
export function getBounds(config: EncodingConfig): [number, number][] {
  const bounds: [number, number][] = [];

  if (config.includeGlobal) {
    for (const key of GLOBAL_PARAM_KEYS) {
      const b = GLOBAL_PARAM_BOUNDS[key];
      bounds.push([b.min, b.max]);
    }
  }

  if (config.includeKnockouts && config.knockoutGeneIndices) {
    // Knockouts are continuous [0, 1], thresholded at 0.5
    for (let i = 0; i < config.knockoutGeneIndices.length; i++) {
      bounds.push([0, 1]);
    }
  }

  if (config.includeMorphogens) {
    for (let i = 0; i < NUM_LINEAGES; i++) {
      bounds.push([0, 2]);  // Morphogen strength
    }
  }

  if (config.includeModifiers && config.modifierGeneIndices) {
    // Gene modifiers: 0 = fully inhibited, 1 = normal, 2 = overexpressed
    for (let i = 0; i < config.modifierGeneIndices.length; i++) {
      bounds.push([0, 2]);
    }
  }

  return bounds;
}

/**
 * Get default parameter vector (center of search space)
 */
export function getDefaultVector(config: EncodingConfig): Float64Array {
  const dims = getDimensions(config);
  const vector = new Float64Array(dims);
  let idx = 0;

  if (config.includeGlobal) {
    for (const key of GLOBAL_PARAM_KEYS) {
      const b = GLOBAL_PARAM_BOUNDS[key];
      // Normalize to [0, 1] range
      vector[idx++] = (b.default - b.min) / (b.max - b.min);
    }
  }

  if (config.includeKnockouts && config.knockoutGeneIndices) {
    // Default: no knockouts (all genes active)
    for (let i = 0; i < config.knockoutGeneIndices.length; i++) {
      vector[idx++] = 1;  // 1 = active, will be thresholded
    }
  }

  if (config.includeMorphogens) {
    for (let i = 0; i < NUM_LINEAGES; i++) {
      vector[idx++] = 0.5;  // Default strength = 1.0 (normalized)
    }
  }

  if (config.includeModifiers && config.modifierGeneIndices) {
    // Default: normal activity (modifier = 1.0 -> normalized = 0.5)
    for (let i = 0; i < config.modifierGeneIndices.length; i++) {
      vector[idx++] = 0.5;  // 0.5 * 2 = 1.0 (normal activity)
    }
  }

  return vector;
}

/**
 * Encode structured parameters to flat vector
 */
export function encode(params: SimulationParams, config: EncodingConfig): Float64Array {
  const dims = getDimensions(config);
  const vector = new Float64Array(dims);
  let idx = 0;

  if (config.includeGlobal) {
    for (const key of GLOBAL_PARAM_KEYS) {
      const b = GLOBAL_PARAM_BOUNDS[key];
      const val = params.global[key];
      // Normalize to [0, 1]
      vector[idx++] = (val - b.min) / (b.max - b.min);
    }
  }

  if (config.includeKnockouts && config.knockoutGeneIndices) {
    for (const geneIdx of config.knockoutGeneIndices) {
      // 0 if knocked out, 1 if active
      vector[idx++] = params.knockouts.has(geneIdx) ? 0 : 1;
    }
  }

  if (config.includeMorphogens) {
    for (let i = 0; i < NUM_LINEAGES; i++) {
      const m = params.morphogens[i];
      // Normalize strength [0, 2] to [0, 1]
      vector[idx++] = m ? m.strength / 2 : 0.5;
    }
  }

  if (config.includeModifiers && config.modifierGeneIndices) {
    for (const geneIdx of config.modifierGeneIndices) {
      // Normalize modifier [0, 2] to [0, 1]
      const modifier = params.geneModifiers.get(geneIdx) ?? 1.0;
      vector[idx++] = modifier / 2;
    }
  }

  return vector;
}

/**
 * Decode flat vector to structured parameters
 */
export function decode(vector: Float64Array, config: EncodingConfig): SimulationParams {
  let idx = 0;

  // Global parameters
  const global: GlobalParams = {} as GlobalParams;
  if (config.includeGlobal) {
    for (const key of GLOBAL_PARAM_KEYS) {
      const b = GLOBAL_PARAM_BOUNDS[key];
      // Denormalize from [0, 1] to [min, max]
      const normalized = Math.max(0, Math.min(1, vector[idx++]));
      global[key] = b.min + normalized * (b.max - b.min);
    }
  } else {
    // Use defaults
    for (const key of GLOBAL_PARAM_KEYS) {
      global[key] = GLOBAL_PARAM_BOUNDS[key].default;
    }
  }

  // Knockouts
  const knockouts = new Set<number>();
  if (config.includeKnockouts && config.knockoutGeneIndices) {
    for (const geneIdx of config.knockoutGeneIndices) {
      const val = vector[idx++];
      // Threshold at 0.5: < 0.5 = knocked out
      if (val < 0.5) {
        knockouts.add(geneIdx);
      }
    }
  }

  // Morphogens
  const morphogens: MorphogenPerLineage[] = [];
  if (config.includeMorphogens) {
    for (let i = 0; i < NUM_LINEAGES; i++) {
      const normalized = Math.max(0, Math.min(1, vector[idx++]));
      morphogens.push({
        enabled: true,  // Always enabled, strength can be 0
        strength: normalized * 2,  // [0, 2] range
      });
    }
  } else {
    // Default morphogens
    for (let i = 0; i < NUM_LINEAGES; i++) {
      morphogens.push({ enabled: true, strength: 1.0 });
    }
  }

  // Gene modifiers
  const geneModifiers = new Map<number, number>();
  if (config.includeModifiers && config.modifierGeneIndices) {
    for (const geneIdx of config.modifierGeneIndices) {
      const normalized = Math.max(0, Math.min(1, vector[idx++]));
      const modifier = normalized * 2;  // [0, 2] range
      // Only store non-default modifiers (not exactly 1.0)
      if (Math.abs(modifier - 1.0) > 0.05) {
        geneModifiers.set(geneIdx, modifier);
      }
    }
  }

  return { global, knockouts, morphogens, geneModifiers };
}

/**
 * Get human-readable parameter names for display
 */
export function getParameterNames(config: EncodingConfig): string[] {
  const names: string[] = [];

  if (config.includeGlobal) {
    for (const key of GLOBAL_PARAM_KEYS) {
      names.push(formatParamName(key));
    }
  }

  if (config.includeKnockouts && config.knockoutGeneIndices) {
    for (const geneIdx of config.knockoutGeneIndices) {
      names.push(`KO_${config.geneNames[geneIdx] ?? `Gene${geneIdx}`}`);
    }
  }

  if (config.includeMorphogens) {
    for (let i = 0; i < NUM_LINEAGES; i++) {
      names.push(`Morphogen_Lin${i + 1}`);
    }
  }

  if (config.includeModifiers && config.modifierGeneIndices) {
    for (const geneIdx of config.modifierGeneIndices) {
      names.push(`Mod_${config.geneNames[geneIdx] ?? `Gene${geneIdx}`}`);
    }
  }

  return names;
}

function formatParamName(key: string): string {
  // Convert camelCase to readable format
  return key.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase()).trim();
}

/**
 * Format parameter values for display
 */
export function formatParams(params: SimulationParams, config: EncodingConfig): Record<string, string> {
  const result: Record<string, string> = {};

  if (config.includeGlobal) {
    for (const key of GLOBAL_PARAM_KEYS) {
      result[formatParamName(key)] = params.global[key].toFixed(2);
    }
  }

  if (config.includeKnockouts) {
    const knockedOut = Array.from(params.knockouts).map(i => config.geneNames[i] ?? `Gene${i}`);
    result['Knockouts'] = knockedOut.length > 0 ? knockedOut.join(', ') : 'None';
  }

  if (config.includeMorphogens) {
    for (let i = 0; i < NUM_LINEAGES; i++) {
      const m = params.morphogens[i];
      result[`Morphogen Lin${i + 1}`] = m ? m.strength.toFixed(2) : '1.00';
    }
  }

  if (config.includeModifiers) {
    const modifiers: string[] = [];
    for (const [geneIdx, modifier] of params.geneModifiers) {
      const geneName = config.geneNames[geneIdx] ?? `Gene${geneIdx}`;
      const label = modifier < 0.5 ? 'Inhibited' : modifier > 1.5 ? 'Overexpr' : 'Mod';
      modifiers.push(`${geneName}: ${modifier.toFixed(2)}`);
    }
    result['Gene Modifiers'] = modifiers.length > 0 ? modifiers.join(', ') : 'None';
  }

  return result;
}

/**
 * Export parameters as JSON string
 */
export function exportParamsJSON(params: SimulationParams): string {
  return JSON.stringify({
    global: params.global,
    knockouts: Array.from(params.knockouts),
    morphogens: params.morphogens,
    geneModifiers: Array.from(params.geneModifiers.entries()),
  }, null, 2);
}

/**
 * Import parameters from JSON string
 */
export function importParamsJSON(json: string): SimulationParams {
  const data = JSON.parse(json);
  return {
    global: data.global,
    knockouts: new Set(data.knockouts),
    morphogens: data.morphogens,
    geneModifiers: new Map(data.geneModifiers ?? []),
  };
}

/**
 * Encode parameters as URL-safe base64
 */
export function encodeParamsURL(params: SimulationParams): string {
  const json = JSON.stringify({
    g: params.global,
    k: Array.from(params.knockouts),
    m: params.morphogens,
    mod: Array.from(params.geneModifiers.entries()),
  });
  // Use btoa for base64 encoding
  return btoa(json);
}

/**
 * Decode parameters from URL-safe base64
 */
export function decodeParamsURL(encoded: string): SimulationParams {
  try {
    const json = atob(encoded);
    const data = JSON.parse(json);
    return {
      global: data.g,
      knockouts: new Set(data.k),
      morphogens: data.m,
      geneModifiers: new Map(data.mod ?? []),
    };
  } catch {
    throw new Error('Invalid parameter encoding');
  }
}
