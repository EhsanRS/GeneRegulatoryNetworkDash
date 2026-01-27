/**
 * CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer
 *
 * Ideal for this parameter inference problem because:
 * - Handles non-convex, multi-modal fitness landscapes
 * - Works well with ~50-100 dimensional parameter spaces
 * - No gradient required (our simulation is a black box)
 * - Self-adapts step sizes and search directions
 */

export interface CMAESConfig {
  dimensions: number;
  populationSize?: number;  // lambda - defaults to 4 + floor(3 * ln(n))
  initialMean?: Float64Array;  // Starting point in parameter space
  sigma: number;  // Initial step size (0.3-0.5 recommended for normalized params)
  bounds?: [number, number][];  // Optional bounds per dimension [min, max]
  maxGenerations?: number;
}

export interface CMAESResult {
  params: Float64Array;
  fitness: number;
  generation: number;
  converged: boolean;
}

export class CMAES {
  private n: number;  // Dimensions
  private lambda: number;  // Population size
  private mu: number;  // Number of parents
  private weights: Float64Array;  // Recombination weights
  private mueff: number;  // Variance effective selection mass

  // Strategy parameters
  private cc: number;  // Time constant for cumulation for C
  private cs: number;  // Time constant for cumulation for sigma control
  private c1: number;  // Learning rate for rank-one update
  private cmu: number;  // Learning rate for rank-mu update
  private damps: number;  // Damping for sigma

  // Dynamic state
  private mean: Float64Array;
  private sigma: number;
  private C: Float64Array;  // Covariance matrix (stored as flat array, n*n)
  private pc: Float64Array;  // Evolution path for C
  private ps: Float64Array;  // Evolution path for sigma
  private B: Float64Array;  // Eigenvectors of C (stored as flat array, n*n)
  private D: Float64Array;  // sqrt of eigenvalues
  private invsqrtC: Float64Array;  // C^(-1/2)

  private generation: number = 0;
  private counteval: number = 0;
  private bounds?: [number, number][];

  // Best solution tracking
  private bestParams: Float64Array;
  private bestFitness: number = Infinity;

  // For eigendecomposition
  private eigenUpdateInterval: number;
  private lastEigenUpdate: number = 0;

  constructor(config: CMAESConfig) {
    this.n = config.dimensions;

    // Default population size
    this.lambda = config.populationSize ?? Math.floor(4 + 3 * Math.log(this.n));
    this.mu = Math.floor(this.lambda / 2);

    // Compute weights
    this.weights = new Float64Array(this.mu);
    let sumWeights = 0;
    for (let i = 0; i < this.mu; i++) {
      this.weights[i] = Math.log(this.mu + 0.5) - Math.log(i + 1);
      sumWeights += this.weights[i];
    }
    // Normalize weights
    for (let i = 0; i < this.mu; i++) {
      this.weights[i] /= sumWeights;
    }

    // Compute mueff
    let sumWeightsSq = 0;
    for (let i = 0; i < this.mu; i++) {
      sumWeightsSq += this.weights[i] * this.weights[i];
    }
    this.mueff = 1 / sumWeightsSq;

    // Strategy parameter settings
    this.cc = (4 + this.mueff / this.n) / (this.n + 4 + 2 * this.mueff / this.n);
    this.cs = (this.mueff + 2) / (this.n + this.mueff + 5);
    this.c1 = 2 / ((this.n + 1.3) * (this.n + 1.3) + this.mueff);
    this.cmu = Math.min(
      1 - this.c1,
      2 * (this.mueff - 2 + 1 / this.mueff) / ((this.n + 2) * (this.n + 2) + this.mueff)
    );
    this.damps = 1 + 2 * Math.max(0, Math.sqrt((this.mueff - 1) / (this.n + 1)) - 1) + this.cs;

    // Initialize state
    this.mean = config.initialMean ? new Float64Array(config.initialMean) : new Float64Array(this.n).fill(0.5);
    this.sigma = config.sigma;
    this.bounds = config.bounds;

    // Initialize covariance matrix as identity
    this.C = new Float64Array(this.n * this.n);
    for (let i = 0; i < this.n; i++) {
      this.C[i * this.n + i] = 1;
    }

    // Initialize evolution paths
    this.pc = new Float64Array(this.n);
    this.ps = new Float64Array(this.n);

    // Initialize eigen decomposition
    this.B = new Float64Array(this.n * this.n);
    for (let i = 0; i < this.n; i++) {
      this.B[i * this.n + i] = 1;
    }
    this.D = new Float64Array(this.n).fill(1);
    this.invsqrtC = new Float64Array(this.n * this.n);
    for (let i = 0; i < this.n; i++) {
      this.invsqrtC[i * this.n + i] = 1;
    }

    this.eigenUpdateInterval = Math.floor(this.lambda / (10 * this.n * (this.c1 + this.cmu)));
    if (this.eigenUpdateInterval < 1) this.eigenUpdateInterval = 1;

    this.bestParams = new Float64Array(this.mean);
  }

  /**
   * Sample a population of candidate solutions
   */
  samplePopulation(): Float64Array[] {
    const population: Float64Array[] = [];

    for (let k = 0; k < this.lambda; k++) {
      // Sample from N(0, I)
      const z = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) {
        z[i] = this.randn();
      }

      // Transform: y = B * D * z
      const y = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) {
        y[i] = 0;
        for (let j = 0; j < this.n; j++) {
          y[i] += this.B[i * this.n + j] * this.D[j] * z[j];
        }
      }

      // x = mean + sigma * y
      const x = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) {
        x[i] = this.mean[i] + this.sigma * y[i];

        // Apply bounds if specified
        if (this.bounds) {
          const [min, max] = this.bounds[i] ?? [0, 1];
          x[i] = Math.max(min, Math.min(max, x[i]));
        }
      }

      population.push(x);
    }

    return population;
  }

  /**
   * Update the distribution based on evaluated fitness values
   * @param population - The sampled population
   * @param fitness - Fitness values (lower is better)
   */
  update(population: Float64Array[], fitness: number[]): void {
    this.generation++;
    this.counteval += population.length;

    // Sort by fitness (ascending - lower is better)
    const indices = fitness.map((f, i) => ({ f, i })).sort((a, b) => a.f - b.f);

    // Track best solution
    if (indices[0].f < this.bestFitness) {
      this.bestFitness = indices[0].f;
      this.bestParams = new Float64Array(population[indices[0].i]);
    }

    // Compute new mean from best mu individuals
    const oldMean = new Float64Array(this.mean);
    this.mean.fill(0);
    for (let i = 0; i < this.mu; i++) {
      const idx = indices[i].i;
      for (let j = 0; j < this.n; j++) {
        this.mean[j] += this.weights[i] * population[idx][j];
      }
    }

    // Apply bounds to mean
    if (this.bounds) {
      for (let i = 0; i < this.n; i++) {
        const [min, max] = this.bounds[i] ?? [0, 1];
        this.mean[i] = Math.max(min, Math.min(max, this.mean[i]));
      }
    }

    // Cumulation for sigma control (ps)
    const psNext = new Float64Array(this.n);
    const meanShift = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) {
      meanShift[i] = (this.mean[i] - oldMean[i]) / this.sigma;
    }

    // ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * invsqrtC * (mean-oldMean)/sigma
    const invsqrtCTimesShift = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) {
      invsqrtCTimesShift[i] = 0;
      for (let j = 0; j < this.n; j++) {
        invsqrtCTimesShift[i] += this.invsqrtC[i * this.n + j] * meanShift[j];
      }
    }

    const csFactor = Math.sqrt(this.cs * (2 - this.cs) * this.mueff);
    for (let i = 0; i < this.n; i++) {
      this.ps[i] = (1 - this.cs) * this.ps[i] + csFactor * invsqrtCTimesShift[i];
    }

    // Compute |ps|^2
    let psNorm = 0;
    for (let i = 0; i < this.n; i++) {
      psNorm += this.ps[i] * this.ps[i];
    }
    psNorm = Math.sqrt(psNorm);

    // Expected value of |N(0,I)|
    const chiN = Math.sqrt(this.n) * (1 - 1 / (4 * this.n) + 1 / (21 * this.n * this.n));

    // Heaviside function for stalling
    const hsig = psNorm / Math.sqrt(1 - Math.pow(1 - this.cs, 2 * this.generation)) / chiN < 1.4 + 2 / (this.n + 1) ? 1 : 0;

    // Cumulation for covariance matrix (pc)
    const ccFactor = Math.sqrt(this.cc * (2 - this.cc) * this.mueff);
    for (let i = 0; i < this.n; i++) {
      this.pc[i] = (1 - this.cc) * this.pc[i] + hsig * ccFactor * meanShift[i];
    }

    // Update covariance matrix
    // C = (1-c1-cmu)*C + c1*(pc*pc' + (1-hsig)*cc*(2-cc)*C) + cmu*sum(w_i*(x_i-oldMean)*(x_i-oldMean)')
    const factor1 = 1 - this.c1 - this.cmu + (1 - hsig) * this.c1 * this.cc * (2 - this.cc);

    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < this.n; j++) {
        // Rank-one update
        this.C[i * this.n + j] = factor1 * this.C[i * this.n + j] + this.c1 * this.pc[i] * this.pc[j];

        // Rank-mu update
        for (let k = 0; k < this.mu; k++) {
          const idx = indices[k].i;
          const yi = (population[idx][i] - oldMean[i]) / this.sigma;
          const yj = (population[idx][j] - oldMean[j]) / this.sigma;
          this.C[i * this.n + j] += this.cmu * this.weights[k] * yi * yj;
        }
      }
    }

    // Update sigma
    this.sigma *= Math.exp((this.cs / this.damps) * (psNorm / chiN - 1));

    // Eigendecomposition of C
    if (this.generation - this.lastEigenUpdate > this.eigenUpdateInterval) {
      this.eigenDecomposition();
      this.lastEigenUpdate = this.generation;
    }
  }

  /**
   * Perform eigendecomposition of covariance matrix
   * Using power iteration for simplicity (Jacobi would be more robust)
   */
  private eigenDecomposition(): void {
    // Simple eigendecomposition using Jacobi rotations
    // This is a simplified version - for production, use a proper library

    const A = new Float64Array(this.C);  // Copy C
    const V = new Float64Array(this.n * this.n);

    // Initialize V as identity
    for (let i = 0; i < this.n; i++) {
      V[i * this.n + i] = 1;
    }

    // Jacobi rotations
    const maxIter = 50;
    for (let iter = 0; iter < maxIter; iter++) {
      // Find largest off-diagonal element
      let maxVal = 0;
      let p = 0, q = 1;
      for (let i = 0; i < this.n; i++) {
        for (let j = i + 1; j < this.n; j++) {
          const val = Math.abs(A[i * this.n + j]);
          if (val > maxVal) {
            maxVal = val;
            p = i;
            q = j;
          }
        }
      }

      if (maxVal < 1e-10) break;

      // Compute rotation angle
      const app = A[p * this.n + p];
      const aqq = A[q * this.n + q];
      const apq = A[p * this.n + q];

      let theta: number;
      if (Math.abs(aqq - app) < 1e-10) {
        theta = Math.PI / 4;
      } else {
        theta = 0.5 * Math.atan2(2 * apq, aqq - app);
      }

      const c = Math.cos(theta);
      const s = Math.sin(theta);

      // Apply rotation to A
      for (let i = 0; i < this.n; i++) {
        if (i !== p && i !== q) {
          const aip = A[i * this.n + p];
          const aiq = A[i * this.n + q];
          A[i * this.n + p] = A[p * this.n + i] = c * aip - s * aiq;
          A[i * this.n + q] = A[q * this.n + i] = s * aip + c * aiq;
        }
      }

      const newApp = c * c * app - 2 * c * s * apq + s * s * aqq;
      const newAqq = s * s * app + 2 * c * s * apq + c * c * aqq;
      A[p * this.n + p] = newApp;
      A[q * this.n + q] = newAqq;
      A[p * this.n + q] = A[q * this.n + p] = 0;

      // Apply rotation to V
      for (let i = 0; i < this.n; i++) {
        const vip = V[i * this.n + p];
        const viq = V[i * this.n + q];
        V[i * this.n + p] = c * vip - s * viq;
        V[i * this.n + q] = s * vip + c * viq;
      }
    }

    // Extract eigenvalues and eigenvectors
    for (let i = 0; i < this.n; i++) {
      this.D[i] = Math.sqrt(Math.max(A[i * this.n + i], 1e-10));
    }

    // B = V (eigenvectors as columns)
    for (let i = 0; i < this.n * this.n; i++) {
      this.B[i] = V[i];
    }

    // Compute C^(-1/2) = B * D^(-1) * B'
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < this.n; j++) {
        this.invsqrtC[i * this.n + j] = 0;
        for (let k = 0; k < this.n; k++) {
          this.invsqrtC[i * this.n + j] += this.B[i * this.n + k] * (1 / this.D[k]) * this.B[j * this.n + k];
        }
      }
    }
  }

  /**
   * Sample from standard normal distribution (Box-Muller transform)
   */
  private randn(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Get the best solution found so far
   */
  getBest(): CMAESResult {
    return {
      params: new Float64Array(this.bestParams),
      fitness: this.bestFitness,
      generation: this.generation,
      converged: this.isConverged()
    };
  }

  /**
   * Get current mean (center of distribution)
   */
  getMean(): Float64Array {
    return new Float64Array(this.mean);
  }

  /**
   * Get current sigma (step size)
   */
  getSigma(): number {
    return this.sigma;
  }

  /**
   * Reset sigma to a new value (for fine-tuning phase)
   */
  resetSigma(newSigma: number): void {
    this.sigma = newSigma;
  }

  /**
   * Get current generation
   */
  getGeneration(): number {
    return this.generation;
  }

  /**
   * Check if optimization has converged
   */
  isConverged(): boolean {
    // Check if sigma is very small
    if (this.sigma < 1e-10) return true;

    // Check if covariance matrix condition number is too high
    const maxD = Math.max(...this.D);
    const minD = Math.min(...this.D);
    if (maxD / minD > 1e7) return true;

    return false;
  }

  /**
   * Get population size
   */
  getPopulationSize(): number {
    return this.lambda;
  }
}
