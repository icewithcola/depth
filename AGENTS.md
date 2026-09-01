# Guidelines for AI Agents (AGENTS.md)

This document establishes operational rules, architectural standards, coding conventions, and testing policies for AI agents working in this repository (`icewithcola/depth`).

---

## 1. Environment & Command Execution (Nix Shell)

The host system runs on Nix / NixOS. Standard tools like `node` and `npm` are provided through the **Nix Flake devShell** (`flake.nix`) and are **not** present in the global host `PATH`.

### Command Execution Rules
- **Always** run Node/npm commands through `nix develop --command ...`.
- **Never** execute bare `npm`, `npx`, `tsc`, `vite`, or `vitest` directly.

### Essential Commands
```sh
# Install dependencies
nix develop --command npm install

# Start local HTTPS development server (WebGPU requires HTTPS or localhost)
nix develop --command npm run dev

# Run automated tests
nix develop --command npm test

# Run TypeScript type check
nix develop --command npm run typecheck

# Build production bundle to dist/
nix develop --command npm run build

# Download the MoGe-2 ONNX model asset (public/models/moge-2-vits-normal.onnx)
nix develop --command npm run model:download
```

---

## 2. Test Necessity & "Anti-Test-Bloat" Policy

### Core Principle
**Do not write useless unit tests.**

Avoid test bloat and "AI-slop" tests that mock large portions of the browser/DOM or test superficial glue code. Every test file and test case in this repository must provide a high signal-to-noise ratio and genuine regression protection for complex logic.

### ❌ What NOT to Test (Forbidden / Discouraged)
- **DOM & Event Listener Plumbing**: Do not write hundreds of lines mocking `document.getElementById`, `addEventListener`, `click()`, `preventDefault()`, `dialog.showModal()`, or CSS class toggling (e.g., mock-heavy tests for `DepthApp`).
- **Standard / Public Math & Textbook Algorithms**: Do not write unit tests for standard, public, textbook, or well-known mathematical formulas (e.g. standard percentiles/quantiles, basic linear interpolation, standard vector/matrix arithmetic, basic trigonometry). If an algorithm is standard and well-understood, skip testing it.
- **Mock-Heavy Wrapper Tests**: Do not test functions where 90% of the code is setting up mock dependencies only to verify that `expect(mockFn).toHaveBeenCalled()`. If you are testing your mocks rather than real logic, delete the test.
- **Trivial Browser API Passthroughs**: Do not write synthetic tests verifying standard web APIs (e.g. `ClipboardItem`, `navigator.clipboard.writeText`) unless there is non-trivial domain transformation logic.
- **Trivial Getters/Setters/Boilerplate**: Do not write tests for simple property accessors or constant definitions.

### ✅ What to Test (Only High-Risk, Non-Trivial Domain Transformations)
Tests are strictly reserved for **project-specific, non-trivial computer vision / graphics transformations and regression protection**:
- **Domain-Specific Coordinate Conversions & Camera Calibration**:
  - Converting MoGe OpenCV camera space ($+X$ right, $+Y$ down, $+Z$ forward) into Three.js WebGL space ($+X$ right, $+Y$ up, $-Z$ forward).
  - Off-axis perspective projection matrix derivation from normalized intrinsic matrix $K$ (`createProjectionMatrix`).
  - Coupled affine focal length & depth shift least-squares solver (`recoverFocalShift`).
- **Custom Mesh Topology & Discontinuity Logic**:
  - Relative depth edge threshold detection (`depthMapEdge`).
  - Disocclusion tear handling & triangle rejection guards.
  - Largest connected component extraction (`largestConnectedComponentMask`).
- **Critical Numerical Edge Cases & Regressions**:
  - Handling `NaN`, `Infinity`, and invalid model sentinels in point clouds.

### Guidelines for Writing Tests
1. **Rule of Thumb**: When in doubt, **do not write a test**. Only write a test when there is a complex, project-specific domain algorithm with genuine risk of regression.
2. **Prefer Pure Functions**: If testing is truly needed, test pure functions directly with raw numerical inputs and outputs with zero mocks.
3. **Keep Tests Fast & Lightweight**: Tests must execute in milliseconds with zero async mock overhead.

---

## 3. Code Style & Architecture

### Repository Architecture
- `src/moge/`: MoGe-2 ONNX inference pipeline, image preprocessing (resizing, padding, token counting), postprocessing (affine recovery), and pure geometric math. Pure math should remain decoupled from DOM and WebGPU state.
- `src/scene/`: Three.js rendering, WebGL2 context, camera projection matrix derivation, depth-aware mesh generation, and spatial camera controllers.
- `src/platform/`: Environment checks, WebGPU hardware adapter diagnostics, OS clipboard utilities.
- `src/app.ts`, `src/main.ts`: Application orchestrator, DOM element wiring, and UI event lifecycle.

### TypeScript Conventions
- **Strict Typing**: The project enforces strict compiler settings (`strict: true`, `noUncheckedIndexedAccess: true`, `exactOptionalPropertyTypes: true`, `noUnusedLocals: true`, `noUnusedParameters: true`).
- **No `any`**: Use explicit domain interfaces or generics. Never introduce `any` to bypass type errors.
- **TypedArrays for Geometry**: Always use TypedArrays (`Float32Array`, `Uint8Array`, `Uint32Array`, `Int32Array`) for coordinates, vertex buffers, depth buffers, normals, and masks to maximize performance and avoid memory pressure.
- **Explicit Range & Finite Checks**: Check array bounds and guard against `NaN` / `Infinity` when dealing with model outputs or user inputs.

### Code Organization & Formatting
- **2-Space Indentation & Semicolons**: Standard TypeScript formatting with trailing semicolons and single quotes.
- **Lifecycle Management**: Any class managing WebGPU inference sessions, Three.js textures/geometries/materials, or DOM event handlers must implement a complete `dispose()` method.
- **Domain Terminology**: Use standard computer vision and 3D graphics naming conventions (e.g. `intrinsics`, `fov`, `clipPlanes`, `disocclusion`, `depthMapEdge`).
- **Zero Extraneous Dependencies**: Prefer native Web APIs and typed math utilities over adding heavy external npm libraries.

---

## 4. Agent Verification Checklist

Before submitting changes or completing a task, execute the verification suite in sequence:

1. **Type Check**:
   ```sh
   nix develop --command npm run typecheck
   ```
   *(Must pass with zero TypeScript diagnostics).*

2. **Automated Tests**:
   ```sh
   nix develop --command npm test
   ```
   *(All tests must pass. Ensure no unnecessary mock bloat was introduced).*

3. **Production Build**:
   ```sh
   nix develop --command npm run build
   ```
   *(Vite production build must succeed).*
