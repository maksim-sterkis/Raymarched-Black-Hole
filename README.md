# Kerr Black Hole Raymarcher

A real-time, browser-based relativistic raymarcher implemented in a single fragment shader using WebGL, Three.js, and GLSL. This project simulates the extreme gravitational lensing, frame-dragging, and Doppler-beaming effects of a spinning (Kerr) black hole, coupled with a two-pass HDR cinematic post-processing pipeline.

### Link [Live Interactive Demo](https://maksim-sterkis.github.io/Raymarched-Black-Hole/)

---

## Features

* **Relativistic Raymarching:** Simulates photon trajectories bending through highly curved spacetime.
* **Two-Pass HDR Pipeline:** Renders raw physics calculations into a `HalfFloatType` off-screen frame buffer before tone mapping to preserve high-dynamic-range luminosity.
* **Cinematic Post-Processing:** Complete screen-space lens pipeline including:
  * Golden Ratio spiral bloom (eliminates grid artifacts).
  * Anamorphic scifi lens flares.
  * Distance-squared chromatic aberration.
  * ACES Filmic tonemapping & gamma correction.
* **Performance Budgeting:** Built-in resolution preset system (including the lightweight *Ultimate Potato* and *Digital Watch* modes) allowing interactive frame rates alongside heavy numerical integration loops.

---

## The Physics & Mathematical Blueprint

To render the black hole in real-time at interactive frame rates, the simulation employs a high-fidelity **pseudo-Newtonian gravitomagnetic vector field** rather than evaluating the complete, computationally crushing Einstein Field Equations. 

### 1. Photon Path Integration (RK4)
Photons are stepped through space using a **Runge-Kutta 4th Order (RK4)** integration scheme. At each step, the photon's position $\vec{x}$ and velocity $\vec{v}$ are updated by evaluating the local acceleration field four times to minimize numerical drift.

### 2. Gravitational Acceleration
The core lensing effect uses a modified Einstein-Infeld-Hoffmann-inspired acceleration vector to mimic general relativity:

$$\vec{a}_{\text{grav}} = -\frac{3GMh^2}{r^5}\vec{r}$$

Where:
* $G$ is the gravitational constant, and $M$ is the black hole mass ($GM = 0.5$).
* $h^2$ is the specific angular momentum squared of the photon orbit ($\vec{h} = \vec{r} \times \vec{v}$).
* $r$ is the Euclidean distance to the singularity.

### 3. Kerr Frame Dragging (Ergosphere Approximation)
To capture the rotational twist of a Kerr black hole, a **gravitomagnetic drag force** is introduced based on the spin parameter ($a = u\_spin \cdot GM$). The angular momentum of the spinning singularity generates a gravitomagnetic dipole field $\vec{B}_g$:

$$\vec{B}_g = \frac{3(\vec{J} \cdot \hat{r})\hat{r} - \vec{J}}{r^3}$$

The dragging acceleration applied to the photon is calculated using a relativistic velocity cross-product:

$$\vec{a}_{\text{drag}} = 2 (\vec{v} \times \vec{B}_g)$$

Inside the ergosphere ($r < r_{\text{ergo}}$), this dragging torque is boosted smoothly up to the event horizon ($r_+$) to simulate the frame-dragging effect that forces light to circulate in the direction of the black hole's spin.

---

## Engineering Tradeoffs & Approximations

To make this simulation run in a browser tab instead of a supercomputer cluster, several brilliant graphics optimizations were made:

1. **Vector Field Force Field vs. Boyer-Lindquist Geodesics:** True Kerr raytracing requires solving four coupled, non-linear, second-order differential equations using Christoffel symbols for null geodesics. This simulation replaces that with an explicit acceleration force field. This achieves a visually identical spatial warping effect at a fraction of the GPU processing cost.
2. **Procedural Accretion Disk (Domain Warped FBM):** The gas disk uses a 6-octave Fractal Brownian Motion (FBM) noise pattern. To simulate the extreme shear forces of the differential rotation near the Innermost Stable Circular Orbit (ISCO), the noise space is warped using a time-dilated twist function: $\theta_{\text{twist}} = -t \cdot \frac{3}{r} \cdot \left(1 - \frac{2GM}{r}\right)$.
3. **Adaptive Ray Stepping:** Instead of uniform steps, the raymarching loop adaptively tightens step sizes near the photon sphere ($r \approx 2.0$) and disk boundaries where warping is most severe, while taking larger steps in empty asymptotic space.

---

## Local Development

Since this entire simulation is contained within a single HTML file, you don't need a heavy local development environment:

1. Clone the repository: `git clone https://github.com/maksim-sterkis/Raymarched-Black-Hole.git`
2. Open `index.html` in any modern web browser that supports WebGL2.
3. Use the control panel to tweak parameters such as spin, tilt, disk turbulence, and camera effects.

*Pro-tip: If the simulation lags on your GPU, drop the resolution to **Ultimate Potato** to explore the scene, pause the simulation, turn the resolution up to **Ultra**, and let the engine render a pristine, mathematically complete screenshot frame.*
