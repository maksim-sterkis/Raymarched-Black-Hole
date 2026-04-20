import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ==============================================================================
// SHADER DEFINITIONS
// ==============================================================================

const vertexShader = `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = vec4(position, 1.0);
            }
        `;

const fragmentShader = `
            uniform float u_time;
            uniform vec2 u_resolution;
            uniform vec3 u_cameraPos;
            uniform vec3 u_cameraDir;
            uniform vec3 u_cameraUp;
            uniform vec3 u_cameraRight;
            uniform sampler2D u_cubemap;
            uniform float u_spin; // New: Kerr Spin Parameter (0.0 to 0.998)
            uniform float u_tilt; // New: System Tilt Angle

            #define MAX_STEPS 800
            #define BASE_STEP_SIZE 0.05
            #define GM 0.5

            // --- MATH UTILS ---
            mat3 rotZ(float angle) {
                float s = sin(angle);
                float c = cos(angle);
                return mat3(
                    c, s, 0.0,
                    -s, c, 0.0,
                    0.0, 0.0, 1.0
                );
            }

            // --- PROCEDURAL NOISE ---
            float hash13(vec3 p3) {
                p3  = fract(p3 * .1031);
                p3 += dot(p3, p3.zyx + 31.32);
                return fract((p3.x + p3.y) * p3.z);
            }

            float noise(vec3 x) {
                vec3 i = floor(x);
                vec3 f = fract(x);
                f = f * f * (3.0 - 2.0 * f);

                return mix(mix(mix(hash13(i + vec3(0,0,0)), hash13(i + vec3(1,0,0)), f.x),
                               mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
                           mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x),
                               mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y), f.z);
            }

            float fbm(vec3 p) {
                float f = 0.0;
                float amp = 0.5;
                for(int i = 0; i < 2; i++) {
                    f += amp * noise(p);
                    p *= 2.02;
                    amp *= 0.5;
                }
                return f;
            }

            // --- TONE MAPPING & BLACKBODY ---
            vec3 ACESFilm(vec3 x) {
                float a = 2.51;
                float b = 0.03;
                float c = 2.43;
                float d = 0.59;
                float e = 0.14;
                return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
            }

            vec3 blackbody(float temp) {
                vec3 c = vec3(255.0);
                float t = clamp(temp, 1000.0, 40000.0) / 100.0;
                if (t <= 66.0) { c.r = 255.0; } else { c.r = clamp(329.698727446 * pow(t - 60.0, -0.1332047592), 0.0, 255.0); }
                if (t <= 66.0) { c.g = clamp(99.4708025861 * log(t) - 161.1195681661, 0.0, 255.0); } else { c.g = clamp(288.1221695283 * pow(t - 60.0, -0.0755148492), 0.0, 255.0); }
                if (t >= 66.0) { c.b = 255.0; } else if (t <= 19.0) { c.b = 0.0; } else { c.b = clamp(138.5177312231 * log(t - 10.0) - 305.0447927307, 0.0, 255.0); }
                return c / 255.0;
            }

            // --- RK4 KERR GEODESIC ACCELERATION ---
            vec3 calcAcceleration(vec3 pos, vec3 vel) {
                float r2 = dot(pos, pos);
                float r = sqrt(r2);
                vec3 pos_hat = pos / r;

                vec3 h_vec = cross(pos, vel);
                float h2 = dot(h_vec, h_vec);

                // Base Schwarzschild gravity
                vec3 a_grav = -3.0 * GM * h2 / (r2 * r2 * r + 0.00001) * pos;

                // Kerr Frame Dragging (Gravitomagnetic approximation)
                float a_param = u_spin * GM;
                vec3 J = vec3(0.0, a_param * GM, 0.0); // Spin is aligned with Y axis

                // Gravitomagnetic field Bg (Lense-Thirring)
                vec3 Bg = (3.0 * dot(J, pos_hat) * pos_hat - J) / (r2 * r + 0.0001);

                // a_drag is proportional to v x Bg.
                vec3 a_drag = cross(vel, Bg) * 2.0;

                // --- Ergosphere Logic & Force Enforcement ---
                float cos_theta = pos.y / (r + 0.0001);
                float r_ergo = GM + sqrt(max(0.0, GM*GM - a_param*a_param * cos_theta*cos_theta));
                float r_plus = GM + sqrt(max(0.0, GM*GM - a_param*a_param));

                if (r < r_ergo) {
                    // Inside the ergosphere, space itself rotates faster than light.
                    // We artificially boost the frame-dragging to violently force co-rotation on photons.
                    float drag_boost = 1.0 + 2.0 * smoothstep(r_ergo, r_plus, r);
                    a_drag *= drag_boost;
                } else {
                    // Slight boost outside to ensure the D-shaped shadow is visible
                    a_drag *= 1.8;
                }

                return a_grav + a_drag;
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

                vec3 ro = u_cameraPos;
                vec3 cw = normalize(u_cameraDir);
                vec3 cu = normalize(u_cameraRight);
                vec3 cv = normalize(u_cameraUp);
                vec3 rd = normalize(uv.x * cu + uv.y * cv + 1.2 * cw);

                // --- APPLY SYSTEM TILT ---
                mat3 tiltMat = rotZ(u_tilt);
                ro = tiltMat * ro;
                rd = tiltMat * rd;

                vec3 p = ro;

                // --- STOCHASTIC DITHERING ---
                // Generates a random offset to break up banding/wood-grain artifacts
                float dither = hash13(vec3(gl_FragCoord.xy, u_time));
                p += rd * dither * BASE_STEP_SIZE;

                vec3 col = vec3(0.0);
                float transmittance = 1.0;
                bool hitBlackHole = false;
                float diskThickness = 0.1;

                // --- KERR METRIC CONSTANTS ---
                float a_star = u_spin;
                float a_param = u_spin * GM;

                // 1. Dynamic Event Horizon (r_+)
                float r_plus = GM + sqrt(max(0.0, GM*GM - a_param*a_param));

                // 2. Dynamic ISCO (Bardeen, Press, Teukolsky)
                float z1 = 1.0 + pow(max(0.0, 1.0 - a_star*a_star), 1.0/3.0) * (pow(1.0 + a_star, 1.0/3.0) + pow(max(0.0, 1.0 - a_star), 1.0/3.0));
                float z2 = sqrt(3.0 * a_star*a_star + z1*z1);
                float r_isco = GM * (3.0 + z2 - sqrt(max(0.0, (3.0 - z1)*(3.0 + z1 + 2.0*z2))));

                for(int i = 0; i < MAX_STEPS; i++) {
                    if (i == MAX_STEPS - 1) { hitBlackHole = true; break; }

                    float r = length(p);

                    // Cross the outer event horizon
                    if (r < r_plus) {
                        hitBlackHole = true;
                        break;
                    }

                    if (r > 100.0) break;

                    // Accretion disk exists outside the Event Horizon to allow Plunge Region rendering
                    bool inDisk = abs(p.y) < diskThickness && r > r_plus && r < 25.0;

                    float currentStep;
                    if (inDisk) {
                        currentStep = 0.02; // Increased local sampling density for smoother clouds
                    } else {
                        // Adaptive stepping scales based on distance to the new dynamic horizon
                        currentStep = BASE_STEP_SIZE * clamp((r - r_plus) * 0.8, 0.1, 5.0);
                        if (p.y * rd.y < 0.0) {
                            float distToDisk = abs(p.y) - diskThickness;
                            // Clamp lower bound to 0.02 to match inDisk density
                            if (distToDisk > 0.0) currentStep = min(currentStep, max(0.02, distToDisk * 0.8));
                        }
                    }

                    if (inDisk) {
                        float verticalFade = smoothstep(diskThickness, 0.0, abs(p.y));
                        // Power-law density falloff detached from ISCO, with soft outer edge
                        float radialFade = pow(3.0 / r, 1.0) * smoothstep(25.0, 15.0, r);
                        float fadeProduct = verticalFade * radialFade;

                        if (fadeProduct > 0.001) {
                            float timeDilation = max(0.001, 1.0 - (2.0*GM) / r);
                            float twistAngle = -u_time * (3.0 / r) * timeDilation;
                            mat2 twistRot = mat2(cos(twistAngle), -sin(twistAngle), sin(twistAngle), cos(twistAngle));
                            vec2 swirledXZ = twistRot * p.xz;
                            vec3 pRot = vec3(swirledXZ.x, p.y * 3.0, swirledXZ.y);

                            float n = fbm(pRot * 2.0);
                            n = pow(n, 2.5);

                            float density = n * fadeProduct * 12.0;
                            if (r < r_isco) {
                                // Drop the density down to a tiny fraction so the gap becomes a dark void
                                density *= smoothstep(r_plus, r_isco, r) * 0.15;
                            }

                            if (density > 0.0) {
                                // 1. Kerr Orbital Velocity (Prograde)
                                vec3 velocityDir = normalize(vec3(-p.z, 0.0, p.x));
                                float v_mag = sqrt(GM / r) / (1.0 + a_star * pow(GM / r, 1.5));
                                v_mag = min(v_mag, 0.999);

                                // 2. Relativistic Doppler Shift
                                float v_proj = dot(rd, velocityDir) * v_mag;
                                float invGamma = sqrt(1.0 - v_mag * v_mag);
                                float dopplerShift = invGamma / (1.0 + v_proj);

                                // 3. Gravitational Redshift (Approximation in equatorial plane)
                                float gravRedshift = sqrt(max(0.05, 1.0 - r_plus / r));

                                float D = dopplerShift * gravRedshift;
                                float beaming = pow(D, 3.0);
                                float alpha = 1.0 - exp(-density * currentStep * 2.0);

                                // Standardized Temperature Profile detached from ISCO
                                float baseTemp = 6500.0 * pow(3.0 / max(r, 1.0), 1.5);
                                float observedTemp = baseTemp * D;
                                vec3 gasCol = blackbody(observedTemp);

                                // Boosted emission multiplier for better visibility
                                vec3 emission = gasCol * density * beaming * 1.5;
                                col += transmittance * emission * currentStep;
                                transmittance *= (1.0 - alpha);
                            }
                        }
                    }

                    float dt = currentStep;
                    if (r > 25.0) {
                        p += rd * dt;
                    } else {
                        vec3 k1_p = rd;
                        vec3 k1_v = calcAcceleration(p, rd);

                        vec3 k2_p = rd + 0.5 * dt * k1_v;
                        vec3 k2_v = calcAcceleration(p + 0.5 * dt * k1_p, k2_p);

                        vec3 k3_p = rd + 0.5 * dt * k2_v;
                        vec3 k3_v = calcAcceleration(p + 0.5 * dt * k2_p, k3_p);

                        vec3 k4_p = rd + dt * k3_v;
                        vec3 k4_v = calcAcceleration(p + dt * k3_p, k4_p);

                        p += (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p);
                        rd += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
                        rd = normalize(rd);
                    }

                    if (i > MAX_STEPS - 40) transmittance *= 0.75;
                    if (transmittance < 0.01) break;
                }

                if (!hitBlackHole) {
                    // Un-tilt the ray to match the static background skybox
                    mat3 invTilt = rotZ(-u_tilt);
                    vec3 world_rd = invTilt * rd;

                    #define PI 3.14159265359
                    vec2 bg_uv = vec2(atan(world_rd.z, world_rd.x) / (2.0 * PI) + 0.5, asin(clamp(world_rd.y, -1.0, 1.0)) / PI + 0.5);
                    vec3 bgCol = texture2D(u_cubemap, bg_uv).rgb;
                    bgCol = pow(bgCol, vec3(2.2)) * 4.0;

                    float obs_r = length(u_cameraPos);
                    float bg_redshift = sqrt(max(0.0001, 1.0 - (2.0*GM) / obs_r));

                    // Relaxed background redshift for better outer edge visibility
                    bgCol *= bg_redshift;

                    col += transmittance * bgCol;
                }

                col = ACESFilm(col);
                col = pow(col, vec3(1.0 / 2.2));
                gl_FragColor = vec4(col, 1.0);
            }
        `;

// ==============================================================================
// THREE.JS SETUP
// ==============================================================================

const renderer = new THREE.WebGLRenderer({ antialias: false });
// Set pixel ratio to 1. We will handle dynamic internal resolution manually.
renderer.setPixelRatio(1);

document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();

const renderCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const geometry = new THREE.PlaneGeometry(2, 2);

const textureLoader = new THREE.TextureLoader();
const milkyWayTexture = textureLoader.load('https://unpkg.com/three-globe/example/img/night-sky.png');
milkyWayTexture.wrapS = THREE.RepeatWrapping;
milkyWayTexture.wrapT = THREE.ClampToEdgeWrapping;
milkyWayTexture.minFilter = THREE.LinearFilter;
milkyWayTexture.magFilter = THREE.LinearFilter;
milkyWayTexture.colorSpace = THREE.SRGBColorSpace;

const material = new THREE.ShaderMaterial({
    vertexShader: vertexShader,
    fragmentShader: fragmentShader,
    uniforms: {
        u_time: { value: 0.0 },
        u_resolution: { value: new THREE.Vector2() }, // Set dynamically by resize()
        u_cameraPos: { value: new THREE.Vector3() },
        u_cameraDir: { value: new THREE.Vector3() },
        u_cameraUp: { value: new THREE.Vector3() },
        u_cameraRight: { value: new THREE.Vector3() },
        u_cubemap: { value: milkyWayTexture },
        u_spin: { value: 0.99 }, // Default near-maximal spin
        u_tilt: { value: 15.0 * Math.PI / 180.0 } // Default tilt
    },
    depthWrite: false,
    depthTest: false
});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

const virtualCamera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100);
// Adjusted initial camera zoom position
virtualCamera.position.set(0, 4.0, 15.0);

const controls = new OrbitControls(virtualCamera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.rotateSpeed = 0.5;
controls.autoRotate = false;
controls.enablePan = false;
controls.minDistance = 2.5;
// Increased maximum zoom distance
controls.maxDistance = 40.0;

const clock = new THREE.Clock();

// Handle UI Controls
const spinSlider = document.getElementById('spinSlider');
const spinValue = document.getElementById('spinValue');

spinSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    spinValue.innerText = val.toFixed(3);
    material.uniforms.u_spin.value = val;
});

const tiltSlider = document.getElementById('tiltSlider');
const tiltValue = document.getElementById('tiltValue');

tiltSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    tiltValue.innerText = val + '°';
    material.uniforms.u_tilt.value = val * Math.PI / 180.0;
});

// --- DYNAMIC RESOLUTION SCALER ---
// Target a fixed number of pixels (e.g. 400,000 is ~854x480) for rock-solid FPS.
// The browser will automatically upscale this internal buffer to fit the screen.
const TARGET_PIXELS = 400000;

function resize() {
    const cssWidth = window.innerWidth;
    const cssHeight = window.innerHeight;
    const aspect = cssWidth / cssHeight;

    // Calculate internal buffer size to maintain a constant total pixel count
    let internalHeight = Math.sqrt(TARGET_PIXELS / aspect);
    let internalWidth = internalHeight * aspect;

    // Cap at native physical resolution to prevent scaling UP on low-res screens
    const maxNativeWidth = cssWidth * window.devicePixelRatio;
    if (internalWidth > maxNativeWidth) {
        internalWidth = maxNativeWidth;
        internalHeight = cssHeight * window.devicePixelRatio;
    }

    internalWidth = Math.round(internalWidth);
    internalHeight = Math.round(internalHeight);

    // false = Do NOT update the CSS style.
    // This allows the canvas to stay 100vw/100vh while rendering at the lower internal resolution.
    renderer.setSize(internalWidth, internalHeight, false);

    virtualCamera.aspect = aspect;
    virtualCamera.updateProjectionMatrix();
    material.uniforms.u_resolution.value.set(internalWidth, internalHeight);
}

window.addEventListener('resize', resize);

// Trigger immediately to initialize the correct resolution
resize();

function animate() {
    requestAnimationFrame(animate);

    controls.update();
    virtualCamera.updateMatrixWorld();

    const elements = virtualCamera.matrixWorld.elements;
    material.uniforms.u_time.value = clock.getElapsedTime();
    material.uniforms.u_cameraPos.value.copy(virtualCamera.position);
    material.uniforms.u_cameraRight.value.set(elements[0], elements[1], elements[2]).normalize();
    material.uniforms.u_cameraUp.value.set(elements[4], elements[5], elements[6]).normalize();
    material.uniforms.u_cameraDir.value.set(-elements[8], -elements[9], -elements[10]).normalize();

    renderer.render(scene, renderCamera);
}

animate();