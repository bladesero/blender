EEVEE Vulkan DLSS Reconstruction
===============================

Material-independent SR (2026-10-12)
------------------------------------

SR eligibility is independent of material nodes. The complete native low-resolution
image, including deferred refraction, forward transparency, Shader to RGB, SSS,
volume integration, Holdout, motion blur and DOF, is the SR input. Hardware ray
tracing and RR material evaluation are not prerequisites. The historical validation
sections later in this document describe older, narrower implementations.

The reference-depth prepass runs at input resolution with pixel jitter but without
lens jitter. It uses the original depth-material graph and geometry deformation.
Transparency uses a deterministic per-fragment opacity threshold of 0.5; rejected
fragments leave the background guide intact. A second depth-tested replay after
clearing velocity prevents a static foreground from inheriting background motion.
These private guides never replace depth used by native shading. Refraction uses
surface motion; volumes use the underlying surface or sky. A single depth/motion
pair cannot describe every path through layered or refractive media.

The SR current-color bias is a heuristic, not a correctness guarantee. It combines
refraction closure coverage, color changes caused by volume/forward/post-effect
composition, partial output coverage, reprojected depth changes and relative RGB
changes. History sampling compensates projection jitter. Animated nodes do not
need a separate material compiler. The SDK's reserved transparency mask is not used.

SR color/coverage integration is material-independent. Viewport edits, time changes
and camera movement restart the running mean; ordinary SR motion history remains
available. Final shutter substeps do not restart the mean. Pixel jitter is removed
for accumulation; lens and shutter effects remain in the image. Motion-blurred final
frames use their native sample budget and one reset SR evaluation after integration,
explicitly reported as ``single-evaluation``. They do not reuse animation-frame
history. Cancellation and native retry restore the original scene time.

Coverage is converted from native Combined transmittance and reconstructed separately
with locally bounded bicubic interpolation. Output remains scene-linear premultiplied
RGBA, including Holdout; neither experimental NGX alpha scaling nor division by small
alpha is used. Current coverage is used while navigating/animating. This preserves
alpha semantics but does not promise full-resolution edge detail or perfectly matched
RGB/alpha filtering in all scenes. RR continues to output opaque alpha.

States distinguish Off, Preparing, Active, Unavailable and Failed. Shader compilation
keeps the saved request active and resets history on completion. Material features
never select native SR fallback. Unsupported workflows and device/runtime/resource
failures still do; failed final reconstruction is discarded before native retry.
With ``--debug-gpu``, input dimensions, samples, evaluation counts and reset reasons
are logged separately from successful NGX execution.

Reproducible acceptance runner::

  python tests/python/eevee_dlss_sr_materials.py --blender PATH/TO/blender.exe --output captures/sr
  python tests/python/eevee_dlss_sr_materials.py --blender PATH/TO/blender.exe --output captures/native --mode NATIVE --samples 256
  python tests/python/eevee_dlss_sr_materials.py --blender PATH/TO/blender.exe --output captures/motion --case MIXED --viewport --frames 300
  python tests/python/eevee_dlss_sr_materials.py --blender PATH/TO/blender.exe --output captures/f12 --case COMBINED --f12

Use ``--quality BALANCED`` / ``PERFORMANCE``, ``--orthographic``, ``--dof-jitter``, and ``--sequence
80,81,80`` for the corresponding regressions. The runner saves fixtures, linear EXRs,
float arrays, PNGs and GPU logs. It requires actual successful SR at a smaller input
size, rejects fallback and output-resolution transparency, checks finite output and
alpha, and checks Vulkan synchronization diagnostics. ``--profile`` enables the
existing NGX-only timestamp and SDK VRAM report. GPU tests
``GPUVulkanTest.dlss_sr_bias_lifetime`` and ``dlss_sr_bias_validation`` cover optional
SR input reuse, missing masks, resource lifetime and invalid dimensions.

Use ``--reference captures/native`` to record linear RGB, alpha and white-background
composite mean absolute errors. ``--dump-inputs --dump-sample 32`` saves the last
32-sample progressive SR inputs; use ``--dump-sample 1`` for shutter integration.
Transparent fixtures also save black/white composite PNGs. To test native retry,
render a native reference at the same sample budget and pass ``--inject-failure
--reference PATH`` to the SR runner. This requires a discarded failed reconstruction,
a complete native retry, matching alpha (absolute tolerance 0.002) and RGB MAE below
0.001. Native lighting caches can make the retry differ slightly from a cold render.

The focused opacity-guide fixture is run inside Blender::

  blender --background --factory-startup --gpu-backend vulkan --debug-gpu --python-exit-code 1 --python tests/python/eevee_dlss_sr_guides.py -- --output captures/guides

Motion-blur cancellation uses the windowed job test::

  blender --factory-startup --gpu-backend vulkan --debug-gpu --python tests/python/eevee_dlss_cancel.py -- --output captures/cancel --sr --motion-blur

Validation snapshot (RTX 4070 SUPER, driver 610.88)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Release compilation and the following real-GPU checks passed on 2026-10-12:

* 18 material/effect fixtures at Quality, Balanced and orthographic Performance
  (54 renders). Every SR case executed successful NGX at a smaller input size,
  without material fallback, output-resolution transparency or validation errors.
* 300 captured dynamic mixed-material viewport frames, with 639 successful NGX
  evaluations, plus a separate 13-capture toggle/navigation/quality/split-view test.
* Five transparent-guide thresholds (0, 0.4, 0.5, 0.6, 1), including depth readback
  and zero motion for final frames; asynchronous F12, 80/81/80 repeatability,
  motion-blur cancellation/time restoration and injected NGX failure/native retry.
* Explicit lens jitter: an 8-sample request used 38 progressive SR evaluations to
  preserve complete lens patterns and the SR minimum. With shutter integration it
  used the native 14-sample effective budget and exactly one SR evaluation per frame.
* 15 Vulkan GPU/render-graph tests; SR/RR data-pass compatibility (six cases);
  RR 1/2/1 frame repeatability (maximum repeat error 0).

At 320x240, the updated mixed fixture with Quality SR and 32 samples had RGB MAE
0.007049 against native 256 samples. Alpha-fixture errors were RGB MAE 0.008425,
alpha MAE 0.006620 and white-composite MAE 0.011987. These are measurements, not
universal acceptance thresholds. Visual comparison shows softer thin features and
more low-sample shadow noise; transparent edges do not match native 256-sample output.
The combined motion-blur failure retry had exact alpha and RGB MAE 0.000159 against
native 32 samples after restoring native volume-grid dimensions.

For the 32-evaluation mixed Quality run, NGX timestamps totaled 9.743 ms, with a
0.290 ms steady evaluation mean. SDK-reported memory was 30,143,204 bytes; sampled
whole-device peak was 4547 MiB (includes other applications). These are NGX-only
timings, not total GPU render time or a speedup claim. Full GPU timing remains
unmeasured: RenderDoc injection caused NGX initialization failure (0xbad00002), so
its fallback captures were rejected as SR measurements. The runner explicitly stores
``total_gpu_ms: null`` rather than substituting CPU time or native fallback timing.

Machine-readable results are summarized in ``eevee_dlss_sr_validation.json``.
Local evidence is under ``E:/gitrepo/dlss_universal_validation``: EXRs, float arrays,
fixture blend files, input/output readbacks, 300 viewport PNGs and validation logs.
This is a functional regression baseline, not exhaustive perceptual certification:
the 300-frame sequence does not cover every material/quality/effect permutation,
and additional GPUs/drivers and true animated simulation caches remain untested.

Rendering successfully is distinct from perceptual acceptance. Compare captures with
native high-sample output and composite transparent images over both black and white.
Inspect long sequences for trails, disocclusion, flicker and lost detail; do not infer
quality or a performance improvement from NGX success alone.

Scope
-----

This experimental path reconstructs the Combined viewport or final-render image before
color management. It uses NVIDIA NGX 310.4.0 directly: SR transformer preset K and RR
transformer preset D. Windows, Vulkan, RTX and EEVEE are required. Cycles and frame
generation are outside its scope. Implementation and visual acceptance are separate;
the validation records below identify which configurations have actually been tested.

Build and use
-------------

On Windows x64, obtain the official NVIDIA DLSS SDK separately. SDK 310.4.0 is the
integration baseline. Its headers, static libraries and signed runtime are not vendored
in Blender's source tree. Configure with::

  cmake -S blender -B build -DWITH_VULKAN_BACKEND=ON -DWITH_NVIDIA_DLSS=ON -DNVIDIA_DLSS_ROOT=C:/SDK/DLSS
  cmake --build build --config Release --target blender
  cmake --install build --config Release

The SDK directory must contain ``include/nvsdk_ngx_helpers_vk.h``, the dynamic CRT
libraries under ``lib/Windows_x86_64/x64`` and ``lib/Windows_x86_64/rel/nvngx_dlss.dll``.
RR additionally uses ``include/nvsdk_ngx_helpers_dlssd_vk.h`` and the optional signed
``lib/Windows_x86_64/rel/nvngx_dlssd.dll``. Runtimes are copied next to the executable.
Missing RR support falls back to native denoising with SR. Debug builds use the iterator-debugging-0
variant. ``WITH_NVIDIA_DLSS`` defaults to OFF; ordinary builds do not need the SDK.

Start Blender with ``--gpu-backend vulkan``. Select EEVEE, use a rendered viewport,
then select Quality, Balanced or Performance under Render Properties > Sampling >
Viewport > DLSS Super Resolution. The corresponding RNA property is
``scene.eevee.dlss_quality``. Off restores native EEVEE. The requested quality is saved
in the blend file; support is checked locally at runtime.

Native fallback is used for unsupported GPUs/backends/builds, missing runtime, NGX
failure, non-Combined passes, render regions, XR, viewport compositing,
and independent Grease Pencil rendering. The viewport information text describes
fallback. Viewport image renders retain their native path. An unsupported configuration
does not silently change the saved user setting.

Final renders use the separate ``scene.eevee.dlss_render_quality`` setting. RR switches
are ``scene.eevee.dlss_rr`` (viewport) and ``scene.eevee.dlss_render_rr`` (final). All new
settings default to off, and RR requires quality other than Off. Final output supports
Combined only; data passes, AOVs, Cryptomatte, multiview and render borders fall back.
SR evaluates all EEVEE materials at input resolution, including refraction, volumes,
Holdout, Shader to RGB, SSS, Translucent, Dithered and Blended surfaces. Transparent
film, DOF and motion blur no longer select native SR fallback. All surface/volume
color is composited before SR; there is no output-resolution material shading.
RR retains its existing restrictions on motion blur, DOF, volumes, transparent film,
Holdout and Shader to RGB.
RR uses mixed tracing: lightweight PBR and transparency use hardware queries;
SSS, Translucent, glass and complex graphs retain native screen/probe shading without
blocking RR. Unsupported geometry remains a frame-level restriction. Primary materials
are always evaluated by EEVEE; no saved material or image is rewritten.

Overscan expands the NGX target and camera view; output adapters crop back to the exact
requested dimensions. Render border, panoramic cameras and multiview remain unsupported.

Pipeline and lifetime
---------------------

``DLSSModule`` owns shared reconstruction and guides. ``DLSSViewportAdapter`` owns
viewport camera history and display framebuffer access; ``DLSSRenderAdapter`` owns
final linear RGBA output. Final reconstruction does not require a window or ``rv3d``.

* NGX supplies the input size for the selected output size and quality.
* EEVEE shades at input resolution. SR retains native denoising. RR traces at input
  resolution. A per-ray ownership image bypasses spatial, temporal and bilateral
  denoising only for hardware-resolved RR rays. Native rays retain the user-selected
  denoising stages; filters reject RR-owned neighbors/history.
  Lighting composition, material weights and world-space illumination caches remain.
* RR guides are captured before GBuffer and ray-time pool release. Evaluated GBuffer
  reflection color already contains integrated BRDF reflectance and closure weights.
  Normals are world-space, roughness is packed in alpha, and hit distance is in world
  units (the environment sentinel is converted to zero). RR replaces the SR call.
* A compute pass converts reverse Z to conventional depth, resolves dense camera and
  object motion, converts motion to input-pixel units, and flips inputs to top-left
  origin. Motion excludes projection jitter.
* The Vulkan external-compute graph node transitions images to GENERAL, declares all
  input/output dependencies, records NGX, and invalidates cached Vulkan bindings.
  Its memory access masks cover SDK-internal transfer clears/copies as well as shaders.
* DLSS replaces Film's color accumulation/reprojection for the active viewport.
  SR combines its high-resolution RGB with independently reconstructed coverage;
  RR retains opaque alpha. Depth for overlays is
  sampled separately; it is not reconstructed by DLSS.
* UI and overlays remain at output resolution. Display color management remains in
  Blender's existing viewport pipeline.
* The feature handle, textures and history belong to a Film instance. Different
  viewports never share a feature handle or parameter map. SDK initialization is shared.
* Size/quality changes and destruction wait for pending GPU work before releasing NGX
  resources. Normal frames do not introduce a CPU/GPU completion wait.
* A viewport NGX failure produces a temporary defined image and selects native EEVEE
  on the next update. Final rendering discards the failed reconstruction and restarts
  once at native resolution. Temporary scaled images are never accepted as final output.

Final render behavior
---------------------

Each output frame, view and View Layer creates independent history. Without motion blur,
SR/RR evaluates once per sample. Static-frame motion is zero; subpixel jitter is passed
separately. Render samples are raised to at least 32, shown in render progress; SR rounds
to complete native lens sampling patterns. With motion blur, SR instead keeps the native
sampling budget, integrates all shutter samples first, then evaluates once with reset.
This single-evaluation mode has no SR history and can resolve less detail.
The final reconstructed sample bypasses Film TAA. A compute adapter restores bottom-left
orientation and premultiplied RGBA before Film Combined texture lookup/readback, Render Result,
ordinary Combined compositing, PNG or linear EXR output. No history is borrowed from
another animation frame, including when rendering frames in reverse order.

Temporal behavior
-----------------

Jitter uses a persistent Halton sequence in input-pixel units, independent of EEVEE's
navigation sample resets. For top-left NGX inputs, projection translation becomes
``(jitter.x, -jitter.y)``. History is reset on activation, size/quality changes, camera
identity/projection changes, backward or skipped timeline frames, large camera changes,
and depsgraph edits at a fixed timeline time. Ordinary navigation and consecutive
animation frames use motion reprojection. History decisions use the current synced
camera and are committed only for rendered samples, not display-only redraws.

Validation
----------

Run ``VKRenderGraphTestCompute.external_compute_dependencies_and_rebind`` in
``blender_test``. This checks producer/consumer synchronization and re-binding of the
same pipeline and descriptor set after external compute.

The viewport test must run in a windowed process, not ``--background``::

  blender --factory-startup --gpu-backend vulkan --debug-gpu --python tests/python/eevee_dlss_viewport.py -- --output captures

It saves composited PNG captures and a JSON report for native rendering, all SR modes, navigation,
settling, camera cuts, object motion, overlays, fallback, re-enabling, viewport splitting
and disabling. Also run it with ``--gpu-backend opengl`` for backend fallback. Inspect
the console for ``DLSS SR`` lines with successful NGX status and both retained/reset
history, and for Vulkan validation errors. A finite/non-empty capture alone is not proof
that DLSS executed or that temporal image quality is acceptable.

On Windows, enable the installed Khronos validation layer and synchronization validation
in the process environment before starting the test::

  $env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
  $env:VK_LAYER_VALIDATE_SYNC = '1'

Add ``--hardware-raytracing`` after the script's ``--output`` argument to test SR together
with this branch's EEVEE hardware ray tracing. The test also exits with live SR handles
in two viewports to exercise GPU completion and SDK teardown.

Inspect thin geometry, checkerboard detail, disocclusion, moving silhouettes and overlays
against native rendering. Production acceptance also needs representative content with
hair, transparency and animated materials; those have more demanding motion/coverage
semantics. This first version does not claim full-resolution overlay-depth accuracy.

Original SR baseline validation (2026-10-11, before RR/final-render extension)
---------------------------------------------------------------------------

Release build, SDK 310.4.0, RTX 4070 SUPER, NVIDIA driver 610.88:

* Vulkan render-graph tests: 51 passed, one existing scheduler case skipped.
* Vulkan SR: 13 captures, 1,141 successful NGX evaluations, retained and reset history.
* Vulkan SR with hardware RT: 13 captures, 1,109 successful NGX evaluations, two
  independent active viewports, disable/re-enable, and clean exit with live features.
* Both Vulkan runs had zero validation errors with synchronization validation enabled.
  Existing unused fragment-output warnings also appear before SR is enabled.
* OpenGL fallback: 13 captures, no NGX evaluations, clean exit.
* A 64x64 final render with Quality selected completed on native EEVEE without NGX.

Visual inspection caught and corrected an inverted jitter sign. Static checkerboard
edges and thin bars now resolve without that artifact. Stochastic lighting noise remains:
SR does not substitute for Ray Reconstruction or high-sample offline rendering. These
checks establish the integration baseline, not production acceptance across all content.

References
----------

* https://github.com/NVIDIA/DLSS/tree/v310.4.0
* https://github.com/NVIDIA/DLSS/blob/v310.4.0/doc/DLSS_Programming_Guide_Release.pdf
* https://github.com/NVIDIA/DLSS/blob/v310.4.0/doc/DLSS-RR%20Integration%20Guide.pdf

RR/final-render diagnostics
---------------------------

Add ``--rr`` to the viewport runner for the supported opaque RR subset. The final-render
runner writes PNG, EXR, raw arrays and timing records::

  blender -b --factory-startup --gpu-backend vulkan --debug-gpu --python tests/python/eevee_dlss_render.py -- --output captures --mode RR --sequence 1,2,1

Repeat with ``--mode SR``, ``--mode NATIVE --samples 256``, reverse frame order and each
``--fallback`` case. Repeat without ``-b`` for windowed final output. Compare the GPU logs,
not just images: successful ``DLSS RR`` and ``DLSS RR tracing`` lines confirm actual RR
evaluation and native ray-denoiser bypass. ``external_rr_inputs_and_reuse`` tests the seven
input dependencies and subsequent guide reuse in the Vulkan render graph.

``BLENDER_DLSS_DUMP_DIR`` opts into first-sample guide readback (width/height/channels in
``.txt``, float32 data in ``.f32``). Production has no guide readback. With ``--debug-gpu``,
``BLENDER_DLSS_TEST_FAIL_AT=3`` injects an NGX failure at evaluation index 3 to exercise the
native retry. Unset these variables for performance measurement.

``BLENDER_DLSS_PROFILE=1`` records Vulkan timestamps for the first 64 NGX evaluations
and reports SDK allocated VRAM at feature destruction. This measures the NGX phase,
not total EEVEE GPU time. Query pools are absent when profiling is disabled.

Validate dumped guides with ``tests/python/eevee_dlss_guides.py``. Run the asynchronous
F12 job and cancellation/recovery harness in a windowed process::

  blender --factory-startup --gpu-backend vulkan --debug-gpu --python tests/python/eevee_dlss_cancel.py -- --output captures --complete

Add ``--sr`` for SR. Omit ``--complete`` to inject cancellation after three samples,
then render another frame to check recovery. ``BLENDER_DLSS_TEST_CANCEL_AT`` is a
debug-GPU-only hook through the normal render cancellation gate; this test does not
simulate a physical Escape key. Use the same saved ``.blend`` for cross-process
determinism comparisons (``--save-fixture`` and ``--use-existing-scene`` in the render
runner), rather than separately generating a new fixture in each process.

RR and final-render baseline validation (2026-10-11, before material extension)
----------------------------------------------------------------------------

Release build and installation succeeded with SDK 310.4.0 on RTX 4070 SUPER,
driver 610.88. The final implementation retains experimental status.

* Render-graph suite: 52 passed, one pre-existing scheduler test skipped.
* Final viewport regressions: 13 SR captures and 15 RR captures, including all quality
  levels, motion, camera changes, live RR/SR switching, independent split viewports,
  fallback, reactivation and destruction with active features.
* Actual NGX SR/RR success and RR native-denoiser bypass were checked in logs.
  Selected Vulkan runs with synchronization validation had no validation errors.
* Final SR and RR completed through background rendering and the asynchronous F12
  render job. Quality, Balanced and Performance produced the requested output size.
  PNG, linear EXR and Combined-only exposure compositing were checked.
* Identical saved-scene RR frames rendered in forward order, reverse order and a
  separate single-frame process were bit-identical. Windowed versus background
  output was also bit-identical. These results are specific to this fixed environment.
* Mid-render NGX fault injection discarded reconstruction and retried natively once;
  its output was bit-identical to the matching native render. Cancellation followed
  by a fresh render completed without stale results or validation errors.
* Twelve native fallback cases passed: transparent film, depth pass, DOF, border,
  AOV, Cryptomatte, motion blur, transmission, normal-mapped unsupported hardware-RT
  material, volume, point cloud and multiview. Missing RR runtime retained SR;
  missing SR runtime and OpenGL used native rendering.
* Guide readback checked finite values, unit world normals, roughness range,
  reflectance, hit-distance/background conventions and zero final-frame motion.

The opaque PBR fixture includes diffuse, metallic, mirror and rough reflections,
thin bars, moving geometry and camera changes. Captures were visually inspected for
orientation, gross ghosting, reflection stability and noise. Against a native
256-sample reference, linear RGB mean absolute error was 0.0104024 for SR and
0.00689949 for RR. This scalar metric does not establish general perceptual quality.

At 1920x1080 and 32 samples, this simple fixture measured:

====================== =========== =========== ===========
Measurement            Native      SR          RR
====================== =========== =========== ===========
Render wall time       0.548 s     1.244 s     1.274 s
NGX steady GPU mean    n/a         0.653 ms    2.373 ms
Sampled GPU peak       4072 MiB    3990 MiB    4025 MiB
SDK allocated VRAM     n/a         94.0 MB     160.1 MB
====================== =========== =========== ===========

Wall time includes initialization/readback/output overhead. GPU means exclude the
first evaluation. Memory peaks are sampled whole-device ``nvidia-smi`` values with
approximately 2395 MiB baseline; they are not exact per-process peaks. Native was
faster on this fixture. Total EEVEE GPU timing and exact allocation peaks remain
unmeasured, and no speedup is claimed.

Local evidence is in ``E:/gitrepo/dlss_rr_validation/validation_summary.json``, with
images, raw arrays, guide dumps and per-case logs beside it. Older failed harness
iterations remain in that directory; the summary explicitly selects final runs.

Outstanding acceptance includes physical non-RTX hardware, broad production scenes,
long animation/disocclusion sequences and exhaustive perceptual review. At this baseline,
normal-mapped and layered secondary-hit materials selected native rendering; successful
fallback is not acceptance of those materials under RR. Other unsupported effects and
data workflows likewise retain the native path. These limits prevent claiming full
production visual acceptance from API success alone.


Material compatibility extension
--------------------------------

Historical baseline: the SR transparency split and closure-specific integration below
are superseded by the material-independent SR implementation described at the end.

SR and RR use separate capability checks. Diagnostics list stable reason codes, actual
mode, View Layer/pass names and material/object names when known. Shader compilation is
allowed to finish before evaluating material restrictions; placeholder materials must
not permanently force a native retry. Viewport text summarizes the first reason and the
console reports the full list when it changes. New settings remain off in old files.

Blended surfaces are excluded from low-resolution SR input. Their radiance and RGB
transmittance are evaluated at the expanded output resolution against a freshly rasterized
opaque/Dithered depth buffer. The final composition is ``L + T * reconstructed_background``
in linear HDR. Viewports use four spatial samples per update, with no previous-frame
transparent history. Final renders average L and T within the output frame. Screen ray-cast
nodes combined with Blended surfaces remain unsupported. This path has additional depth,
lighting and memory costs and does not claim a speedup.

Dithered coverage, SSS and thin Translucent shading require static Monte Carlo convergence
in addition to SR. Their SR color input therefore has a scene-linear running mean on an
unjittered grid, with cubic resampling into the current jittered sample. Camera movement,
depsgraph changes, time changes, mode/size changes and new output frames reset this mean.
RR does not use it. It substantially reduces stochastic transparency/shadow noise but can
soften small details; this is not equivalent to native full-resolution accumulation.
Final output still uses the last SR result and never runs Film TAA over that result.

Secondary-hit Normal Map evaluation supports Tangent, Object, World and Blender legacy
spaces, OpenGL/DirectX conventions, Strength, explicit UV Map nodes and independent texture
and tangent UV sets. Mesh corner normals and MikkTSpace tangents match raster interpolation,
including mirrored UVs and nonuniform/negative scale. Geometry normals remain separate for
ray bias and geometry checks. Missing UV/tangent data and unsupported node chains are
reported, not silently replaced with Base Color or geometric normals. Texture format,
source restrictions still apply. The former 64 MiB secondary-image storage pool is
replaced by the texture-array foundation described below.

Debug values 23 and 24 show secondary shading normals and absolute component error
against GBuffer normals. They are also available in final renders with ``--debug-gpu``.
Boundary pixels can compare different surfaces and are excluded from interior error
measurements. RR guides continue to use evaluated GBuffer normals; maps are not applied twice.

Run compatibility fixtures with::

  blender --factory-startup -b --gpu-backend vulkan --debug-gpu --python-exit-code 1 --python tests/python/eevee_dlss_compatibility.py -- --case BLENDED --overscan 3 --output captures
  blender --factory-startup -b --gpu-backend vulkan --debug-gpu --python-exit-code 1 --python tests/python/eevee_dlss_compatibility.py -- --case NORMAL_UV --mode RR --output captures

The viewport runner accepts ``--compatibility BLENDED|DITHERED|SSS|NORMAL`` and
``--overscan``. The F12/cancellation runner accepts ``--blended`` or ``--normal``.
Use ``--profile-gpu`` in a dedicated output directory for debug-group CPU timings,
including ``DLSS.OutputTransparency.Depth`` and ``DLSS.OutputTransparency.Shade``.
This Vulkan backend does not emit GPU durations for these groups. Only the NGX phase
has explicit Vulkan timestamp measurements through ``BLENDER_DLSS_PROFILE=1``.


Material extension validation (2026-10-11)
------------------------------------------

Release build/install succeeded from base ``a95482d2c79`` plus this working-tree
extension, using the same RTX 4070 SUPER / driver 610.88 / SDK 310.4.0 environment.
Selected authoritative runs and exact metrics are listed in
``E:/gitrepo/dlss_compat_validation/validation_summary.json``. Earlier exploratory
runs in that directory are intentionally excluded from the summary.

* SR background fixtures passed for Dithered, Blended (including colored
  transmittance), SSS and thin Translucent materials. Odd 641x361 output, 10 percent
  overscan, orthographic cameras and all quality modes were exercised.
* Normal-mapped RR executed successfully with native denoising bypass. Interior
  component error against raster GBuffer normals was about 0.00005--0.00052 on
  Tangent/Object/World/legacy fixtures, including mirrored UVs, independent normal
  texture/tangent UV sets and negative/nonuniform scale. DirectX decoding, Strength
  textures, animated deformation and material edits also executed successfully.
* The normal-mapped object in the secondary-hit fixture is behind the camera and
  appears only in the mirror. Changing Strength changes the reflection (mean RGBA
  absolute difference 0.0140367), demonstrating secondary-hit evaluation rather
  than only a changed primary GBuffer guide.
* Missing UVs, degenerate tangents and Bump produce explicit reasons and native
  fallback. OpenGL retains native rendering. Vulkan synchronization validation had
  no errors in the selected runs. Existing unconsumed-fragment-output warnings also
  occur in native EEVEE runs and are not claimed to be eliminated.
* Viewport SR regressions passed for ordinary, Blended, Dithered and SSS fixtures
  (13 captures each). Normal-map RR passed 17 captures, including live RR/SR,
  hardware-tracing disable/re-enable, quality/size changes, split views and active
  feature destruction.
* Both Blended SR and normal-mapped RR passed asynchronous F12 completion and
  cancellation followed by recovery. PNG and linear EXR contain finite HDR RGBA
  with the requested dimensions and opaque alpha.
* Blended frames from the same saved file are bit-identical in forward/reverse
  order. Repeated frames after deformation, material edits, Dithered changes and
  SSS also match exactly in this environment. The Blended injected NGX failure
  restarts native rendering once; compared with a fresh native process its RGBA
  MAE is 0.00000703 and maximum difference 0.0112305. This case is not bit-identical
  and is recorded separately from the frame-order tests.
* The supplied character file renders through actual SR with original materials
  and 3 percent overscan after disabling extra passes in a separate test copy.
  Keeping its original passes reports the View Layer and Normal/Diffuse/Glossy/
  Transmission pass names and falls back. The source file was not saved or changed.
  ``character_ready/fixture.blend`` enables SR with Combined-only output at
  1440x1920 and 64 samples. It is an SR test copy, not a supported full-character RR
  scene. At 720x960/32 samples the SR RGB MAE against native 256 samples was
  0.00494604, versus 0.0110403 before static color integration. Fine detail remains
  softer than native; the metric does not establish perceptual equivalence.
* The render-graph regression suite remains 52 passed / one pre-existing skip;
  Python harness syntax and ``git diff --check`` passed.

A separate 1920x1080, 32-sample Blended fixture with 3 percent overscan measured
1.326 seconds native versus 2.104 seconds SR, including initialization/readback and
file/image processing. NGX Vulkan timestamps measured 0.721 ms steady mean per
SR evaluation (23.510 ms total over 32 calls). SDK allocations were 108.8 MB.
Sampled whole-device memory peaked at 4217 MiB native and 4263 MiB SR, against
approximately 2523--2533 MiB baseline; these are not exact per-process peaks.
The Vulkan debug-group profiler reports CPU durations only in this backend, so
transparent-pass and total EEVEE GPU durations remain unmeasured. No acceleration
claim follows from these simple-scene results.

This extension remains experimental. Broad production scenes, long moving
transparency/disocclusion sequences and physical non-RTX hardware still need
acceptance. The mirror fixture also exposes existing hardware-RT shadow stair-step
artifacts. RR SSS/transparency/layered materials and other unsupported effects or
data workflows continue to use native rendering; their support is not implied by
the Normal Map extension. Missing-runtime checks belong to the earlier baseline
and were not repeated after these material-only changes.

Historical full-material foundation (superseded scope)
----------------------------------------------------

The following records predate the lightweight mixed-tracing implementation below.
The proposed full secondary SSS/Translucent/glass evaluator was not accepted and is
outside the revised scope. Earlier native/SR character fallback was not an RR pass;
those historical results must not be confused with current mixed-tracing acceptance.

Implemented infrastructure:

* ``DLSSFrame`` has optional before-transparency color, premultiplied transparency
  radiance, RGB opacity, transparency motion, signed SSS guide, disocclusion and
  current-color bias inputs. Vulkan snapshots their descriptors, validates input
  sizes/pairs and includes every supplied image in the external-compute dependency
  graph. These fields do not establish that the runtime honors every guide.
* Mesh BLAS construction accepts an opaque flag. Changing that flag rebuilds the
  BLAS instead of attempting an invalid Vulkan update. The candidate-hit accessor
  exposes distance, identity, barycentrics and front-face state before commitment.
  EEVEE's production traversal remains opaque until material evaluation is ready.
* Secondary images share mipmapped GPU arrays by image identity, dimensions and
  precision. Byte images are uploaded as scene-linear RGBA16F and float images as
  RGBA32F without resolution reduction. There are eight descriptor groups and the
  device's array-layer limit, with explicit capacity/allocation failure. Image size
  changes leave reusable layer slots; binding is deferred until resources exist.
  The current secondary sampler still uses LOD zero; ray-footprint LOD selection
  is outstanding. Multilayer/animated/tiled sources remain unsupported.
* SSS capture surrounds the existing EEVEE convolution. It saves unweighted diffuse
  lighting in FP32 and writes signed FP16
  ``dot((after - before) * diffuse_albedo, (0.25, 0.5, 0.25))`` with zero outside
  SSS surfaces. With ``--debug-gpu`` and ``BLENDER_DLSS_DUMP_DIR``, SR can export
  this diagnostic without bypassing the RR material gate. Exported buffers use
  top-left row order. Secondary-hit SSS is not implemented by this capture.

Tests and the transparency contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``GPUVulkanTest.dlss_rr_optional_inputs_lifetime`` exercises all optional inputs,
resource reuse/aliasing and active feature destruction. Invalid input sizes and
unpaired transparency inputs are rejected. The render-graph test covers fourteen
RR input images and reuse of all seven auxiliary guides. The non-opaque ray-query
test rejects a foreground candidate, accepts the background triangle and switches
BLAS opacity twice with unchanged topology.

The opt-in ``GPUVulkanTest.dlss_rr_transparency_probe`` exports raw RGBA32F outputs
and a CSV to ``BLENDER_DLSS_PROBE_DIR``. Put the SDK's ``nvngx_dlss.dll`` and
``nvngx_dlssd.dll`` beside ``blender_test.exe`` before running it. The variants are:

0. Fully composited noisy color; no transparency overlay.
1. Background color plus premultiplied RGBA16F overlay and separate RGB opacity.
2. Variant 1 with inverted separate opacity.
3. Background plus native three-channel R11G11B10F overlay/opacity.
4. Variant 3 with inverted separate opacity.
5. Variant 3 with deliberately wrong overlay radiance.
6. Variant 1 with inverted overlay alpha.
7. Deliberately double-counted overlay: composited color plus the overlay again.

On RTX 4070 SUPER / driver 610.88 / SDK 310.4.0 / RR preset D, NGX succeeds, but
variants 1/2 are bit-identical, as are 3/4. Changing overlay radiance or its alpha
does affect output. Thus this experiment has **not validated separate RGB opacity**.
It does not prove that every possible SDK configuration ignores that resource.
No colored-transparency material gate is relaxed on the strength of API success.
The probe passing means execution/export succeeded, not that transparency semantics
or visual quality passed. Its per-variant comparison requires separate inspection.

The material expression/closure compiler, secondary SSS and thin transmission,
production transparent traversal/shadows, transparent motion/history rejection,
solid/thin glass transport and overflow fallback remain outstanding. The original
character RR, F12/animation material acceptance, 300-frame motion sequence and
quality/performance comparisons therefore remain unaccepted.

Foundation validation on 2026-10-11 used the same RTX/driver/SDK configuration:
58 selected Vulkan GPU/render-graph tests passed, with one existing scheduler skip.
Twelve background frames covered normal-map RR, four 2048-square image-array RR,
SSS SR and Blended SR, each in frame order 1,2,1. All 384 expected NGX evaluations
succeeded; the selected logs contained no Vulkan validation errors. Repeated frame
1 outputs were bit-identical, with finite PNG/EXR output at 320x240. The array test
also shared an image across materials and resized it to 1024-square on frame 2.
The SSS diagnostic ranged from -0.2314453125 to 0.223388671875, with 314 negative
and 338 positive pixels and zero nonzero background pixels.

The original character was opened with auto-execution disabled. A separate copy
retained its material graphs and images, disabled extra passes and requested HWRT/RR.
Its small diagnostic render still reported ``requested RR; actual Native``, naming
the unsupported objects/materials and shader roots. The source file's SHA-256 was
unchanged. This is a verified **failed material acceptance**, not an RR success.
Evidence is saved locally under ``E:/gitrepo/rr_material_validation/``;
``validation_summary.json`` separates execution tests from failed transparency
semantics and incomplete full-material acceptance. No 300-frame or new-material
F12/viewport quality acceptance is claimed by this foundation run.

HWRT material mip crash repair
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read-only HWRT material arrays must not request ``GPU_TEXTURE_USAGE_SHADER_WRITE``.
That flag selects the compute mip generator, whose restoration of a previously bound
shader does not preserve specialization constants. Updating a material image while a
specialized shader is bound can therefore hit the ``GPU_shader_bind`` assertion
``Shader requires specialization constants but none was passed``.

The arrays now request shader-read usage. Their Vulkan mip chain is generated with
backend blits, preserving the active shader and specialization state. HWRT, mipmaps
and the assertion remain enabled. This addresses the new HWRT caller; it does not
change the general compute mip helper's state-restoration contract.

``GPUVulkanTest.ray_query_material_image_mips`` reproduced the exact assertion with
the previous usage. It passes with read-only usage and verifies both array layers at
the final mip. The viewport fixture's ``NORMAL_TEXTURE`` case additionally updates
and resizes the image during active rendering.


Lightweight mixed tracing (2026-10-11)
-------------------------------------

The production hardware shader no longer contains a generic node VM, expression
SSBO, register file or Noise evaluator. CPU graph resolution handles node groups,
reroutes and muted links. Each accepted parameter is a constant or one image sample
with fixed affine channel/color and UV transforms. Supported operations include
Multiply color, scalar Add/Subtract/Multiply/Divide, RGB separation, constant Mapping,
and socket conversions. Intermediate clamps, multiple image inputs and other nodes
select native shading with their node path. Shared original-resolution image arrays,
independent UV/tangent attributes and the read-only mip-generation fix remain.
Secondary texture sampling currently uses LOD zero; footprint-based LOD is not added.

Principled base/metal/roughness/emission/alpha, Diffuse, Emission and Transparent
closures are supported within the existing PBR parameter limits. A direct surface /
Transparent Mix is accepted; general Add/Mix graphs use native shading. IOR/specular
level must remain constant, Specular Tint white, and secondary coat, sheen, SSS,
transmission, anisotropy and arbitrary normal expressions remain native. These limits
apply to secondary shading, not to the original EEVEE primary material graph.

Geometry inclusion is independent of material shading support. Non-opaque geometry
stays in BLAS. Hardware tracing checks the raster surface and secondary hits, then
leaves unresolved rays for the existing screen/probe pass. It never skips an unsupported
surface and continues into geometry behind it. Off-screen complex materials deliberately
retain probe/environment approximations; complete off-screen glass/SSS is not claimed.
Lightweight transparent paths repeatedly query the nearest surface, up to 64 crossings.
Radiance uses weighted surface contributions and RGB continuation; Dithered coverage
uses stochastic continuation with probability compensation. Shadow and irradiance-cache
visibility also retain RGB transmission. Unknown visibility and exhausted budgets use
native/probe fallback instead of a fabricated opaque/transparent material.

RR consumes the complete input-resolution HDR composition, including native SSS,
refraction and transparency. Before-transparency color uses the same format and row
orientation. Separate-layer inputs are empty for this contract; supplying both modes
is rejected. There is no SR pass or transparent draw after RR. SSS scattering remains
active. Native/probe hit distances are not presented as hardware reflection distances.
Viewport layered-material updates and camera motion conservatively reset reconstruction
history; local transparent motion conflict masks are not relied on as verified SDK inputs.
This conservative policy can reduce temporal stability during motion.

``RR active; mixed tracing`` reports local screen/probe fallbacks with object/material/
node details. ``--debug-gpu`` additionally reports per-closure routing counts and compares
raw RR-owned ray radiance against the final denoise result. ``max_error=0`` verifies actual
bypass rather than just a mode flag. These readbacks explicitly request HOST_READ and
are diagnostic overhead, not representative performance measurements.

Initial validation on RTX 4070 SUPER / driver 610.88 / NGX 310.4.0:

* Ten selected GPU/render-graph tests passed; the opt-in independent transparency-layer
  experiment was skipped. This includes the specialized-shader mip regression.
* PBR, mixed PBR/SSS/glass/Noise/transparency, independent-UV Normal Map, Dithered,
  colored transparency, SSS and glass rendered 21 background frames in order 1,2,1.
  All 672 RR evaluations succeeded and repeated frame 1 outputs were bit-identical.
  Selected logs contained no Vulkan validation errors. PNG/linear EXR were finite,
  320x240, with opaque alpha.
* The supplied character rendered through actual RR at 320x426 with its original
  material graphs and images. Seven materials reported local fallback. Both active
  hardware closure sets had exactly zero denoise-bypass error. The original file's
  SHA-256 matched before/after this test; only a separate test copy was saved.

Evidence is under ``E:/gitrepo/rr_material_validation/``. The full secondary-material
proposal is intentionally not implemented. API success is distinct from motion/visual
acceptance and per-stage performance measurements; consult the latest validation report
for the completed runs and remaining limitations.

Latest mixed-path validation and open failures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The local ``mixed_rr`` Release install (validation build 16) is experimental.
It has **not passed the complete numerical and visual acceptance criteria**.
The older full-material proposal above is superseded, not silently enabled.

* The latest 300-frame mixed-material sequence completed all 9,600 expected RR
  evaluations, with zero measured hardware-denoise bypass error and no selected
  Vulkan validation/assertion failures. Camera motion, mesh deformation and alpha
  changes are included. This establishes execution/lifetime coverage, not visual
  approval of every frame.
* The supplied character completed actual viewport RR and a 1584x2112, 128-sample
  background RR render. Seven materials retained local native/probe fallbacks.
  The saved acceptance copy retains original graphs and textures; the source file
  hash was checked before/after the copy/render test and remained unchanged.
  PNG and linear EXR are finite and keep the requested dimensions.
* Latest F12 completion, cancellation/recovery and denoising configurations NONE,
  SPATIAL and TEMPORAL completed. Ten selected Vulkan tests passed, with the optional
  separate-transparency-layer experiment skipped. Missing RR DLL selected SR;
  OpenGL and extra Depth pass selected native. Injected NGX failure discarded the
  reconstruction and rerendered natively once.
* Independent-UV Normal Map, image array update/resize, Blended and SSS SR regressions
  completed. The affine parameter fixture and its constant reference are bit-identical.

Forward/reverse 300-frame outputs from validation build 14 differ by up to 2.515625
in linear HDR channels; isolated frame 150 differs from the forward run by 0.328125.
Both exceed the configured 0.0001 absolute tolerance. Repeating a frame inside one
process can be bit-identical, which is insufficient to certify farm/order isolation.
Build 16 targeted reruns still fail (maximum 1.046875). First-sample RR inputs were
identical in the investigated pair; later native lighting/SSS inputs differed slightly
and RR output differences were larger. The root cause remains unresolved.

During that investigation, unused denoise-neighbor tiles were found to be skipped by
RR passthrough. Clearing now precedes passthrough; invalid histories are initialized,
and RR ownership markers survive temporal denoising without bilateral filtering.
These corrections do not constitute a fix for the remaining order dependence.
``BLENDER_DLSS_DUMP_SAMPLE`` selects the diagnostic sample (default 1).

Visual inspection of the character against same-sample native output found noticeable
transparent-cloth grain and ground-shadow differences. Native effects are present,
but preservation of their appearance is not yet accepted. The exact matched-screen-hit
branch, isolated colored-shadow energy and all 300 frames still need dedicated visual
and numerical review. Secondary texture LOD remains zero, and conservative layered
history resets can limit viewport convergence.

At 1280x720 / 32 samples, measured NGX steady GPU times were 1.068 ms per RR sample
and 0.418 ms per SR sample. Whole render/readback/file-export times were 1.630 s RR,
1.523 s SR and 0.773 s native EEVEE with HWRT in this fixture. These are different
measurements and show no overall RR speedup here. Full per-stage GPU profiling and
controlled cold-cache compilation/peak-memory comparison remain outstanding.

``tests/python/eevee_dlss_mixed_validate.py`` writes a machine-readable FAIL report
for the unresolved numerical comparisons, even when every NGX call succeeds.
Local evidence, a binary hash manifest and the detailed report are saved in
``E:/gitrepo/rr_material_validation/lightweight_validation.md``.

Legacy pass compatibility repair (2026-10-12)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Saved Cycles-only Diffuse Indirect, Glossy Indirect and Transmission flags can remain
enabled while EEVEE's View Layer panel visibly shows only Combined. The reconstruction
gate previously tested every raw legacy bit and incorrectly disabled final SR/RR.
It now uses Film's resolved output passes, after obsolete-pass filtering and before
adding reconstruction's internal motion vectors. Ignored flags are preserved in the
file; actual EEVEE data passes still require native rendering.

``eevee_dlss_pass_compatibility_test.py`` runs the hidden-flag fixture in SR and RR,
plus real Depth and EEVEE Transparent output variants. All six cases passed, each
rendering frames 1,2,1. Hidden flags permit 96 successful evaluations per mode;
Depth/Transparent variants perform zero NGX evaluations and name the actual blocker.
The character acceptance copy with the same five hidden flags also completed 32 SR
and 32 RR evaluations. This repairs eligibility, not the earlier visual/order failures.
Viewport and final-render reconstruction switches remain independent.
